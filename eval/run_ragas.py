"""RAGAS-style generation evaluation (ADR-007) -> eval/runs/ragas.json and eval/runs/answers.jsonl.

Default: deterministic stratified sample of 50 golden questions (seed 7). `--limit 0` runs all rows.

Pipeline per question (mirrors the product path, minus the HTTP layer):
  index.retrieve (abstain < 0.42 -> deterministic abstention, generator skipped)
  -> graph subgraph -> assistant.prompt.assemble -> claude generator (temperature 0)
  -> assistant.verify lexical check -> release rule (all sentences grounded: the draft;
     otherwise the verbatim top chunk, the non-streaming veto path of ADR-013).

Metrics
- faithfulness: LLM judge splits the DRAFT into atomic claims and marks each supported by the retrieved context
  (stated or directly derivable). Per-question score = supported/total; mean over drafts with claims.
- answer_relevancy: strictness 1. The judge writes the question the RELEASED answer responds to; score is the
  cosine (bge-large) with the real question; noncommittal/abstention answers on answerable questions score 0.
- context_precision / context_recall: ID based with gold chunk hashes; precision is rank weighted
  (sum_k P@k*rel_k / relevant in top 4); an abstained question scores 0 on both.
- context_entity_recall: share of entities in the gold answer (IDs, numbers, acronyms, capitalised terms)
  found in the retrieved context; abstained = 0.
- hallucination_rate_draft: draft sentences vetoed by the lexical verifier / draft claim sentences.
- released_unsupported_rate (KPI-24): the judge lists unsupported claims in the text actually RELEASED.
- abstention_accuracy_negative: negative questions whose released answer is the abstention text.
Judge replies that cannot be parsed are excluded and counted in `judge_failures` (never counted as supported).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from _llm import PRICES_PER_M, USAGE, ask, parse_json  # noqa: E402
from assistant.config import load_settings  # noqa: E402
from assistant.contracts import CallContext, extract_lot_ids, extract_sku_ids  # noqa: E402
from assistant.graph import Graph  # noqa: E402
from assistant.llm import ABSTAIN_TEXT  # noqa: E402
from assistant.prompt import GoalAnchorBlock, assemble  # noqa: E402
from assistant.rag.index import Models, build_index  # noqa: E402
from assistant.verify import check_draft, split_sentences, strip_citations  # noqa: E402
from ir_eval import load_golden, rank_weighted_precision  # noqa: E402

GEN_MODEL = os.environ.get("GENERATOR_MODEL", "claude-haiku-5-5")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-5-5")
OUT = ROOT / "eval" / "runs"
CATEGORIES = ["factoid", "complex_reasoning", "multi_hop", "summary", "negative"]

# Blueprint targets (design 1.5). Reported for reference only; the pass/fail call belongs to eval/kpis.py.
TARGETS = {"faithfulness": 0.92, "answer_relevancy": 0.90, "context_precision": 0.88, "context_recall": 0.85,
           "context_entity_recall": 0.85, "released_unsupported_rate": 0.005, "abstention_accuracy_negative": 0.98}

JUDGE_SYS = "You are a precise evaluator. Reply with one JSON object and nothing else."


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Largest-remainder allocation by category, then a seeded random pick inside each category."""
    if n <= 0 or n >= len(rows):
        return list(rows)
    by_cat = {c: [r for r in rows if r["category"] == c] for c in CATEGORIES}
    exact = {c: n * len(v) / len(rows) for c, v in by_cat.items()}
    alloc = {c: int(x) for c, x in exact.items()}
    for c in sorted(exact, key=lambda c: -(exact[c] - alloc[c]))[: n - sum(alloc.values())]:
        alloc[c] += 1
    rng = random.Random(seed)
    picked: list[dict] = []
    for c in CATEGORIES:
        picked += rng.sample(sorted(by_cat[c], key=lambda r: r["id"]), alloc[c])
    return sorted(picked, key=lambda r: (CATEGORIES.index(r["category"]), r["id"]))


_ID = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")


def extract_entities(text: str) -> list[str]:
    """IDs, numbers, acronyms and capitalised terms that are not sentence-initial. Deduplicated, ordered."""
    found: list[str] = []
    ids = _ID.findall(text)
    found += ids
    rest = _ID.sub(" ", text)
    found += _NUM.findall(rest)
    found += _ACRONYM.findall(rest)
    for m in _CAP.finditer(rest):
        before = rest[: m.start()].rstrip()
        if before and before[-1] not in ".!?:":     # skip sentence-initial words
            found.append(m.group(0))
    seen: list[str] = []
    for e in found:
        if e not in seen:
            seen.append(e)
    return seen


def entity_in(entity: str, context: str) -> bool:
    if _ID.fullmatch(entity) or entity.isupper() or entity[0].isdigit():
        return re.search(r"(?<![\w.-])" + re.escape(entity) + r"(?![\w-]|\.\d)", context) is not None
    return entity.lower() in context.lower()


def entity_recall(gold_answer: str, context: str) -> float | None:
    ents = extract_entities(gold_answer)
    if not ents:
        return None
    return sum(entity_in(e, context) for e in ents) / len(ents)


def context_recall(retrieved: list[str], gold: list[str]) -> float:
    return len(set(gold) & set(retrieved)) / len(gold) if gold else 0.0


def faith_prompt(context: str, answer: str) -> str:
    return (f"Context:\n{context}\n\nAnswer:\n{answer}\n\nSplit the answer into atomic factual claims. For each claim say whether "
            "it is supported: stated in the context, or following directly from it (simple arithmetic or comparison of stated "
            "numbers counts). Ignore citations and polite filler. "
            'JSON: {"claims": [{"claim": "...", "supported": true|false}]}')


def judge_claims(context: str, answer: str) -> tuple[dict | None, dict]:
    text, usage = ask(JUDGE_MODEL, JUDGE_SYS, faith_prompt(context, answer), max_tokens=900)
    data = parse_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        return None, usage
    claims = [c for c in data["claims"] if isinstance(c, dict) and isinstance(c.get("supported"), bool)]
    if len(claims) != len(data["claims"]):
        return None, usage
    return {"total": len(claims), "supported": sum(c["supported"] for c in claims),
            "unsupported": [c.get("claim", "") for c in claims if not c["supported"]]}, usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="0 = all golden rows")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    settings = load_settings()
    rows = stratified_sample(load_golden(), args.limit, args.seed)
    models = Models(settings.dense_model, settings.rerank_model)
    index = build_index(settings.vault_dir, models)
    graph = Graph.from_notes(index.load.notes)
    judge_usage = {"input": 0, "output": 0, "calls": 0}
    judge_failures = {"faithfulness": 0, "relevancy": 0, "released": 0}
    notes: list[str] = []

    # ---- stage A: retrieval and prompt assembly (sequential, CPU models) -------------------------------------
    recs: list[dict] = []
    for r in rows:
        tenant, q = r["tenant_id"], r["question"]
        out = index.retrieve(tenant, q, bm25_top_k=settings.bm25_top_k, dense_top_k=settings.dense_top_k,
                             rrf_k=settings.rrf_k, rerank_k=settings.rerank_k, min_rerank_score=settings.min_rerank_score,
                             pool=settings.rrf_pool, rerank_pool=settings.rerank_pool)
        chunks = out["chunks"]
        rec = {"row": r, "out": out, "chunks": chunks, "abstain_retrieval": out["abstain"], "prompt": None}
        if not out["abstain"]:
            anchors: list[str] = []
            for c in chunks:
                if c["note_id"] not in anchors and len(anchors) < 4:
                    anchors.append(c["note_id"])
            ctx = CallContext(tenant_id=tenant, user_id="user_eval", session_id=uuid.uuid4(), idempotency_key=uuid.uuid4(),
                              traceparent="00-" + uuid.uuid4().hex + "-" + uuid.uuid4().hex[:16] + "-01")
            try:
                sub = graph.subgraph(ctx, anchors, 2, 320)
            except Exception:   # noqa: BLE001 - FMEA #7: no subgraph, still grounded in chunks
                sub = None
            anchor = GoalAnchorBlock(intent="policy_qa", query=q, entities=tuple(extract_lot_ids(q) + extract_sku_ids(q)))
            rec["prompt"] = assemble(anchor, sub, chunks, observations=[], turns=[])
        recs.append(rec)

    # ---- stage B: generation (parallel) -----------------------------------------------------------------------
    gen_tokens = {"input": 0, "output": 0, "calls": 0}

    def generate(rec: dict) -> None:
        p = rec["prompt"]
        if p is None:
            rec["draft"] = ABSTAIN_TEXT
            return
        try:
            text, u = ask(GEN_MODEL, p.system, p.user, max_tokens=400, temperature=0.0)
            rec["draft"] = text
            rec["gen_usage"] = u
        except Exception as err:   # noqa: BLE001 - FMEA #4: extractive fallback, counted
            rec["draft"] = None
            rec["gen_error"] = type(err).__name__

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(generate, recs))
    for rec in recs:
        if "gen_usage" in rec:
            gen_tokens["input"] += rec["gen_usage"]["input"]
            gen_tokens["output"] += rec["gen_usage"]["output"]
            gen_tokens["calls"] += 1

    # ---- verification and release rule ------------------------------------------------------------------------
    for rec in recs:
        chunks = rec["chunks"]
        retrieved = {c["chunk_hash"]: c["text"] for c in chunks}
        draft = rec["draft"]
        if draft is None:                       # generator down: verbatim cited quote, as in the product
            rec.update(sentences=[], verdicts=[], released=chunks[0]["text"], draft_clean=None, path="generator_error")
            continue
        sentences = split_sentences(draft)
        verdicts = check_draft(sentences, retrieved, ABSTAIN_TEXT)
        rec["sentences"], rec["verdicts"] = sentences, verdicts
        rec["draft_clean"] = " ".join(strip_citations(s) for s in sentences)
        if rec["abstain_retrieval"] or (sentences and all(v.grounded for v in verdicts)):
            rec["released"], rec["path"] = rec["draft_clean"], "released_draft"
        else:
            rec["released"], rec["path"] = (chunks[0]["text"] if chunks else ABSTAIN_TEXT), "vetoed_to_quote"
        rec["abstained_final"] = rec["released"].strip() == ABSTAIN_TEXT

    # ---- stage C: judges (parallel) ---------------------------------------------------------------------------
    def judge(rec: dict) -> None:
        context = "\n\n".join(c["text"] for c in rec["chunks"])
        draft_clean = rec.get("draft_clean")
        if draft_clean and draft_clean.strip() != ABSTAIN_TEXT and not rec["abstain_retrieval"]:
            res, u = judge_claims(context, draft_clean)
            rec["faith"], rec["judge_u1"] = res, u
            if res is None:
                rec["faith_failed"] = True
        released = rec["released"]
        if released.strip() != ABSTAIN_TEXT:
            if released == draft_clean and "faith" in rec and rec["faith"] is not None:
                rec["rel_claims"] = rec["faith"]
            elif released == draft_clean and rec.get("faith_failed"):
                rec["rel_claims"], rec["released_failed"] = None, True
            else:
                res, u = judge_claims(context, released)
                rec["rel_claims"], rec["judge_u2"] = res, u
                if res is None:
                    rec["released_failed"] = True
        if rec["row"]["category"] != "negative":
            if released.strip() == ABSTAIN_TEXT:
                rec["rel_question"] = {"question": "", "noncommittal": True}
            else:
                text, u = ask(JUDGE_MODEL, JUDGE_SYS, f"Answer:\n{released}\n\nWrite the single question this answer responds to, "
                              "and say whether the answer is noncommittal (declines, says it cannot find the information, or evades). "
                              'JSON: {"question": "...", "noncommittal": true|false}', max_tokens=200)
                rec["judge_u3"] = u
                data = parse_json(text)
                if isinstance(data, dict) and isinstance(data.get("question"), str) and isinstance(data.get("noncommittal"), bool):
                    rec["rel_question"] = data
                else:
                    rec["rel_question"] = None
                    rec["relevancy_failed"] = True

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(judge, recs))
    for rec in recs:
        for k in ("judge_u1", "judge_u2", "judge_u3"):
            if k in rec:
                judge_usage["input"] += rec[k]["input"]
                judge_usage["output"] += rec[k]["output"]
                judge_usage["calls"] += 1
        judge_failures["faithfulness"] += bool(rec.get("faith_failed"))
        judge_failures["released"] += bool(rec.get("released_failed"))
        judge_failures["relevancy"] += bool(rec.get("relevancy_failed"))

    # ---- stage D: answer relevancy embeddings (sequential) -----------------------------------------------------
    for rec in recs:
        rq = rec.get("rel_question")
        if rec["row"]["category"] == "negative" or rq is None:
            continue
        if rq["noncommittal"] or not rq["question"].strip():
            rec["relevancy"] = 0.0
        else:
            v = models.embed([rec["row"]["question"], rq["question"]])
            rec["relevancy"] = float(max(0.0, np.dot(v[0], v[1])))

    # ---- scoring -----------------------------------------------------------------------------------------------
    def mean(xs: list[float]) -> float | None:
        return round(float(np.mean(xs)), 4) if xs else None

    for rec in recs:
        r = rec["row"]
        ranked = [c["chunk_hash"] for c in rec["chunks"]]
        gold = r["gold_chunk_hashes"]
        if r["category"] != "negative":
            ab = rec["abstain_retrieval"]
            rec["precision"] = 0.0 if ab else rank_weighted_precision(ranked, set(gold))
            rec["recall"] = 0.0 if ab else context_recall(ranked, gold)
            er = entity_recall(r["gold_answer"], "\n".join(c["text"] for c in rec["chunks"]))
            rec["entity_recall"] = None if er is None else (0.0 if ab else er)

    def agg(sel: list[dict]) -> dict:
        ans = [x for x in sel if x["row"]["category"] != "negative"]
        faiths = [x["faith"]["supported"] / x["faith"]["total"] for x in sel if x.get("faith") and x["faith"]["total"]]
        rel_c = [x["rel_claims"] for x in sel if x.get("rel_claims")]
        tot = sum(c["total"] for c in rel_c)
        neg = [x for x in sel if x["row"]["category"] == "negative"]
        drafts = [x for x in sel if x.get("sentences") and not x["abstain_retrieval"]]
        claim_sents = [(s, v) for x in drafts for s, v in zip(x["sentences"], x["verdicts"]) if strip_citations(s) != ABSTAIN_TEXT]
        return {
            "n": len(sel), "n_answerable": len(ans), "n_negative": len(neg),
            "faithfulness": mean(faiths), "n_faithfulness": len(faiths),
            "answer_relevancy": mean([x["relevancy"] for x in ans if "relevancy" in x]),
            "context_precision": mean([x["precision"] for x in ans]), "context_recall": mean([x["recall"] for x in ans]),
            "context_entity_recall": mean([x["entity_recall"] for x in ans if x.get("entity_recall") is not None]),
            "hallucination_rate_draft": (round(sum(not v.grounded for _, v in claim_sents) / len(claim_sents), 4)
                                         if claim_sents else None),
            "released_unsupported_rate": (round(sum(c["total"] - c["supported"] for c in rel_c) / tot, 4) if tot else None),
            "released_claims": tot,
            "abstention_accuracy_negative": mean([float(x["abstained_final"]) for x in neg]),
            "false_abstain_rate_answerable_e2e": mean([float(x["abstained_final"]) for x in ans]),
            "vetoed_to_quote": sum(x["path"] == "vetoed_to_quote" for x in sel),
        }

    overall = agg(recs)
    per_cat = {c: agg([x for x in recs if x["row"]["category"] == c]) for c in CATEGORIES if any(x["row"]["category"] == c for x in recs)}
    n_q = len(recs)
    price = PRICES_PER_M.get(GEN_MODEL)
    cost_per_1000 = (round((gen_tokens["input"] * price[0] + gen_tokens["output"] * price[1]) / 1e6 / n_q * 1000, 4)
                     if price and n_q else None)
    if args.limit != 0:
        notes.append(f"Stratified sample of {len(recs)} of {len(load_golden())} golden rows (seed {args.seed}); with n this small a "
                     "metric's sampling error is several points, so scores within ~0.05 of a threshold are inconclusive (ADR-007).")
    notes += [
        "Golden questions and gold answers were LLM-generated from this vault and LLM-validated (eval/golden/README.md); not independent.",
        "Faithfulness is the mean over drafts that contain claims; abstained answers contribute no claims. Abstentions on answerable "
        "questions score 0 on relevancy, precision, recall and entity recall.",
        "The verifier is the lexical default (ADR-013), not NLI; the vetoed_to_quote path releases the verbatim top chunk.",
        "Generator call mirrors AnthropicGenerator (temperature 0, max_tokens 400) but goes through eval/_llm.py to capture token usage.",
    ]
    if any(judge_failures.values()):
        notes.append(f"Judge replies that could not be parsed were excluded, not scored as supported: {judge_failures}.")
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n": n_q, "limit": args.limit,
        "faithfulness": overall["faithfulness"], "answer_relevancy": overall["answer_relevancy"],
        "context_entity_recall": overall["context_entity_recall"], "context_precision": overall["context_precision"],
        "context_recall": overall["context_recall"], "hallucination_rate_draft": overall["hallucination_rate_draft"],
        "released_unsupported_rate": overall["released_unsupported_rate"],
        "abstention_accuracy_negative": overall["abstention_accuracy_negative"],
        "false_abstain_rate_answerable_e2e": overall["false_abstain_rate_answerable_e2e"],
        "vetoed_to_quote": overall["vetoed_to_quote"], "released_claims": overall["released_claims"],
        "tokens": gen_tokens, "judge_tokens": judge_usage,
        "generator_cost_usd_per_1000_queries": cost_per_1000,
        "generator_model": GEN_MODEL, "judge_model": JUDGE_MODEL, "judge_failures": judge_failures,
        "generation_failures": sum(1 for x in recs if x.get("gen_error")),
        "targets": TARGETS, "per_category": per_cat, "usage_by_model": USAGE.by_model, "notes": notes,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ragas.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUT / "answers.jsonl").open("w", encoding="utf-8") as fh:
        for x in recs:
            fh.write(json.dumps({
                "id": x["row"]["id"], "category": x["row"]["category"], "tenant_id": x["row"]["tenant_id"],
                "question": x["row"]["question"], "gold_answer": x["row"]["gold_answer"],
                "retrieval_abstain": x["abstain_retrieval"], "top_score": round(x["out"]["top_score"], 4),
                "retrieved": [{"note_id": c["note_id"], "chunk_hash": c["chunk_hash"], "rerank": round(c["rerank_score"], 4)} for c in x["chunks"]],
                "gold_chunk_hashes": x["row"]["gold_chunk_hashes"], "draft": x.get("draft"), "released": x["released"],
                "path": x["path"], "abstained_final": x["abstained_final"],
                "ungrounded_sentences": [s for s, v in zip(x.get("sentences", []), x.get("verdicts", [])) if not v.grounded],
                "faithfulness_claims": x.get("faith"), "released_claims": x.get("rel_claims"), "relevancy": x.get("relevancy"),
                "precision": x.get("precision"), "recall": x.get("recall"), "entity_recall": x.get("entity_recall"),
            }, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_category", "notes", "targets")}, indent=2))


if __name__ == "__main__":
    main()
