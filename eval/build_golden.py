"""Build the 250-question golden set (blueprint Gate 3: 25/25/20/15/15 stratification).

Counts are exact: 62 factoid, 63 complex_reasoning, 50 multi_hop, 38 summary, 37 negative.
Generation: claude-haiku-5-5 (GOLDEN_GEN_MODEL). Validation: claude-sonnet-5-5 (GOLDEN_JUDGE_MODEL).
Every accepted answerable item was judged supported by its gold chunks; every negative was judged
unanswerable against the tenant's WHOLE vault. Rejected items are logged to eval/golden/build_log.jsonl.

Run:  python eval/build_golden.py            (writes eval/golden/golden.jsonl)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from _llm import USAGE, ask, parse_json  # noqa: E402
from assistant.rag.corpus import Chunk, load_vault, note_links  # noqa: E402
from assistant.security import is_poisoned  # noqa: E402

GEN_MODEL = os.environ.get("GOLDEN_GEN_MODEL", "claude-haiku-5-5")
JUDGE_MODEL = os.environ.get("GOLDEN_JUDGE_MODEL", "claude-sonnet-5-5")
OUT_DIR = ROOT / "eval" / "golden"

# category -> {tenant: quota}. Totals: 62 / 63 / 50 / 38 / 37 = 250.
QUOTAS = {
    "factoid": {"tenant_a": 54, "tenant_b": 8},
    "complex_reasoning": {"tenant_a": 59, "tenant_b": 4},
    "multi_hop": {"tenant_a": 46, "tenant_b": 4},
    "summary": {"tenant_a": 35, "tenant_b": 3},
    "negative": {"tenant_a": 32, "tenant_b": 5},
}
PREFIX = {"factoid": "fact", "complex_reasoning": "cplx", "multi_hop": "hop", "summary": "summ", "negative": "neg"}

GEN_SYSTEM = """You write evaluation questions for a retrieval-augmented operations assistant.
The assistant answers from a vault of short policy, SOP, safety and catalog notes (synthetic demo data).
Rules for every question you write:
- It must be understandable on its own to a user who can search the vault. Name the SKU, lot, procedure or
  plant explicitly. Never say "the passage", "the chunk", "this document" or "the text".
- Use exact IDs, numbers and protocol references (SKU-099, LOT-00044, PROT-SAF-7) where they matter.
- The gold answer uses ONLY facts in the given passages. Never add outside knowledge.
Reply with one JSON object and nothing else."""

JUDGE_SYSTEM = """You are a strict reviewer of evaluation items for a retrieval-augmented assistant.
You are given a question, a gold answer and the passages the gold answer is meant to come from.
Reply with one JSON object and nothing else."""


def norm_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9-]+", text.lower()))


def too_similar(question: str, existing: list[str], threshold: float = 0.7) -> bool:
    a = norm_tokens(question)
    for other in existing:
        b = norm_tokens(other)
        if a and b and len(a & b) / len(a | b) >= threshold:
            return True
    return False


def labeled(chunks: list[Chunk], prefix: str = "C") -> str:
    return "\n\n".join(f"[{prefix}{i}] ({c.note_id})\n{c.text}" for i, c in enumerate(chunks, 1))


class Source:
    """What one generation task is built from."""

    def __init__(self, category: str, tenant: str, chunks: list[Chunk], note_ids: list[str], extra: dict | None = None):
        self.category, self.tenant, self.chunks, self.note_ids, self.extra = category, tenant, chunks, note_ids, extra or {}


def build_pools(load) -> dict[str, dict[str, list[Source]]]:
    by_note: dict[str, list[Chunk]] = {}
    for c in load.chunks:
        by_note.setdefault(c.note_id, []).append(c)
    pools: dict[str, dict[str, list[Source]]] = {k: {} for k in QUOTAS}
    for note_id, chunks in by_note.items():
        tenant = chunks[0].tenant_id
        pools["factoid"].setdefault(tenant, []).extend(Source("factoid", tenant, [c], [note_id]) for c in chunks)
        # complex reasoning: single chunks with numbers, plus consecutive pairs
        for c in chunks:
            if re.search(r"\d", c.text):
                pools["complex_reasoning"].setdefault(tenant, []).append(Source("complex_reasoning", tenant, [c], [note_id]))
        for i in range(len(chunks) - 1):
            pools["complex_reasoning"].setdefault(tenant, []).append(
                Source("complex_reasoning", tenant, chunks[i:i + 2], [note_id]))
        if len(chunks) >= 2:
            pools["summary"].setdefault(tenant, []).append(Source("summary", tenant, chunks, [note_id]))
    for note_id, note in load.notes.items():
        for target in note_links(note):
            other = load.notes.get(target)
            if other is None or other.tenant_id != note.tenant_id or note_id not in by_note or target not in by_note:
                continue
            pools["multi_hop"].setdefault(note.tenant_id, []).append(Source(
                "multi_hop", note.tenant_id, by_note[note_id] + by_note[target], [note_id, target],
                {"split": len(by_note[note_id])}))
    for tenant in ("tenant_a", "tenant_b"):
        other_tenant = "tenant_b" if tenant == "tenant_a" else "tenant_a"
        near = [Source("negative", tenant, by_note[n][:2], [n], {"style": "near_miss"})
                for n, ch in by_note.items() if ch[0].tenant_id == tenant]
        cross = [Source("negative", tenant, by_note[n][:2], [n], {"style": "cross_tenant"})
                 for n, ch in by_note.items() if ch[0].tenant_id == other_tenant]
        pools["negative"][tenant] = near + cross
    return pools


def vault_text(load, tenant: str) -> str:
    return "\n\n".join(f"[{c.note_id}] {c.text}" for c in load.chunks if c.tenant_id == tenant)


def gen_prompt(src: Source, prior: list[str]) -> str:
    avoid = ("\nQuestions already written for this material (write a clearly different one):\n- " + "\n- ".join(prior[-6:])
             if prior else "")
    if src.category == "factoid":
        return (f"Passage:\n{labeled(src.chunks)}\n{avoid}\n\nWrite ONE factoid question answered by a single fact in the passage. "
                'JSON: {"question": "...", "answer": "one or two sentences", "chunks_used": ["C1"]}')
    if src.category == "complex_reasoning":
        return (f"Passages:\n{labeled(src.chunks)}\n{avoid}\n\nWrite ONE question that needs reasoning over these passages: a "
                "scenario with concrete numbers where a rule must be applied, a comparison, a calculation or a conditional "
                "decision. The answer states the conclusion and the rule or numbers it follows from. "
                'JSON: {"question": "...", "answer": "...", "chunks_used": ["C1"]}')
    if src.category == "multi_hop":
        names = " and ".join(src.note_ids)
        return (f"Passages from two linked notes ({names}):\n{labeled(src.chunks)}\n{avoid}\n\nWrite ONE question that can only be "
                "answered by combining at least one fact from EACH of the two notes. The answer gives both facts. "
                'JSON: {"question": "...", "answer": "...", "chunks_used": ["C1", "C3"]}  (labels of the passages you used, '
                "at least one from each note)")
    if src.category == "summary":
        return (f"Note ({src.note_ids[0]}), all its passages:\n{labeled(src.chunks)}\n{avoid}\n\nWrite ONE question asking for a "
                "summary or overview of this note's rules (not a single fact). The answer is 2-4 sentences covering every passage. "
                'JSON: {"question": "...", "answer": "...", "chunks_used": ["C1", "C2"]}')
    raise ValueError(src.category)


def neg_prompt(src: Source, tenant_digest: str, prior: list[str]) -> str:
    avoid = ("\nNegative questions already written (be different):\n- " + "\n- ".join(prior[-10:]) if prior else "")
    if src.extra["style"] == "cross_tenant":
        return (f"A user of {src.tenant} asks about something that exists only at ANOTHER company's plant. Material from that other "
                f"plant:\n{labeled(src.chunks)}\n{avoid}\n\nWrite ONE natural question about a detail from that material, phrased as a "
                f"{src.tenant} user would ask it (do not mention that it is another tenant). The vault of {src.tenant} does not contain "
                'this. JSON: {"question": "...", "answer": ""}')
    return (f"Digest of the vault for {src.tenant}:\n{tenant_digest}\n\nFocus note:\n{labeled(src.chunks)}\n{avoid}\n\nWrite ONE plausible "
            "question a user might ask about the topic of the focus note, but about a specific detail the vault does NOT state "
            "(for example a number, time limit, owner, SKU or protocol that is not in the vault). It should be a near miss: on topic, "
            'not answerable from the vault. Do not ask about weather, news or other unrelated subjects. JSON: {"question": "...", "answer": ""}')


def judge_prompt(item: dict, chunks: list[Chunk], tenant_vault: str | None) -> str:
    if item["category"] == "negative":
        return (f"Vault of {item['tenant_id']} (everything the assistant may use):\n{tenant_vault}\n\nQuestion: {item['question']}\n\n"
                "Can this question be answered, even partly, from the vault above? "
                'JSON: {"answerable": true|false, "partially_answerable": true|false, "reason": "short"}')
    extra = ""
    if item["category"] == "multi_hop":
        extra = ' "needs_both_notes": true|false (the answer needs facts from at least two different notes),'
    return (f"Passages:\n{labeled(chunks)}\n\nQuestion: {item['question']}\nGold answer: {item['gold_answer']}\n\n"
            "Check the item. JSON: {\"supported\": true|false (every claim in the gold answer is stated or directly derivable from the "
            "passages), \"answers_question\": true|false, \"self_contained\": true|false (question is understandable without seeing "
            f"the passages, and does not refer to them),{extra} \"reason\": \"short\"}}")


def run_task(src: Source, load, prior: list[str], prior_all: list[str], tenant_vaults: dict[str, str], digests: dict[str, str]) -> tuple[dict | None, dict]:
    log = {"category": src.category, "tenant_id": src.tenant, "notes": src.note_ids}
    try:
        prompt = neg_prompt(src, digests[src.tenant], prior) if src.category == "negative" else gen_prompt(src, prior)
        text, _ = ask(GEN_MODEL, GEN_SYSTEM, prompt, max_tokens=600, temperature=0.8)
        data = parse_json(text)
        if not isinstance(data, dict) or not str(data.get("question", "")).strip():
            return None, {**log, "rejected": "unreadable generation"}
        question = str(data["question"]).strip()
        answer = str(data.get("answer", "")).strip()
        if is_poisoned(question):
            return None, {**log, "rejected": "question flagged by injection shield", "question": question}
        if too_similar(question, prior_all):
            return None, {**log, "rejected": "near duplicate", "question": question}
        if src.category == "negative":
            used = []
        else:
            labels = data.get("chunks_used") or []
            idx = sorted({int(m) - 1 for lab in labels for m in re.findall(r"\d+", str(lab)) if 0 < int(m) <= len(src.chunks)})
            if not idx:
                return None, {**log, "rejected": "no chunks_used", "question": question}
            if src.category == "summary":
                idx = list(range(len(src.chunks)))
            if src.category == "multi_hop":
                split = src.extra["split"]
                if not (any(i < split for i in idx) and any(i >= split for i in idx)):
                    return None, {**log, "rejected": "multi-hop used only one note", "question": question}
            used = [src.chunks[i] for i in idx]
        item = {"category": src.category, "tenant_id": src.tenant, "question": question, "gold_answer": answer,
                "gold_chunk_hashes": [c.chunk_hash for c in used]}
        jtext, _ = ask(JUDGE_MODEL, JUDGE_SYSTEM, judge_prompt(item, used, tenant_vaults[src.tenant]), max_tokens=300)
        verdict = parse_json(jtext)
        if not isinstance(verdict, dict):
            return None, {**log, "rejected": "unreadable judge reply", "question": question}
        if src.category == "negative":
            ok = verdict.get("answerable") is False and verdict.get("partially_answerable") is False
        else:
            ok = verdict.get("supported") is True and verdict.get("answers_question") is True and verdict.get("self_contained") is True
            if src.category == "multi_hop":
                ok = ok and verdict.get("needs_both_notes") is True
        if not ok:
            return None, {**log, "rejected": "judge", "question": question, "verdict": verdict}
        item["notes"] = {"source_notes": src.note_ids, "style": src.extra.get("style"), "generator": GEN_MODEL,
                         "judge": JUDGE_MODEL, "judge_reason": verdict.get("reason", "")}
        return item, {**log, "accepted": True, "question": question}
    except Exception as err:   # noqa: BLE001 - one failed task must not stop the build
        return None, {**log, "rejected": f"error {type(err).__name__}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    load = load_vault(ROOT / "knowledge_vault", is_poisoned)
    assert not load.quarantined, "vault has quarantined chunks"
    pools = build_pools(load)
    tenant_vaults = {t: vault_text(load, t) for t in ("tenant_a", "tenant_b")}
    digests = {t: "\n".join(f"- {n.note_id}: {n.body.split(chr(10) + chr(10))[0][:140]}" for n in load.notes.values() if n.tenant_id == t)
               for t in ("tenant_a", "tenant_b")}

    accepted: dict[tuple[str, str], list[dict]] = {}
    log_lines: list[dict] = []
    cursors: dict[tuple[str, str], int] = {}
    per_source_prior: dict[tuple, list[str]] = {}
    lock = threading.Lock()
    shuffled = {cat: {t: rng.sample(v, len(v)) for t, v in by_t.items()} for cat, by_t in pools.items()}

    for round_no in range(1, args.max_rounds + 1):
        tasks = []
        for cat, by_t in QUOTAS.items():
            for tenant, quota in by_t.items():
                have = len(accepted.get((cat, tenant), []))
                need = quota - have
                if need <= 0:
                    continue
                pool = shuffled[cat][tenant]
                for _ in range(need + max(2, need // 2)):   # over-generate to absorb judge rejections
                    i = cursors.get((cat, tenant), 0)
                    cursors[(cat, tenant)] = i + 1
                    tasks.append(pool[i % len(pool)])
        if not tasks:
            break
        print(f"round {round_no}: {len(tasks)} tasks", flush=True)
        all_questions: dict[str, list[str]] = {}
        for items in accepted.values():
            for it in items:
                all_questions.setdefault(it["category"], []).append(it["question"])

        def one(src: Source):
            key = (src.category, src.tenant, tuple(src.note_ids), tuple(c.chunk_hash for c in src.chunks))
            with lock:
                prior_src = list(per_source_prior.get(key, []))
                prior_all = list(all_questions.get(src.category, []))
                if src.category == "negative":
                    prior_src = prior_all[-10:]
            item, log = run_task(src, load, prior_src, prior_all, tenant_vaults, digests)
            with lock:
                log_lines.append(log)
                if item is not None:
                    bucket = accepted.setdefault((item["category"], item["tenant_id"]), [])
                    cross_dupe = too_similar(item["question"], [x["question"] for v in accepted.values() for x in v])
                    if len(bucket) < QUOTAS[item["category"]][item["tenant_id"]] and not cross_dupe:
                        bucket.append(item)
                        per_source_prior.setdefault(key, []).append(item["question"])
                    else:
                        log["rejected"] = "quota full or duplicate"
                        log.pop("accepted", None)
            return None

        with ThreadPoolExecutor(args.workers) as pool_ex:
            for fut in as_completed([pool_ex.submit(one, s) for s in tasks]):
                fut.result()
        counts = {f"{c}/{t}": f"{len(accepted.get((c, t), []))}/{q}" for c, bt in QUOTAS.items() for t, q in bt.items()}
        print("  ", counts, flush=True)

    # assemble final file in a stable order
    rows: list[dict] = []
    for cat in QUOTAS:
        n = 0
        for tenant in QUOTAS[cat]:
            for item in accepted.get((cat, tenant), []):
                n += 1
                rows.append({"id": f"{PREFIX[cat]}-{n:03d}", **{k: item[k] for k in (
                    "category", "tenant_id", "question", "gold_answer", "gold_chunk_hashes", "notes")}})
    short = {f"{c}/{t}": (len(accepted.get((c, t), [])), q) for c, bt in QUOTAS.items() for t, q in bt.items()
             if len(accepted.get((c, t), [])) != q}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "golden.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (OUT_DIR / "build_log.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in log_lines) + "\n", encoding="utf-8")
    cat_counts = Counter(r["category"] for r in rows)
    rejects = Counter(l.get("rejected", "accepted") if not l.get("accepted") else "accepted" for l in log_lines)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "total": len(rows),
               "by_category": dict(cat_counts), "by_tenant": dict(Counter(r["tenant_id"] for r in rows)),
               "generator": GEN_MODEL, "judge": JUDGE_MODEL, "short_of_quota": short, "task_outcomes": dict(rejects),
               "usage": USAGE.by_model, "seed": args.seed}
    (OUT_DIR / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if short:
        sys.exit(f"did not reach quota: {short}")


if __name__ == "__main__":
    main()
