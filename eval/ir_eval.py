"""Deterministic, LLM-free retrieval evaluation over the golden set -> eval/runs/ir.json.

KPI-07/08/09 (stage latency), KPI-10 (recall@50), KPI-11 (precision@4), KPI-12 (MRR@4), KPI-13 (nDCG@4),
KPI-23 (retrieval-level abstention on negatives) and the false-abstain rate on answerable questions.

Definitions (binary relevance = chunk hash is in gold_chunk_hashes):
- recall@50: share of gold chunks inside the union of BM25 top-50 and dense top-50 candidates.
  NOTE: the vault has fewer than 50 chunks per tenant, so the dense top-50 is the whole tenant corpus and this is
  trivially 1.0. It says nothing about retrieval quality at scale.
- precision@4: RAGAS-style rank-weighted context precision over the reranked top 4 =
  sum_k(precision@k * rel_k) / (relevant chunks in top 4); 0 when none. Reported ranking-only (primary) and with
  abstained questions scored 0 (`*_abstain_zero`, the ADR-007 convention).
- MRR@4, nDCG@4: binary gains, ideal DCG uses min(|gold|, 4) hits.

Run:  python eval/ir_eval.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.config import load_settings  # noqa: E402
from assistant.rag.index import Models, build_index, tokenize  # noqa: E402

GOLDEN = ROOT / "eval" / "golden" / "golden.jsonl"
OUT = ROOT / "eval" / "runs" / "ir.json"
K = 4


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_weighted_precision(ranked_hashes: list[str], gold: set[str], k: int = K) -> float:
    top = ranked_hashes[:k]
    hits, total = 0, 0.0
    for i, h in enumerate(top, 1):
        if h in gold:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def plain_precision(ranked_hashes: list[str], gold: set[str], k: int = K) -> float:
    top = ranked_hashes[:k]
    return sum(h in gold for h in top) / len(top) if top else 0.0


def reciprocal_rank(ranked_hashes: list[str], gold: set[str], k: int = K) -> float:
    for i, h in enumerate(ranked_hashes[:k], 1):
        if h in gold:
            return 1.0 / i
    return 0.0


def ndcg(ranked_hashes: list[str], gold: set[str], k: int = K) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, h in enumerate(ranked_hashes[:k], 1) if h in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def pct(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 2) if values else 0.0


def mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def main() -> None:
    settings = load_settings()
    rows = load_golden()
    models = Models(settings.dense_model, settings.rerank_model)
    index = build_index(settings.vault_dir, models)
    for q in ("warm up query about lots", "shelf life", "temperature"):   # exclude one-off model warm-up from latency
        index.retrieve("tenant_a", q)

    answerable = [r for r in rows if r["category"] != "negative"]
    negatives = [r for r in rows if r["category"] == "negative"]
    per_row: list[dict] = []
    lat: dict[str, list[float]] = {"bm25": [], "dense": [], "rrf": [], "rerank": []}

    for r in rows:
        tenant, query = r["tenant_id"], r["question"]
        out = index.retrieve(tenant, query, bm25_top_k=settings.bm25_top_k, dense_top_k=settings.dense_top_k,
                             rrf_k=settings.rrf_k, rerank_k=settings.rerank_k,
                             min_rerank_score=settings.min_rerank_score, pool=settings.rrf_pool,
                             rerank_pool=settings.rerank_pool)
        for stage, ms in out["stage_latency_ms"].items():
            lat[stage].append(ms)
        ranked = [c["chunk_hash"] for c in out["chunks"]]
        ti = index.tenants[tenant]
        bm = ti.bm25.get_scores(tokenize(query))
        bm_top = {int(i) for i in np.argsort(-bm)[:settings.bm25_top_k] if bm[i] > 0}
        qv = models.embed([query], query=True)
        _, ids = ti.hnsw.search(qv, min(settings.dense_top_k, len(ti.chunks)))
        cand = bm_top | {int(i) for i in ids[0] if i >= 0}
        cand_hashes = {ti.chunks[i].chunk_hash for i in cand}
        gold = set(r["gold_chunk_hashes"])
        row = {"id": r["id"], "category": r["category"], "abstain": out["abstain"], "top_score": out["top_score"]}
        if r["category"] != "negative":
            row.update(recall50=len(gold & cand_hashes) / len(gold), precision=rank_weighted_precision(ranked, gold),
                       plain_precision=plain_precision(ranked, gold), rr=reciprocal_rank(ranked, gold),
                       ndcg=ndcg(ranked, gold), recall4=len(gold & set(ranked)) / len(gold))
        per_row.append(row)

    ans = [p for p in per_row if p["category"] != "negative"]
    neg = [p for p in per_row if p["category"] == "negative"]

    def zero(p: dict, key: str) -> float:
        return 0.0 if p["abstain"] else p[key]

    by_cat = {}
    for cat in sorted({p["category"] for p in ans}):
        cp = [p for p in ans if p["category"] == cat]
        by_cat[cat] = {"n": len(cp), "recall_at_50": mean([p["recall50"] for p in cp]),
                       "precision_at_4": mean([p["precision"] for p in cp]), "mrr_at_4": mean([p["rr"] for p in cp]),
                       "ndcg_at_4": mean([p["ndcg"] for p in cp]), "recall_at_4": mean([p["recall4"] for p in cp]),
                       "false_abstain_rate": mean([float(p["abstain"]) for p in cp])}
    sweep = {}
    for t in (0.1, 0.2, 0.3, 0.42, 0.5, 0.6, 0.7, 0.8, 0.9):
        sweep[str(t)] = {"negative_abstention": mean([float(p["top_score"] < t) for p in neg]),
                         "answerable_false_abstain": mean([float(p["top_score"] < t) for p in ans])}
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_answerable": len(ans), "n_negative": len(neg),
        "recall_at_50": mean([p["recall50"] for p in ans]),
        "precision_at_4": mean([p["precision"] for p in ans]),
        "mrr_at_4": mean([p["rr"] for p in ans]),
        "ndcg_at_4": mean([p["ndcg"] for p in ans]),
        "abstention_accuracy_negative": mean([float(p["abstain"]) for p in neg]),
        "false_abstain_rate_answerable": mean([float(p["abstain"]) for p in ans]),
        "precision_at_4_abstain_zero": mean([zero(p, "precision") for p in ans]),
        "mrr_at_4_abstain_zero": mean([zero(p, "rr") for p in ans]),
        "ndcg_at_4_abstain_zero": mean([zero(p, "ndcg") for p in ans]),
        "plain_precision_at_4": mean([p["plain_precision"] for p in ans]),
        "recall_at_4": mean([p["recall4"] for p in ans]),
        "latency_ms": {s: {"p50": pct(v, 50), "p95": pct(v, 95)} for s, v in lat.items()},
        "per_category": by_cat,
        "threshold_sweep": sweep,
        "hardware": {"cpus": os.cpu_count(), "device": "cpu",
                     "runtime": "sentence-transformers / PyTorch CPU (not ONNX Runtime)"},
        "config": {"bm25_top_k": settings.bm25_top_k, "dense_top_k": settings.dense_top_k, "rrf_k": settings.rrf_k,
                   "rrf_pool": settings.rrf_pool, "rerank_pool": settings.rerank_pool, "rerank_k": settings.rerank_k,
                   "min_rerank_score": settings.min_rerank_score, "dense_model": settings.dense_model,
                   "rerank_model": settings.rerank_model, "chunks_per_tenant":
                   {t: len(i.chunks) for t, i in index.tenants.items()}},
        "notes": [
            "recall@50 is trivially 1.0 when a tenant has fewer than 50 chunks: dense top-50 returns the whole corpus.",
            "precision/MRR/nDCG are ranking-only (abstention ignored); *_abstain_zero scores abstained questions 0 (ADR-007).",
            "Latency is single-query, sequential, on this CPU. KPI-07/08/09 targets assume target hardware (ADR-006/020).",
            "The golden set was generated by an LLM from this vault (see eval/golden/README.md); it is not independent.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_category", "threshold_sweep", "config", "notes")}, indent=2))


if __name__ == "__main__":
    main()
