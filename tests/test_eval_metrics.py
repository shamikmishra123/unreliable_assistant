"""Unit tests for the evaluation metric functions (eval/ir_eval.py, eval/run_ragas.py). No models, no LLM."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import ir_eval  # noqa: E402
import run_ragas  # noqa: E402
from _llm import parse_json  # noqa: E402


def test_rank_weighted_precision_follows_ragas_definition():
    gold = {"a", "b"}
    assert ir_eval.rank_weighted_precision(["a", "x", "b", "y"], gold) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert ir_eval.rank_weighted_precision(["x", "y", "z", "w"], gold) == 0.0
    assert ir_eval.rank_weighted_precision(["a", "b", "x", "y"], gold) == 1.0
    assert ir_eval.rank_weighted_precision(["x", "a"], {"a"}) == pytest.approx(0.5)
    assert ir_eval.rank_weighted_precision([], gold) == 0.0


def test_plain_precision_is_bounded_by_gold_size():
    assert ir_eval.plain_precision(["a", "x", "y", "z"], {"a"}) == 0.25


def test_mrr_and_ndcg():
    assert ir_eval.reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert ir_eval.reciprocal_rank(["x", "y", "z", "w", "a"], {"a"}) == 0.0   # outside top 4
    assert ir_eval.ndcg(["a"], {"a"}) == 1.0
    assert ir_eval.ndcg(["x", "a"], {"a"}) == pytest.approx(1 / math.log2(3))
    assert ir_eval.ndcg(["a", "b", "x", "y"], {"a", "b"}) == 1.0
    assert ir_eval.ndcg(["x"], set()) == 0.0


def test_context_recall():
    assert run_ragas.context_recall(["a", "x"], ["a", "b"]) == 0.5
    assert run_ragas.context_recall([], ["a"]) == 0.0
    assert run_ragas.context_recall(["a"], []) == 0.0


def test_entity_extraction_and_recall():
    ents = run_ragas.extract_entities("Quarantine within 1 hour under PROT-SAF-7 for LOT-00044 at 8 degrees Celsius. QA reviews it.")
    assert {"PROT-SAF-7", "LOT-00044", "1", "8", "QA", "Celsius"} <= set(ents)
    assert "Quarantine" not in ents                      # sentence-initial word is skipped
    ctx = "Storage is 2 to 8 degrees Celsius under protocol PROT-SAF-7."
    assert run_ragas.entity_recall("Use PROT-SAF-7 at 8 degrees Celsius.", ctx) == 1.0
    assert run_ragas.entity_recall("Use PROT-SAF-7 at 9 degrees Celsius.", ctx) == pytest.approx(2 / 3)
    assert run_ragas.entity_recall("no entities here at all", ctx) is None
    assert not run_ragas.entity_in("8", "range 18 to 28")          # digits match whole numbers only
    assert not run_ragas.entity_in("LOT-00044", "LOT-000445")


def _rows():
    cats = {"factoid": 62, "complex_reasoning": 63, "multi_hop": 50, "summary": 38, "negative": 37}
    return [{"id": f"{c}-{i}", "category": c} for c, n in cats.items() for i in range(n)]


def test_stratified_sample_is_deterministic_and_proportional():
    rows = _rows()
    a = run_ragas.stratified_sample(rows, 50, 7)
    assert a == run_ragas.stratified_sample(rows, 50, 7)
    assert a != run_ragas.stratified_sample(rows, 50, 8)
    counts = {c: sum(r["category"] == c for r in a) for c in run_ragas.CATEGORIES}
    assert counts == {"factoid": 12, "complex_reasoning": 13, "multi_hop": 10, "summary": 8, "negative": 7}
    assert len(run_ragas.stratified_sample(rows, 0, 7)) == 250


def test_parse_json_tolerates_fences_and_rejects_garbage():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} done') == {"a": [1, 2]}
    assert parse_json("not json at all") is None


def test_golden_file_if_present_matches_the_stratification():
    path = ROOT / "eval" / "golden" / "golden.jsonl"
    if not path.exists():
        pytest.skip("golden set not built")
    rows = ir_eval.load_golden(path)
    from collections import Counter
    assert Counter(r["category"] for r in rows) == {"factoid": 62, "complex_reasoning": 63, "multi_hop": 50, "summary": 38, "negative": 37}
    assert len({r["id"] for r in rows}) == 250
    assert all(r["gold_chunk_hashes"] == [] for r in rows if r["category"] == "negative")
    assert all(r["gold_chunk_hashes"] for r in rows if r["category"] != "negative")
    assert all(len(h) == 64 for r in rows for h in r["gold_chunk_hashes"])
