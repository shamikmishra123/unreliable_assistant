import uuid

import pytest

from assistant.contracts import CallContext, ErrorCode, ToolError
from assistant.harness import Tier
from assistant.rag.corpus import load_vault
from assistant.security import is_poisoned
from pathlib import Path

VAULT = Path(__file__).resolve().parent.parent / "knowledge_vault"


def _ctx(tenant="tenant_a"):
    return CallContext(tenant_id=tenant, user_id="user_op1", session_id=uuid.uuid4(),
                       idempotency_key=uuid.uuid4(),
                       traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01")


def test_seed_vault_has_no_quarantined_chunks():
    load = load_vault(VAULT, is_poisoned)
    assert load.quarantined == []
    assert len(load.notes) == 30


def test_exact_identifier_retrieval_per_tenant(svc):
    a = svc.index.retrieve("tenant_a", "What is the shelf life of SKU-099?")
    b = svc.index.retrieve("tenant_b", "What is the shelf life of SKU-099?")
    assert a["chunks"][0]["note_id"] == "catalog-sku-099" and not a["abstain"]
    assert b["chunks"][0]["note_id"] == "catalog-sku-099-riverside"
    # Tenant isolation: tenant_b never sees tenant_a chunk hashes or notes
    a_hashes = {c["chunk_hash"] for c in a["chunks"]}
    assert not a_hashes & {c["chunk_hash"] for c in b["chunks"]}
    assert all(c["note_id"].endswith("riverside") or c["note_id"].startswith("sop-lot-release-riverside")
               for c in b["chunks"])


def test_out_of_scope_query_abstains(svc):
    out = svc.index.retrieve("tenant_a", "What is the weather in Paris tomorrow?")
    assert out["abstain"] and out["top_score"] < 0.42


def test_domain_filter(svc):
    out = svc.index.retrieve("tenant_a", "policy", domain="policy")
    assert out["chunks"] and all(c["note_id"] in {"policy-assistant-scope", "policy-pii", "glossary", "sop-training",
                                 "policy-access-control", "policy-data-retention"}
                                 for c in out["chunks"])


def test_unknown_fact_source_is_refused(svc):
    call = svc.harness.call
    with pytest.raises(ToolError) as info:
        call("put_session_fact", {"fact": {"key": "shelf", "value": "24", "source_chunk_hash": "f" * 64}},
             _ctx(), Tier.TIER_2_COMPUTE)
    assert info.value.code == ErrorCode.INVALID_PARAMS


def test_graph_subgraph_stays_in_tenant_and_respects_depth(svc):
    ctx = _ctx("tenant_a")
    sub = svc.graph.subgraph(ctx, ["sop-lot-release"], depth=2, max_tokens=800)
    ids = {n["id"] for n in sub["nodes"]}
    assert "sop-lot-release" in ids and "sop-quarantine" in ids
    assert "catalog-sku-099-riverside" not in ids
    assert sub["token_count"] <= 800 and sub["raw_token_estimate"] >= sub["token_count"]
