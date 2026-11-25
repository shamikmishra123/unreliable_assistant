import json
import sqlite3
import uuid

import pytest

from assistant.contracts import CallContext, ErrorCode, ToolError
from assistant.harness import Tier
from assistant.state import StateConflict
from assistant.worker import execute_approval


def _ctx(tenant="tenant_a", user="user_op1", key=None):
    return CallContext(tenant_id=tenant, user_id=user, session_id=uuid.uuid4(),
                       idempotency_key=key or uuid.uuid4(),
                       traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01")


def test_lot_id_is_repaired_before_validation(svc):
    out = svc.harness.call("lookup_lot", {"lot_id": "lot 44"}, _ctx(), Tier.TIER_1_READ)
    assert out["found"] and out["lot_id"] == "LOT-00044" and out["status"] == "AVAILABLE"


def test_unrepairable_id_asks_for_clarification(svc):
    with pytest.raises(ToolError) as info:
        svc.harness.call("lookup_lot", {"lot_id": "lot-abc"}, _ctx(), Tier.TIER_1_READ)
    assert info.value.code == ErrorCode.AUTO_REPAIR_FAILED


def test_rbac_blocks_operator_from_direct_mutation_gate_only_without_tier(svc):
    args = {"action": "release", "payload": {"lot_id": "LOT-00044"}, "justification": "test"}
    with pytest.raises(ToolError) as info:
        svc.harness.call("submit_transaction", args, _ctx(), Tier.TIER_1_READ)
    assert info.value.code == ErrorCode.RBAC_DENIED


def test_idempotent_replay_returns_cached_result_and_rejects_changed_args(svc):
    key = uuid.uuid4()
    ctx = _ctx(key=key)
    first = svc.harness.call("lookup_lot", {"lot_id": "LOT-00044"}, ctx, Tier.TIER_1_READ)
    again = svc.harness.call("lookup_lot", {"lot_id": "LOT-00044"}, ctx, Tier.TIER_1_READ)
    assert first == again
    with pytest.raises(ToolError) as info:
        svc.harness.call("lookup_lot", {"lot_id": "LOT-00099"}, ctx, Tier.TIER_1_READ)
    assert info.value.code == ErrorCode.IDEMPOTENCY_REPLAY


def test_breaker_opens_after_two_failures_on_same_args(svc):
    args = {"fact": {"key": "shelf", "value": "24", "source_chunk_hash": "e" * 64}}
    for _ in range(2):
        with pytest.raises(ToolError) as info:
            svc.harness.call("put_session_fact", args, _ctx(), Tier.TIER_2_COMPUTE)
        assert info.value.code == ErrorCode.INVALID_PARAMS
    with pytest.raises(ToolError) as info:
        svc.harness.call("put_session_fact", args, _ctx(), Tier.TIER_2_COMPUTE)
    assert info.value.code == ErrorCode.CIRCUIT_OPEN


def test_tenant_scope_violation_is_hard_fail_and_audited(svc):
    with pytest.raises(ToolError) as info:
        svc.harness.call("lookup_lot", {"lot_id": "LOT-88392"}, _ctx("tenant_b"), Tier.TIER_1_READ)
    assert info.value.code == ErrorCode.TENANT_SCOPE_VIOLATION
    events = svc.state.audit_events("tenant_b")
    assert any(e["event"].get("type") == "security" for e in events)


def test_lots_are_tenant_scoped(svc):
    a = svc.harness.call("lookup_lot", {"lot_id": "LOT-00044"}, _ctx("tenant_a"), Tier.TIER_1_READ)
    b = svc.harness.call("lookup_lot", {"lot_id": "LOT-00044"}, _ctx("tenant_b"), Tier.TIER_1_READ)
    assert a["status"] == "AVAILABLE" and b["status"] == "RELEASED"


def _request_approval(svc, action="release", lot="LOT-00044", user="user_op1", tenant="tenant_a"):
    out = svc.harness.call("submit_transaction",
                           {"action": action, "payload": {"lot_id": lot}, "justification": "req"},
                           _ctx(tenant, user), Tier.TIER_2_COMPUTE)
    assert out["status"] == "PENDING_APPROVAL"
    return out["approval_id"]


def test_submit_creates_pending_record_and_does_not_mutate(svc):
    apr = _request_approval(svc)
    assert svc.state.get_lot("tenant_a", "LOT-00044")["status"] == "AVAILABLE"
    assert svc.state.get_approval("tenant_a", apr)["status"] == "PENDING_APPROVAL"


def test_four_eyes_rule_and_single_decision(svc):
    apr = _request_approval(svc, user="user_reva")
    with pytest.raises(PermissionError):
        svc.state.decide("tenant_a", apr, "user_reva", "APPROVE", "self")
    row = svc.state.decide("tenant_a", apr, "user_revother", "APPROVE", "ok")
    assert row["status"] == "APPROVED"
    with pytest.raises(StateConflict):
        svc.state.decide("tenant_a", apr, "user_other", "REJECT", "late")


def test_approved_action_runs_only_in_worker_path(svc):
    apr = _request_approval(svc, action="quarantine")
    row = svc.state.decide("tenant_a", apr, "user_reva", "APPROVE", "ok")
    assert svc.state.get_lot("tenant_a", "LOT-00044")["status"] == "AVAILABLE"
    result = execute_approval(svc.state, row)
    assert result == {"lot_id": "LOT-00044", "status": "QUARANTINED"}
    assert svc.state.get_lot("tenant_a", "LOT-00044")["status"] == "QUARANTINED"


def test_pending_approval_expires_to_rejected(svc):
    apr = _request_approval(svc)
    conn = sqlite3.connect(svc.state.path)
    conn.execute("UPDATE approvals SET expires_at='2000-01-01T00:00:00+00:00' WHERE approval_id=?", (apr,))
    conn.commit()
    conn.close()
    row = svc.state.get_approval("tenant_a", apr)
    assert row["status"] == "REJECTED" and row["decision_reason"].startswith("EXPIRED")


def test_audit_chain_verifies_and_detects_tampering(svc):
    svc.state.append_audit("tenant_a", {"type": "x", "n": 1})
    svc.state.append_audit("tenant_a", {"type": "x", "n": 2})
    assert svc.state.verify_audit()[0]
    conn = sqlite3.connect(svc.state.path)
    conn.execute("UPDATE audit SET body=replace(body, '\"n\":1', '\"n\":9') WHERE seq=(SELECT MIN(seq) FROM audit)")
    conn.commit()
    conn.close()
    assert not svc.state.verify_audit()[0]


def test_tool_calls_are_audited_with_hashes_not_arguments(svc):
    svc.harness.call("lookup_lot", {"lot_id": "LOT-00044"}, _ctx(), Tier.TIER_1_READ)
    events = [e["event"] for e in svc.state.audit_events("tenant_a") if e["event"].get("type") == "tool_call"]
    assert events and all("LOT-00044" not in json.dumps(e) for e in events)
