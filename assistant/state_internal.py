"""Internal (host-only) tools of mcp-state-store: audit chain, hot turns, reviewer decisions.

These are not part of the LLM-facing contract and never pass through ToolExecutionHarness.call(). The state
server requires a bearer token on its HTTP transport before any tool (internal or not) is reachable.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import APPROVAL_RE, TENANT_RE, USER_RE
from .harness import Tier, ToolSpec
from .state import StateConflict, StateStore
from .contracts import ErrorCode, ToolError
from .worker import execute_approval

STATE_SERVER = "mcp-state-store"


class _S(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class AppendAuditIn(_S):
    tenant_id: str = Field(pattern=TENANT_RE)
    event: dict


class VerifyAuditIn(_S):
    pass


class AuditEventsIn(_S):
    tenant_id: str | None = Field(default=None, pattern=TENANT_RE)
    limit: int = Field(default=200, ge=1, le=5000)


class AddTurnIn(_S):
    tenant_id: str = Field(pattern=TENANT_RE)
    session_id: str = Field(min_length=1, max_length=64)
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)


class RecentTurnsIn(_S):
    tenant_id: str = Field(pattern=TENANT_RE)
    session_id: str = Field(min_length=1, max_length=64)
    n: int = Field(default=3, ge=1, le=10)


class DecideApprovalIn(_S):
    tenant_id: str = Field(pattern=TENANT_RE)
    approval_id: str = Field(pattern=APPROVAL_RE)
    reviewer_id: str = Field(pattern=USER_RE)
    decision: Literal["APPROVE", "REJECT"]
    reason: str = Field(default="", max_length=512)


class ListApprovalsIn(_S):
    tenant_id: str = Field(pattern=TENANT_RE)
    status: Literal["PENDING_APPROVAL", "APPROVED", "REJECTED"] | None = None
    limit: int = Field(default=50, ge=1, le=500)


class ApprovalViewIn(_S):
    tenant_id: str = Field(pattern=TENANT_RE)
    approval_id: str = Field(pattern=APPROVAL_RE)


def build_internal_tools(state: StateStore) -> list[ToolSpec]:
    T = Tier.TIER_3_MUTATE

    def append_audit(m: AppendAuditIn) -> dict:
        seq, digest = state.append_audit(m.tenant_id, m.event)
        return {"seq": seq, "hash": digest}

    def verify_audit(m: VerifyAuditIn) -> dict:
        ok, rows = state.verify_audit()
        return {"ok": ok, "rows": rows}

    def audit_events(m: AuditEventsIn) -> dict:
        return {"events": state.audit_events(m.tenant_id)[-m.limit:]}

    def add_turn(m: AddTurnIn) -> dict:
        state.add_turn(m.tenant_id, m.session_id, m.role, m.text)
        return {"ok": True}

    def recent_turns(m: RecentTurnsIn) -> dict:
        return {"turns": state.recent_turns(m.tenant_id, m.session_id, m.n)}

    def decide_approval(m: DecideApprovalIn) -> dict:
        try:
            row = state.decide(m.tenant_id, m.approval_id, m.reviewer_id, m.decision, m.reason)
        except KeyError:
            raise ToolError(ErrorCode.INVALID_PARAMS, "unknown approval") from None
        except StateConflict as err:
            raise ToolError(ErrorCode.INVALID_PARAMS, f"conflict: {err}", {"http": 409}) from None
        except PermissionError as err:
            state.append_audit(m.tenant_id, {"type": "security", "event": "FOUR_EYES_REFUSED",
                                             "approval_id": m.approval_id, "reviewer": m.reviewer_id})
            raise ToolError(ErrorCode.RBAC_DENIED, str(err), {"http": 403}) from None
        result = execute_approval(state, row) if row["status"] == "APPROVED" else None
        state.append_audit(m.tenant_id, {"type": "hitl_decision", "approval_id": m.approval_id,
                                         "decision": row["status"], "reviewer": m.reviewer_id,
                                         "action": row["action"]})
        return {"approval_id": m.approval_id, "status": row["status"], "executed": result is not None,
                "result": result}

    def list_approvals(m: ListApprovalsIn) -> dict:
        return {"approvals": state.list_approvals(m.tenant_id, m.status, m.limit)}

    def approval_view(m: ApprovalViewIn) -> dict:
        import json

        row = state.get_approval(m.tenant_id, m.approval_id)
        if row is None:
            raise ToolError(ErrorCode.INVALID_PARAMS, "unknown approval")
        return {"approval_id": row["approval_id"], "status": row["status"], "action": row["action"],
                "lot_id": json.loads(row["payload"]).get("lot_id"), "reviewer_id": row["reviewer_id"],
                "decided_at": row["decided_at"], "executed": row["executed_at"] is not None}

    def spec(name, model, handler):
        return ToolSpec(name, STATE_SERVER, T, model, handler, internal=True)

    return [spec("append_audit", AppendAuditIn, append_audit), spec("verify_audit", VerifyAuditIn, verify_audit),
            spec("audit_events", AuditEventsIn, audit_events), spec("add_turn", AddTurnIn, add_turn),
            spec("recent_turns", RecentTurnsIn, recent_turns), spec("decide_approval", DecideApprovalIn, decide_approval),
            spec("list_approvals", ListApprovalsIn, list_approvals), spec("approval_view", ApprovalViewIn, approval_view)]
