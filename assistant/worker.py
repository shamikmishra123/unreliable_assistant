"""Approved-action worker (design §1.2.4: the action runs only after a Reviewer approves)."""
from __future__ import annotations

from .contracts import LOT_STATUSES
from .state import StateStore


def execute_approval(state: StateStore, approval: dict) -> dict:
    import json

    tenant = approval["tenant_id"]
    payload = json.loads(approval["payload"])
    action = approval["action"]
    lot_id = payload["lot_id"]
    if action in ("release", "quarantine", "hold", "reject", "update_status"):
        target = {"release": "RELEASED", "quarantine": "QUARANTINED", "hold": "ON_HOLD",
                  "reject": "REJECTED"}.get(action) or payload["status"]
        assert target in LOT_STATUSES
        state.set_lot_status(tenant, lot_id, target)
        result = {"lot_id": lot_id, "status": target}
    elif action in ("open_deviation", "open_capa"):
        record_id = state.add_deviation(tenant, lot_id, action, payload["description"])
        result = {"lot_id": lot_id, "record_id": record_id, "kind": action}
    else:  # pragma: no cover - the payload schema refuses unknown actions first
        raise ValueError(f"unknown action {action}")
    state.mark_executed(tenant, approval["approval_id"])
    return result
