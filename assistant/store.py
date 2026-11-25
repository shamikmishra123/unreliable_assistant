"""State facade used by the orchestrator and API. Two implementations with the same surface:

- LocalStore: calls the StateStore handlers in process (inproc mode, tests, evals)
- RemoteStore: calls mcp-state-store over MCP; audit and turns are written behind a queue, in order, with a local
  WAL file on failure (FMEA #6: an unaudited write is never allowed to disappear silently)
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from .contracts import ErrorCode, ToolError
from .state import StateStore
from .state_internal import build_internal_tools


class Store(Protocol):
    def append_audit(self, tenant_id: str, event: dict[str, Any]) -> None: ...
    def add_turn(self, tenant_id: str, session_id: str, role: str, text: str) -> None: ...
    def recent_turns(self, tenant_id: str, session_id: str, n: int = 3) -> list[dict[str, str]]: ...
    def approval_view(self, tenant_id: str, approval_id: str) -> dict[str, Any] | None: ...
    def decide(self, tenant_id: str, approval_id: str, reviewer_id: str, decision: str, reason: str) -> dict[str, Any]: ...
    def list_approvals(self, tenant_id: str, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]: ...
    def verify_audit(self) -> tuple[bool, int]: ...
    def audit_events(self, tenant_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]: ...
    def flush(self) -> None: ...


class LocalStore:
    def __init__(self, state: StateStore):
        self.state = state
        self._h = {s.name: s.handler for s in build_internal_tools(state)}

    def _call(self, name: str, **kw: Any) -> dict[str, Any]:
        from . import state_internal as si

        model = {"append_audit": si.AppendAuditIn, "verify_audit": si.VerifyAuditIn, "audit_events": si.AuditEventsIn,
                 "add_turn": si.AddTurnIn, "recent_turns": si.RecentTurnsIn, "decide_approval": si.DecideApprovalIn,
                 "list_approvals": si.ListApprovalsIn, "approval_view": si.ApprovalViewIn}[name]
        return self._h[name](model(**kw))

    def append_audit(self, tenant_id, event):
        self.state.append_audit(tenant_id, event)

    def add_turn(self, tenant_id, session_id, role, text):
        self.state.add_turn(tenant_id, session_id, role, text)

    def recent_turns(self, tenant_id, session_id, n=3):
        return self.state.recent_turns(tenant_id, session_id, n)

    def approval_view(self, tenant_id, approval_id):
        try:
            return self._call("approval_view", tenant_id=tenant_id, approval_id=approval_id)
        except ToolError:
            return None

    def decide(self, tenant_id, approval_id, reviewer_id, decision, reason):
        return self._call("decide_approval", tenant_id=tenant_id, approval_id=approval_id,
                          reviewer_id=reviewer_id, decision=decision, reason=reason)

    def list_approvals(self, tenant_id, status=None, limit=50):
        return self._call("list_approvals", tenant_id=tenant_id, status=status, limit=limit)["approvals"]

    def verify_audit(self):
        return self.state.verify_audit()

    def audit_events(self, tenant_id=None, limit=200):
        return self.state.audit_events(tenant_id)[-limit:]

    def flush(self):
        return None


class RemoteStore:
    SERVER = "mcp-state-store"

    def __init__(self, host: Any, wal_path: Path):
        self.host = host
        self.wal_path = wal_path
        self._q: queue.Queue = queue.Queue()
        self._stop = False
        self.write_errors = 0
        self._thread = threading.Thread(target=self._drain, name="store-writer", daemon=True)
        self._thread.start()
        self._replay_wal()

    # ---- ordered write-behind ----------------------------------------------------
    def _enqueue(self, tool: str, args: dict[str, Any]) -> None:
        self._q.put((tool, args))

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            tool, args = item
            try:
                self.host.call(self.SERVER, tool, args)
            except ToolError:
                self.write_errors += 1
                self._wal(tool, args)
            self._q.task_done()

    def _wal(self, tool: str, args: dict[str, Any]) -> None:
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.wal_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tool": tool, "args": args}) + "\n")

    def _replay_wal(self) -> None:
        if not self.wal_path.exists():
            return
        lines = self.wal_path.read_text().splitlines()
        self.wal_path.unlink()
        for line in lines:
            item = json.loads(line)
            self._enqueue(item["tool"], item["args"])

    def flush(self) -> None:
        self._q.join()

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=10)

    # ---- Store surface -------------------------------------------------------------
    def append_audit(self, tenant_id, event):
        self._enqueue("append_audit", {"tenant_id": tenant_id, "event": json.loads(json.dumps(event, default=str))})

    def add_turn(self, tenant_id, session_id, role, text):
        self._enqueue("add_turn", {"tenant_id": tenant_id, "session_id": session_id, "role": role, "text": text[:4000]})

    def recent_turns(self, tenant_id, session_id, n=3):
        self.flush()   # a turn written a moment ago must be visible
        try:
            return self.host.call(self.SERVER, "recent_turns",
                                  {"tenant_id": tenant_id, "session_id": session_id, "n": n})["turns"]
        except ToolError:
            return []

    def approval_view(self, tenant_id, approval_id):
        try:
            return self.host.call(self.SERVER, "approval_view", {"tenant_id": tenant_id, "approval_id": approval_id})
        except ToolError as err:
            if err.code == ErrorCode.INVALID_PARAMS:
                return None
            raise

    def decide(self, tenant_id, approval_id, reviewer_id, decision, reason):
        self.flush()
        return self.host.call(self.SERVER, "decide_approval", {
            "tenant_id": tenant_id, "approval_id": approval_id, "reviewer_id": reviewer_id,
            "decision": decision, "reason": reason})

    def list_approvals(self, tenant_id, status=None, limit=50):
        args: dict[str, Any] = {"tenant_id": tenant_id, "limit": limit}
        if status:
            args["status"] = status
        return self.host.call(self.SERVER, "list_approvals", args)["approvals"]

    def verify_audit(self):
        self.flush()
        r = self.host.call(self.SERVER, "verify_audit", {})
        return r["ok"], r["rows"]

    def audit_events(self, tenant_id=None, limit=200):
        self.flush()
        args: dict[str, Any] = {"limit": limit}
        if tenant_id:
            args["tenant_id"] = tenant_id
        return self.host.call(self.SERVER, "audit_events", args)["events"]
