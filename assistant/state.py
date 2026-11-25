"""State store (design §1.2.4): SQLite locally. Audit chain, approvals, lots, facts, turns.

Every write is serialised on one lock. Audit events are chained with
hash = SHA256(prev_hash + canonical(event)) and are verified on demand.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class StateConflict(Exception):
    """A decision that is not allowed in the current state (already decided, expired, four-eyes)."""


class StateStore:
    def __init__(self, db_path: Path | str, approval_ttl_s: float = 24 * 3600.0):
        self.path = str(db_path)
        self.approval_ttl = timedelta(seconds=approval_ttl_s)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    # ---- audit chain -----------------------------------------------------
    def append_audit(self, tenant_id: str, event: dict[str, Any]) -> tuple[int, str]:
        with self._lock:
            row = self._db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else GENESIS
            body = canonical({"tenant_id": tenant_id, "ts": iso(now_utc()), "event": event})
            digest = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
            cur = self._db.execute("INSERT INTO audit(body, prev_hash, hash) VALUES (?,?,?)",
                                   (body, prev, digest))
            self._db.commit()
            return cur.lastrowid, digest

    def verify_audit(self) -> tuple[bool, int]:
        """Recompute every link. Returns (ok, rows checked)."""
        with self._lock:
            prev, count = GENESIS, 0
            for row in self._db.execute("SELECT seq, body, prev_hash, hash FROM audit ORDER BY seq"):
                count += 1
                if row["prev_hash"] != prev:
                    return False, count
                expect = hashlib.sha256((prev + row["body"]).encode("utf-8")).hexdigest()
                if expect != row["hash"]:
                    return False, count
                prev = row["hash"]
            return True, count

    def audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT body FROM audit ORDER BY seq").fetchall()
        events = [json.loads(r["body"]) for r in rows]
        return [e for e in events if tenant_id is None or e["tenant_id"] == tenant_id]

    # ---- turns (hot session window, last 3 kept in prompts) --------------
    def add_turn(self, tenant_id: str, session_id: str, role: str, text: str) -> None:
        with self._lock:
            self._db.execute("INSERT INTO turns(tenant_id, session_id, role, text, created_at) VALUES (?,?,?,?,?)",
                             (tenant_id, session_id, role, text, iso(now_utc())))
            self._db.commit()

    def recent_turns(self, tenant_id: str, session_id: str, n: int = 3) -> list[dict[str, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, text FROM turns WHERE tenant_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
                (tenant_id, session_id, n)).fetchall()
        return [{"role": r["role"], "text": r["text"]} for r in reversed(rows)]

    # ---- session facts ----------------------------------------------------
    def put_fact(self, tenant_id: str, session_id: str, key: str, value: str,
                 source_chunk_hash: str | None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO facts(tenant_id, session_id, key, value, source_chunk_hash, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (tenant_id, session_id, key, value, source_chunk_hash, iso(now_utc())))
            self._db.commit()
            return cur.lastrowid

    def get_facts(self, tenant_id: str, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT key, value, source_chunk_hash FROM facts WHERE tenant_id=? AND session_id=? ORDER BY id",
                (tenant_id, session_id)).fetchall()
        return [dict(r) for r in rows]

    # ---- lots -------------------------------------------------------------
    def upsert_lot(self, tenant_id: str, lot_id: str, status: str, protocol_ref: str | None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO lots(tenant_id, lot_id, status, protocol_ref, updated_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(tenant_id, lot_id) DO UPDATE SET status=excluded.status, "
                "protocol_ref=excluded.protocol_ref, updated_at=excluded.updated_at",
                (tenant_id, lot_id, status, protocol_ref, iso(now_utc())))
            self._db.commit()

    def get_lot(self, tenant_id: str, lot_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT lot_id, status, protocol_ref, updated_at FROM lots "
                                   "WHERE tenant_id=? AND lot_id=?", (tenant_id, lot_id)).fetchone()
        return dict(row) if row else None

    def lot_owned_elsewhere(self, tenant_id: str, lot_id: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM lots WHERE lot_id=? AND tenant_id<>? LIMIT 1",
                                   (lot_id, tenant_id)).fetchone()
        return row is not None

    def set_lot_status(self, tenant_id: str, lot_id: str, status: str) -> None:
        with self._lock:
            self._db.execute("UPDATE lots SET status=?, updated_at=? WHERE tenant_id=? AND lot_id=?",
                             (status, iso(now_utc()), tenant_id, lot_id))
            self._db.commit()

    def add_deviation(self, tenant_id: str, lot_id: str, kind: str, description: str) -> int:
        with self._lock:
            cur = self._db.execute("INSERT INTO deviations(tenant_id, lot_id, kind, description, created_at) "
                                   "VALUES (?,?,?,?,?)", (tenant_id, lot_id, kind, description, iso(now_utc())))
            self._db.commit()
            return cur.lastrowid

    # ---- approvals (HITL, ADR-013) ---------------------------------------
    def create_approval(self, *, idempotency_key: str, tenant_id: str, session_id: str, requester_id: str,
                        action: str, payload: dict[str, Any], justification: str) -> str:
        """Idempotent on the call's idempotency key: a retried call returns the same approval."""
        with self._lock:
            row = self._db.execute("SELECT approval_id FROM approvals WHERE idempotency_key=?",
                                   (idempotency_key,)).fetchone()
            if row:
                return row["approval_id"]
            approval_id = "apr_" + uuid.uuid4().hex
            created = now_utc()
            self._db.execute(
                "INSERT INTO approvals(approval_id, idempotency_key, tenant_id, session_id, requester_id, action, "
                "payload, justification, status, created_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (approval_id, idempotency_key, tenant_id, session_id, requester_id, action, canonical(payload),
                 justification, "PENDING_APPROVAL", iso(created), iso(created + self.approval_ttl)))
            self._db.commit()
            return approval_id

    def _approval_row(self, tenant_id: str, approval_id: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM approvals WHERE approval_id=? AND tenant_id=?",
                               (approval_id, tenant_id)).fetchone()
        return dict(row) if row else None

    def _expire_if_due(self, row: dict[str, Any]) -> dict[str, Any]:
        if row["status"] == "PENDING_APPROVAL" and now_utc() > datetime.fromisoformat(row["expires_at"]):
            self._db.execute("UPDATE approvals SET status='REJECTED', decided_at=?, decision_reason=? "
                             "WHERE approval_id=?", (iso(now_utc()), "EXPIRED after TTL", row["approval_id"]))
            self._db.commit()
            row = self._approval_row(row["tenant_id"], row["approval_id"])
        return row

    def get_approval(self, tenant_id: str, approval_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._approval_row(tenant_id, approval_id)
            return self._expire_if_due(row) if row else None

    def decide(self, tenant_id: str, approval_id: str, reviewer_id: str, decision: str, reason: str) -> dict[str, Any]:
        """Four-eyes rule: the requester can never decide their own request (ADR-013)."""
        if decision not in ("APPROVE", "REJECT"):
            raise StateConflict("decision must be APPROVE or REJECT")
        with self._lock:
            row = self._approval_row(tenant_id, approval_id)
            if row is None:
                raise KeyError(approval_id)
            row = self._expire_if_due(row)
            if row["status"] != "PENDING_APPROVAL":
                raise StateConflict(f"approval is already {row['status']}")
            if row["requester_id"] == reviewer_id:
                raise PermissionError("four-eyes rule: requester cannot decide their own request")
            status = "APPROVED" if decision == "APPROVE" else "REJECTED"
            self._db.execute("UPDATE approvals SET status=?, reviewer_id=?, decision_reason=?, decided_at=? "
                             "WHERE approval_id=?", (status, reviewer_id, reason, iso(now_utc()), approval_id))
            self._db.commit()
            return self._approval_row(tenant_id, approval_id)

    def mark_executed(self, tenant_id: str, approval_id: str) -> None:
        with self._lock:
            self._db.execute("UPDATE approvals SET executed_at=? WHERE approval_id=? AND tenant_id=?",
                             (iso(now_utc()), approval_id, tenant_id))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS audit(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  body TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lots(
  tenant_id TEXT NOT NULL, lot_id TEXT NOT NULL, status TEXT NOT NULL,
  protocol_ref TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(tenant_id, lot_id));
CREATE TABLE IF NOT EXISTS approvals(
  approval_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL, tenant_id TEXT NOT NULL,
  session_id TEXT NOT NULL, requester_id TEXT NOT NULL, action TEXT NOT NULL, payload TEXT NOT NULL,
  justification TEXT NOT NULL, status TEXT NOT NULL, reviewer_id TEXT, decision_reason TEXT,
  created_at TEXT NOT NULL, expires_at TEXT NOT NULL, decided_at TEXT, executed_at TEXT);
CREATE TABLE IF NOT EXISTS facts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, session_id TEXT NOT NULL,
  key TEXT NOT NULL, value TEXT NOT NULL, source_chunk_hash TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS turns(
  id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, session_id TEXT NOT NULL,
  role TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS deviations(
  id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, lot_id TEXT NOT NULL,
  kind TEXT NOT NULL, description TEXT NOT NULL, created_at TEXT NOT NULL);
"""


def _list_approvals(self: "StateStore", tenant_id: str, status: str | None, limit: int) -> list[dict[str, Any]]:
    with self._lock:
        sql = "SELECT approval_id, requester_id, action, payload, justification, status, reviewer_id, " \
              "created_at, expires_at, decided_at FROM approvals WHERE tenant_id=?"
        args: list[Any] = [tenant_id]
        if status:
            sql += " AND status=?"
            args.append(status)
        rows = self._db.execute(sql + " ORDER BY created_at DESC LIMIT ?", (*args, limit)).fetchall()
        out = []
        for r in rows:
            row = self._expire_if_due({**dict(r), "tenant_id": tenant_id})
            if status is None or row["status"] == status:
                out.append({k: row[k] for k in ("approval_id", "requester_id", "action", "payload", "justification",
                                                "status", "reviewer_id", "created_at", "expires_at", "decided_at")})
        return out


StateStore.list_approvals = _list_approvals  # type: ignore[attr-defined]
