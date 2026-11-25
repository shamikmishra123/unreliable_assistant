"""ToolExecutionHarness (design §1.2): every tool call passes through, in order,
validation (strict Pydantic) -> deterministic regex repair -> circuit breaker -> RBAC -> UUIDv4 idempotency.

Breaker and idempotency are scoped to the tool and the canonical argument hash.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError

from .contracts import (
    REPAIRABLE_FIELDS, CallContext, ErrorCode, ToolError, repair_lot_id, repair_sku_id,
)


class Tier(IntEnum):
    TIER_1_READ = 1
    TIER_2_COMPUTE = 2
    TIER_3_MUTATE = 3


@dataclass(frozen=True)
class ToolSpec:
    name: str
    server: str
    tier: Tier
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], dict[str, Any]]
    hitl_gate: bool = False   # Tier 3 request entry point: callable by operators, never executes here
    internal: bool = False    # host-only plumbing (audit, turns, reviewer decisions); never reachable via call()


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def repair_args(obj: Any, key: str | None = None) -> Any:
    """Deterministic repair of loose ID forms: 'lot 44' -> 'LOT-00044', 'SKU99' -> 'SKU-099'."""
    if isinstance(obj, dict):
        return {k: repair_args(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [repair_args(v, key) for v in obj]
    if isinstance(obj, str) and key in REPAIRABLE_FIELDS:
        if re.match(r"^\s*lot", obj, re.IGNORECASE):
            return repair_lot_id(obj) or obj
        if re.match(r"^\s*sku", obj, re.IGNORECASE):
            return repair_sku_id(obj) or obj
    return obj


class _Cached:
    __slots__ = ("method", "args_hash", "result")

    def __init__(self, method: str, args_hash: str, result: dict[str, Any]):
        self.method, self.args_hash, self.result = method, args_hash, result


class ToolExecutionHarness:
    REPLAY_CAPACITY = 10_000   # ADR-018

    def __init__(self, tools: list[ToolSpec], audit: Callable[[str, dict[str, Any]], Any],
                 breaker_threshold: int = 2):
        self._tools = {t.name: t for t in tools if not t.internal}
        self._audit = audit
        self._threshold = breaker_threshold
        self._fails: dict[tuple[str, str], int] = {}
        self._open: set[tuple[str, str]] = set()
        self._results: OrderedDict[UUID, _Cached] = OrderedDict()
        self._key_locks: dict[UUID, list] = {}
        self._lock = threading.Lock()

    @property
    def tools(self) -> dict[str, ToolSpec]:
        return dict(self._tools)

    def call(self, method: str, params: dict[str, Any], ctx: CallContext, caller_tier: Tier) -> dict[str, Any]:
        spec = self._tools.get(method)
        if spec is None:
            raise ToolError(ErrorCode.METHOD_NOT_FOUND, f"unknown tool {method}")

        # 1 + 2: repair, then strict validation
        raw = repair_args({k: v for k, v in params.items() if k != "ctx"})
        args_hash = canonical_hash(raw)
        key = (method, args_hash)
        try:
            spec.input_model.model_validate({**raw, "ctx": ctx})
        except ValidationError as err:
            code = (ErrorCode.AUTO_REPAIR_FAILED
                    if any(e["loc"] and str(e["loc"][-1]) in REPAIRABLE_FIELDS for e in err.errors())
                    else ErrorCode.SCHEMA_VALIDATION_FAILED)
            self._record_failure(ctx, key, method, code)
            raise ToolError(code, "arguments failed validation",
                            {"fields": [".".join(str(p) for p in e["loc"]) for e in err.errors()][:5]}) from None

        # Circuit breaker on (tool, args_hash)
        if key in self._open:
            self._audit_call(ctx, method, args_hash, ErrorCode.CIRCUIT_OPEN)
            raise ToolError(ErrorCode.CIRCUIT_OPEN, "breaker open for this tool and arguments")

        # 3: RBAC. Tier 3 entry points need at least a compute caller; they never execute the mutation here.
        required = Tier.TIER_2_COMPUTE if spec.hitl_gate else spec.tier
        if caller_tier < required:
            raise ToolError(ErrorCode.RBAC_DENIED, f"{method} requires {required.name}")

        # 4: idempotency, then execution
        entry = self._acquire_key(ctx.idempotency_key)
        try:
            with self._lock:
                cached = self._results.get(ctx.idempotency_key)
                if cached is not None:
                    self._results.move_to_end(ctx.idempotency_key)
            if cached is not None:
                if cached.method == method and cached.args_hash == args_hash:
                    self._audit_call(ctx, method, args_hash, ErrorCode.IDEMPOTENCY_REPLAY)
                    return copy.deepcopy(cached.result)
                raise ToolError(ErrorCode.IDEMPOTENCY_REPLAY, "idempotency key reused with different arguments")

            try:
                model = spec.input_model.model_validate({**raw, "ctx": ctx})
                result = spec.handler(model)
            except ToolError as err:
                self._record_failure(ctx, key, method, err.code)
                if err.code == ErrorCode.TENANT_SCOPE_VIOLATION:
                    self._audit(ctx.tenant_id, {"type": "security", "event": "TENANT_SCOPE_VIOLATION",
                                                "tool": method, "args_hash": args_hash})
                raise
            except Exception:  # pragma: no cover - defensive: never leak internals to the caller
                self._record_failure(ctx, key, method, ErrorCode.INTERNAL_ERROR)
                raise ToolError(ErrorCode.INTERNAL_ERROR, "internal error") from None

            with self._lock:
                self._fails.pop(key, None)
                self._results[ctx.idempotency_key] = _Cached(method, args_hash, copy.deepcopy(result))
                self._results.move_to_end(ctx.idempotency_key)
                while len(self._results) > self.REPLAY_CAPACITY:
                    self._results.popitem(last=False)
            self._audit_call(ctx, method, args_hash, None)
            return result
        finally:
            self._release_key(ctx.idempotency_key, entry)

    def breaker_state(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"tool": t, "args_hash": h[:12], "failures": self._fails.get(( t, h), 0),
                     "open": (t, h) in self._open}
                    for (t, h) in sorted(set(self._fails) | self._open)]

    def reset_breakers(self) -> int:
        with self._lock:
            n = len(self._open)
            self._open.clear()
            self._fails.clear()
            return n

    # ---- helpers ----------------------------------------------------------
    def _acquire_key(self, key: UUID) -> list:
        with self._lock:
            entry = self._key_locks.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        entry[0].acquire()
        return entry

    def _release_key(self, key: UUID, entry: list) -> None:
        entry[0].release()
        with self._lock:
            entry[1] -= 1
            if entry[1] == 0:
                self._key_locks.pop(key, None)

    def _record_failure(self, ctx: CallContext, key: tuple[str, str], method: str, code: ErrorCode) -> None:
        with self._lock:
            self._fails[key] = self._fails.get(key, 0) + 1
            tripped = self._fails[key] >= self._threshold
            if tripped:
                self._open.add(key)
        self._audit_call(ctx, method, key[1], code)
        if tripped:
            self._audit(ctx.tenant_id, {"type": "breaker_open", "tool": method, "args_hash": key[1]})

    def _audit_call(self, ctx: CallContext, method: str, args_hash: str, code: ErrorCode | None) -> None:
        self._audit(ctx.tenant_id, {"type": "tool_call", "tool": method, "args_hash": args_hash,
                                    "outcome": code.name if code else "OK", "trace": ctx.traceparent})
