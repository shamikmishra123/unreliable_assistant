"""Bulkhead isolation (Gate 9, ADR-013): per-pool thread executors, bounded queues, and a trip on infrastructure failures.

Only UPSTREAM_TIMEOUT, INTERNAL_ERROR and CIRCUIT_OPEN count toward tripping; validation, RBAC and scope errors are the
caller's fault and never open a pool.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .contracts import ErrorCode, ToolError

INFRA_CODES = {ErrorCode.UPSTREAM_TIMEOUT, ErrorCode.INTERNAL_ERROR, ErrorCode.CIRCUIT_OPEN}


class Bulkhead:
    def __init__(self, name: str, size: int, *, trip_after: int = 5, cooldown_s: float = 10.0, queue_factor: int = 4):
        self.name, self.size = name, size
        self.trip_after, self.cooldown_s = trip_after, cooldown_s
        self.executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix=f"bh-{name}")
        self._max_pending = size * queue_factor
        self._pending = 0
        self._fails = 0
        self._open_until = 0.0
        self._lock = threading.Lock()
        self.rejected = 0
        self.completed = 0
        self.tripped_count = 0

    @property
    def is_open(self) -> bool:
        return time.monotonic() < self._open_until

    def trip(self, seconds: float | None = None) -> None:
        with self._lock:
            self._open_until = time.monotonic() + (seconds or self.cooldown_s)
            self.tripped_count += 1

    def reset(self) -> None:
        with self._lock:
            self._open_until = 0.0
            self._fails = 0

    def _admit(self) -> None:
        with self._lock:
            if self.is_open:
                self.rejected += 1
                raise ToolError(ErrorCode.UPSTREAM_TIMEOUT, f"{self.name} bulkhead is open")
            if self._pending >= self._max_pending:
                self.rejected += 1
                raise ToolError(ErrorCode.UPSTREAM_TIMEOUT, f"{self.name} bulkhead is full")
            self._pending += 1

    def _done(self, err: ToolError | None) -> None:
        with self._lock:
            self._pending -= 1
            self.completed += 1
            if err is not None and err.code in INFRA_CODES:
                self._fails += 1
                if self._fails >= self.trip_after:
                    self._open_until = time.monotonic() + self.cooldown_s
                    self.tripped_count += 1
                    self._fails = 0
            elif err is None:
                self._fails = 0

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        self._admit()
        loop = asyncio.get_running_loop()
        import contextvars

        ctx = contextvars.copy_context()           # keep the current trace visible inside the worker thread
        try:
            result = await loop.run_in_executor(self.executor, lambda: ctx.run(fn, *args))
        except ToolError as err:
            self._done(err)
            raise
        except BaseException:
            self._done(ToolError(ErrorCode.INTERNAL_ERROR, "unexpected"))
            raise
        self._done(None)
        return result

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"name": self.name, "size": self.size, "pending": self._pending, "open": self.is_open,
                    "rejected": self.rejected, "completed": self.completed, "tripped": self.tripped_count}


class Pools:
    """rag_pool=64, db_pool=128, llm_pool=256, plus a small control pool so health checks and triage never queue
    behind a slow reranker (design §1.3 Bulkheads)."""

    SERVER_POOL = {"mcp-hybrid-rag": "rag_pool", "mcp-graphify-vault": "rag_pool", "mcp-state-store": "db_pool"}

    def __init__(self, sizes: dict[str, int] | None = None):
        sizes = {"rag_pool": 64, "db_pool": 128, "llm_pool": 256, "control_pool": 16, **(sizes or {})}
        self._pools = {name: Bulkhead(name, size) for name, size in sizes.items()}

    def __getitem__(self, name: str) -> Bulkhead:
        return self._pools[name]

    def for_server(self, server: str) -> Bulkhead:
        return self._pools[self.SERVER_POOL.get(server, "control_pool")]

    def stats(self) -> list[dict[str, Any]]:
        return [p.stats() for p in self._pools.values()]
