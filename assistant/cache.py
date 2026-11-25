"""Tenant-scoped answer cache (ADR-003, ADR-012). Exact match on the normalized query, in process.

Key = cache:{tenant}: + SHA256(tenant_id:norm_query). TTL = 300 s +/- 45 s jitter. Egress re-checks tenant_id.
"""
from __future__ import annotations

import hashlib
import random
import threading
import time
from typing import Any


class ResponseCache:
    def __init__(self, ttl_s: float = 300.0, jitter_s: float = 45.0,
                 clock: Any = time.monotonic, rng: random.Random | None = None):
        self.ttl_s = ttl_s
        self.jitter_s = jitter_s
        self._clock = clock
        self._rng = rng or random.Random()
        self._data: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.mismatches = 0

    @staticmethod
    def key(tenant_id: str, norm_query: str, user_id: str | None = None) -> str:
        """ADR-003 tenant-level key; the leak runbook narrows it to SHA256(tenant:user:query) (Gate 11)."""
        material = f"{tenant_id}:{user_id}:{norm_query}" if user_id else f"{tenant_id}:{norm_query}"
        return f"cache:{tenant_id}:" + hashlib.sha256(material.encode()).hexdigest()

    def get(self, tenant_id: str, norm_query: str, user_id: str | None = None) -> dict[str, Any] | None:
        key = self.key(tenant_id, norm_query, user_id)
        with self._lock:
            entry = self._data.get(key)
            if entry is None or entry[0] < self._clock():
                self._data.pop(key, None)
                self.misses += 1
                return None
            value = entry[1]
            if value.get("tenant_id") != tenant_id:   # egress tenant check: block and purge (FMEA #12)
                self._data.pop(key, None)
                self.mismatches += 1
                self.misses += 1
                return None
            self.hits += 1
            return dict(value)

    def put(self, tenant_id: str, norm_query: str, value: dict[str, Any], user_id: str | None = None) -> None:
        ttl = self.ttl_s + self._rng.uniform(-self.jitter_s, self.jitter_s)
        with self._lock:
            self._data[self.key(tenant_id, norm_query, user_id)] = (self._clock() + ttl, dict(value))

    def purge_tenant(self, tenant_id: str) -> int:
        prefix = f"cache:{tenant_id}:"
        with self._lock:
            keys = [k for k in self._data if k.startswith(prefix)]
            for k in keys:
                del self._data[k]
        return len(keys)

    def clear(self) -> int:
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
