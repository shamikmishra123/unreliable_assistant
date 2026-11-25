"""Incident runbooks (Gates 10-11, ADR-018). Rules are evaluated every few seconds by the API.

| Trigger                                         | Runbook action                                                              |
| p95 e2e > 550 ms for 30 s                        | aggressive cache (word-sequence match), top_k 50 -> 25, drop Graphify        |
| mean faithfulness < 0.90 over 5 min              | rerank cutoff 0.42 -> 0.55, strict extractive quote mode                      |
| mcp-hybrid-rag timeout rate > 1%                 | trip rag bulkhead, fail over to the replica server, restart the worker        |
| egress tenant mismatch / key collision           | narrow cache keys to tenant:user, purge the cache                             |
Closing an incident restores the default tuning for that kind.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

LATENCY_P95_MS = 550.0
LATENCY_WINDOW_S = 30.0
FAITHFULNESS_FLOOR = 0.90
FAITHFULNESS_WINDOW_S = 300.0
HANG_TIMEOUT_RATE = 0.01
MIN_SAMPLES = 20

DEFAULTS_BY_KIND = {
    "latency_breach": {"aggressive_cache": False, "top_k": 50, "skip_graph": False},
    "hallucination_surge": {"rerank_cutoff": 0.42, "strict_extractive": False},
    "mcp_hang": {"rag_failover": False},
    "cache_leak": {"cache_scope": "tenant"},
}


@dataclass
class Incident:
    id: str
    kind: str
    opened_at: float
    trigger: str
    actions: list[str] = field(default_factory=list)
    resolved_at: float | None = None

    def view(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "opened_at": self.opened_at, "trigger": self.trigger,
                "actions": self.actions, "resolved_at": self.resolved_at, "open": self.resolved_at is None}


class IncidentManager:
    def __init__(self, *, tracer, runtime, auditor, pools, cache, metrics, host=None, store=None,
                 min_samples: int = MIN_SAMPLES):
        self.tracer, self.runtime, self.auditor = tracer, runtime, auditor
        self.pools, self.cache, self.metrics, self.host, self.store = pools, cache, metrics, host, store
        self.min_samples = min_samples
        self.incidents: list[Incident] = []
        self._lock = threading.Lock()
        self._mismatch_seen = 0
        self.restart_log: list[str] = []

    def _open(self, kind: str) -> bool:
        return any(i.kind == kind and i.resolved_at is None for i in self.incidents)

    def _raise(self, kind: str, trigger: str, actions: list[str]) -> Incident:
        inc = Incident(uuid.uuid4().hex[:8], kind, time.time(), trigger, actions)
        self.incidents.append(inc)
        self.metrics.inc(f"incident_{kind}")
        if self.store is not None:
            self.store.append_audit("tenant_platform" if False else "tenant_ops",
                                    {"type": "incident", "kind": kind, "trigger": trigger, "actions": actions})
        return inc

    def evaluate(self) -> list[Incident]:
        opened: list[Incident] = []
        with self._lock:
            lat = self.tracer.stats("e2e", since_s=LATENCY_WINDOW_S)
            if lat and lat["count"] >= self.min_samples and lat["p95"] > LATENCY_P95_MS and not self._open("latency_breach"):
                self.runtime.update(aggressive_cache=True, top_k=25, skip_graph=True)
                opened.append(self._raise("latency_breach", f"p95 e2e {lat['p95']:.0f} ms > {LATENCY_P95_MS:.0f} ms over {LATENCY_WINDOW_S:.0f}s",
                                          ["aggressive semantic cache on (word-sequence match, ADR-018)", "top_k 50 -> 25",
                                           "graphify subgraph dropped"]))
            if self.auditor is not None and not self._open("hallucination_surge"):
                mean, n = self.auditor.mean_since(FAITHFULNESS_WINDOW_S)
                if mean is not None and n >= max(5, self.min_samples // 4) and mean < FAITHFULNESS_FLOOR:
                    self.runtime.update(rerank_cutoff=0.55, strict_extractive=True)
                    opened.append(self._raise("hallucination_surge", f"mean faithfulness {mean:.2f} < {FAITHFULNESS_FLOOR} over 5 min (n={n})",
                                              ["rerank cutoff 0.42 -> 0.55", "strict extractive quote mode"]))
            if self.host is not None and not self._open("mcp_hang"):
                try:
                    rate = self.host.timeout_rate("hybrid-rag")
                    calls = self.host._conns["hybrid-rag"].stats
                    if calls.calls + calls.timeouts >= self.min_samples and rate > HANG_TIMEOUT_RATE:
                        actions = ["rag bulkhead tripped", "failover to replica index"]
                        self.pools["rag_pool"].trip(10.0)
                        self.runtime.update(rag_failover=True)
                        threading.Thread(target=self._restart_rag, name="rag-restart", daemon=True).start()
                        actions.append("restarting mcp-hybrid-rag worker")
                        opened.append(self._raise("mcp_hang", f"mcp-hybrid-rag timeout rate {rate:.1%} > {HANG_TIMEOUT_RATE:.0%}", actions))
                except KeyError:
                    pass
            mismatches = self.metrics.snapshot().get("egress_tenant_mismatch", 0) + self.metrics.snapshot().get("cache_key_collision", 0)
            if mismatches > self._mismatch_seen and not self._open("cache_leak"):
                self._mismatch_seen = mismatches
                purged = self.cache.clear() if hasattr(self.cache, "clear") else None
                self.runtime.update(cache_scope="user")
                opened.append(self._raise("cache_leak", "tenant key collision / egress tenant mismatch",
                                          ["cache keys narrowed to SHA256(tenant:user:query)", "cache purged", "on-call paged (audit event)"]))
        return opened

    def _restart_rag(self) -> None:
        try:
            self.host.start(["hybrid-rag-replica"])     # serve from the replica while the primary restarts
            self.restart_log.append("replica ready")
            self.host.restart("hybrid-rag")
            self.restart_log.append("restarted hybrid-rag")
        except Exception as err:   # noqa: BLE001
            self.restart_log.append(f"restart failed: {type(err).__name__}")

    def resolve(self, incident_id: str) -> bool:
        with self._lock:
            for inc in self.incidents:
                if inc.id == incident_id and inc.resolved_at is None:
                    inc.resolved_at = time.time()
                    self.runtime.update(**DEFAULTS_BY_KIND[inc.kind])
                    return True
        return False

    def view(self) -> list[dict[str, Any]]:
        return [i.view() for i in self.incidents][::-1]
