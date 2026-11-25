"""Runtime tuning knobs the incident runbooks turn (Gate 11, ADR-018). One shared mutable object, read per request."""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass


@dataclass
class Tuning:
    rerank_cutoff: float = 0.42          # NG-3 abstention gate; hallucination runbook raises it to 0.55
    top_k: int = 50                      # latency runbook cuts to 25
    aggressive_cache: bool = False       # latency runbook: word-sequence match with filler words removed
    strict_extractive: bool = False      # hallucination runbook: quote the top chunk, no free generation
    cache_scope: str = "tenant"          # leak runbook: "user" narrows keys to tenant:user
    skip_graph: bool = False             # latency lever (design §1.4): drop Graphify for simple queries
    rag_failover: bool = False           # hang runbook: use the replica retrieval server

    def snapshot(self) -> dict:
        return asdict(self)


class RuntimeState:
    def __init__(self) -> None:
        self.tuning = Tuning()
        self._lock = threading.Lock()

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if not hasattr(self.tuning, key):
                    raise KeyError(key)
                setattr(self.tuning, key, value)

    def reset(self) -> None:
        with self._lock:
            self.tuning = Tuning()
