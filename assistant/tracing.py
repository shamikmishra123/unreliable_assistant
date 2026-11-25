"""Tracing (Gates 10-11): W3C traceparent, named spans, rolling latency stats, optional OTel export.

Spans are recorded in-process (ring buffer + per-name percentile windows) and can be replayed into the real
OpenTelemetry SDK with their true timestamps (BatchSpanProcessor -> OTLP) when the otel_exporter plugin has
`export: true`. Server-reported stage timings (BM25, dense, RRF, rerank) are added as child spans after the call.
"""
from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

SPAN_NAMES = (
    "gateway", "hook.pre", "cache.lookup", "triage.classify", "mcp.rag.bm25", "mcp.rag.biencoder", "mcp.rag.rrf",
    "mcp.rag.rerank", "mcp.graphify", "mcp.client", "agent.prompt", "agent.react.step", "llm.generate",
    "verify.nli", "hook.post", "hook.post.ragas", "e2e", "ttft",
)


@dataclass
class Span:
    name: str
    start_ms: float          # offset from trace start
    duration_ms: float
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRec:
    trace_id: str
    started_wall: float
    started_perf: float
    spans: list[Span] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


_current: contextvars.ContextVar[TraceRec | None] = contextvars.ContextVar("trace", default=None)


class Tracer:
    def __init__(self, max_traces: int = 300, window: int = 3000):
        self._traces: deque[TraceRec] = deque(maxlen=max_traces)
        self._samples: dict[str, deque[tuple[float, float]]] = {}
        self._window = window
        self._lock = threading.Lock()
        self.exporter: Any = None          # set by the otel_exporter plugin

    def start(self, trace_id: str) -> TraceRec:
        rec = TraceRec(trace_id, time.time(), time.perf_counter())
        _current.set(rec)
        return rec

    def end(self, rec: TraceRec, **attrs: Any) -> None:
        rec.attrs.update(attrs)
        with self._lock:
            self._traces.append(rec)
        if self.exporter is not None:
            with contextlib.suppress(Exception):   # observability must never break a request
                self.exporter.export(rec)

    @contextlib.contextmanager
    def span(self, name: str, **attrs: Any):
        rec = _current.get()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._add(rec, name, (started - rec.started_perf) * 1000 if rec else 0.0,
                      (time.perf_counter() - started) * 1000, attrs)

    def add_span(self, name: str, duration_ms: float, **attrs: Any) -> None:
        """A span whose duration was measured elsewhere (server-reported stage timings). Ends 'now'."""
        rec = _current.get()
        offset = ((time.perf_counter() - rec.started_perf) * 1000 - duration_ms) if rec else 0.0
        self._add(rec, name, max(0.0, offset), duration_ms, attrs)

    def _add(self, rec: TraceRec | None, name: str, start_ms: float, duration_ms: float, attrs: dict[str, Any]) -> None:
        if rec is not None:
            rec.spans.append(Span(name, round(start_ms, 3), round(duration_ms, 3), attrs))
        with self._lock:
            self._samples.setdefault(name, deque(maxlen=self._window)).append((time.time(), duration_ms))

    # ---- queries ---------------------------------------------------------------------
    def stats(self, name: str, since_s: float | None = None) -> dict[str, float] | None:
        with self._lock:
            data = list(self._samples.get(name, ()))
        if since_s is not None:
            cutoff = time.time() - since_s
            data = [d for d in data if d[0] >= cutoff]
        values = sorted(d[1] for d in data)
        if not values:
            return None
        q = lambda p: values[min(len(values) - 1, int(p * len(values)))]  # noqa: E731
        return {"count": len(values), "p50": round(q(0.5), 3), "p95": round(q(0.95), 3),
                "p99": round(q(0.99), 3), "max": round(values[-1], 3)}

    def all_stats(self) -> dict[str, dict[str, float]]:
        with self._lock:
            names = list(self._samples)
        return {n: s for n in names if (s := self.stats(n))}

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            traces = list(self._traces)[-limit:][::-1]
        return [{"trace_id": t.trace_id, "started": t.started_wall, "attrs": t.attrs,
                 "spans": [{"name": s.name, "start_ms": s.start_ms, "duration_ms": s.duration_ms, "attrs": s.attrs}
                           for s in sorted(t.spans, key=lambda s: s.start_ms)]} for t in traces]


class OtelExporter:
    """Replays finished traces into the OpenTelemetry SDK with their real timestamps."""

    def __init__(self, endpoint: str | None = None, span_exporter: Any = None, service: str = "unreliable-assistant"):
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        if span_exporter is not None:
            provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        elif endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
        self._tracer = provider.get_tracer("assistant")
        self._trace = trace
        self.provider = provider
        self.exported = 0

    def export(self, rec: TraceRec) -> None:
        base_ns = int(rec.started_wall * 1e9)
        root = self._tracer.start_span("request", start_time=base_ns)
        ctx = self._trace.set_span_in_context(root)
        end_ns = base_ns
        for s in rec.spans:
            start = base_ns + int(s.start_ms * 1e6)
            end = start + int(s.duration_ms * 1e6)
            end_ns = max(end_ns, end)
            span = self._tracer.start_span(s.name, context=ctx, start_time=start,
                                           attributes={k: str(v) for k, v in s.attrs.items()})
            span.end(end_time=end)
        root.set_attribute("trace_id.assistant", rec.trace_id)
        root.end(end_time=end_ns)
        self.exported += 1


def new_traceparent(trace_id: str) -> str:
    import secrets

    return f"00-{trace_id}-{secrets.token_hex(8)}-01"
