"""Lifecycle plugins (Gate 0, ADR-018): a Pluggy registry driven by mcp_config.json.

Stages: PRE_EXECUTION (pii_redactor -> injection_shield, fail closed), TOOL_DISPATCH (circuit_breaker config),
POST_EXECUTION (ragas_auditor), ALL_SPANS (otel_exporter). The loader refuses a config that drops, disables,
reorders or opens a guardrail, or loosens the breaker, so a guardrail can be shown but never toggled (ADR-012).
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pluggy

from .contracts import ErrorCode, ToolError
from .security import check_user_query, redact_pii

PROJECT = "assistant"
hookspec = pluggy.HookspecMarker(PROJECT)
hookimpl = pluggy.HookimplMarker(PROJECT)

GUARDRAILS = ("pii_redactor", "injection_shield", "circuit_breaker")
TOGGLEABLE = ("ragas_auditor", "otel_exporter")
MAX_IDENTICAL_CALLS = 2


class PluginConfigError(Exception):
    pass


@dataclass
class PreState:
    text: str
    redaction_counts: dict[str, int] = field(default_factory=dict)


class Specs:
    @hookspec
    def pre_execution(self, state: PreState) -> None:
        """Mutate/inspect the (already redacted) user text. Raise ToolError to refuse (fail closed)."""

    @hookspec
    def post_execution(self, response: dict[str, Any]) -> None:
        """Observe a finished response (never modifies the answer)."""

    @hookspec
    def trace_finished(self, trace: Any) -> None:
        """Receive a finished trace (ALL_SPANS)."""


class PiiRedactor:
    @hookimpl
    def pre_execution(self, state: PreState) -> None:
        try:
            red = redact_pii(state.text)
        except Exception:
            raise ToolError(ErrorCode.PII_BLOCKED, "pii redaction failed; request refused") from None
        state.text = red.text
        state.redaction_counts = red.counts


class InjectionShield:
    @hookimpl
    def pre_execution(self, state: PreState) -> None:
        try:
            check_user_query(state.text)
        except ToolError:
            raise
        except Exception:
            raise ToolError(ErrorCode.INJECTION_BLOCKED, "injection check failed; request refused") from None


class RagasAuditor:
    """Lite post-hook: records the verifier's faithfulness per released answer in a time window and flags answers
    below `min_faithfulness`. Full RAGAS (LLM judge) runs offline in eval/run_ragas.py; it costs money per call."""

    def __init__(self, min_faithfulness: float = 0.92, window: int = 2000):
        self.min_faithfulness = min_faithfulness
        self.samples: deque[tuple[float, float]] = deque(maxlen=window)
        self.below_min = 0
        self._lock = threading.Lock()

    @hookimpl
    def post_execution(self, response: dict[str, Any]) -> None:
        if response.get("route") != "HYBRID_RAG" or response.get("abstained") or response.get("cache") == "HIT":
            return
        value = float(response["faithfulness"])
        with self._lock:
            self.samples.append((time.time(), value))
            if value < self.min_faithfulness:
                self.below_min += 1

    def mean_since(self, seconds: float) -> tuple[float | None, int]:
        cutoff = time.time() - seconds
        with self._lock:
            values = [v for t, v in self.samples if t >= cutoff]
        return (sum(values) / len(values), len(values)) if values else (None, 0)


class OtelPlugin:
    def __init__(self, tracer: Any, endpoint: str, export: bool):
        self.tracer, self.endpoint, self.export_enabled = tracer, endpoint, export
        self.exporter = None
        if export:
            from .tracing import OtelExporter

            self.exporter = OtelExporter(endpoint)
            tracer.exporter = self.exporter

    @hookimpl
    def trace_finished(self, trace: Any) -> None:
        return None


@dataclass
class PluginEntry:
    name: str
    stage: str
    priority: int
    enabled: bool
    options: dict[str, Any]
    obj: Any = None


def validate_config(plugins: list[dict[str, Any]]) -> None:
    by_name = {p["name"]: p for p in plugins}
    for name in GUARDRAILS:
        entry = by_name.get(name)
        if entry is None:
            raise PluginConfigError(f"guardrail {name} cannot be dropped")
        if not entry.get("enabled", True):
            raise PluginConfigError(f"guardrail {name} cannot be disabled")
    for name in ("pii_redactor", "injection_shield"):
        if not by_name[name].get("fail_closed", False):
            raise PluginConfigError(f"guardrail {name} must fail closed")
    if by_name["pii_redactor"]["priority"] >= by_name["injection_shield"]["priority"]:
        raise PluginConfigError("pii_redactor must run before injection_shield")
    if int(by_name["circuit_breaker"].get("max_identical_calls", MAX_IDENTICAL_CALLS)) > MAX_IDENTICAL_CALLS:
        raise PluginConfigError(f"circuit_breaker cannot allow more than {MAX_IDENTICAL_CALLS} identical failing calls")


class PluginRegistry:
    def __init__(self, config: list[dict[str, Any]], tracer: Any):
        validate_config(config)
        self.entries: dict[str, PluginEntry] = {}
        for raw in config:
            opts = {k: v for k, v in raw.items() if k not in ("name", "stage", "priority", "enabled")}
            self.entries[raw["name"]] = PluginEntry(raw["name"], raw["stage"], int(raw["priority"]),
                                                    bool(raw.get("enabled", True)), opts)
        e = self.entries
        e["pii_redactor"].obj = PiiRedactor()
        e["injection_shield"].obj = InjectionShield()
        e["ragas_auditor"].obj = RagasAuditor(float(e["ragas_auditor"].options.get("min_faithfulness", 0.92)))
        otel = e["otel_exporter"]
        otel.obj = OtelPlugin(tracer, otel.options.get("endpoint", ""), bool(otel.options.get("export", False)))
        self.tracer = tracer
        self.max_identical_calls = int(e["circuit_breaker"].options.get("max_identical_calls", MAX_IDENTICAL_CALLS))
        self._pm = pluggy.PluginManager(PROJECT)
        self._pm.add_hookspecs(Specs)
        self._rebuild()

    def _rebuild(self) -> None:
        for plugin in list(self._pm.get_plugins()):
            self._pm.unregister(plugin)
        # pluggy calls hooks last-registered-first, so register the highest priority number first
        for entry in sorted((x for x in self.entries.values() if x.enabled and x.obj is not None),
                            key=lambda x: -x.priority):
            self._pm.register(entry.obj, name=entry.name)

    # ---- running -------------------------------------------------------------------
    def run_pre(self, text: str) -> PreState:
        state = PreState(text)
        self._pm.hook.pre_execution(state=state)   # ToolError propagates: fail closed
        return state

    def run_post(self, response: dict[str, Any]) -> None:
        try:
            self._pm.hook.post_execution(response=response)
        except Exception:      # a post-hook failure must never alter or block a response
            pass

    @property
    def auditor(self) -> RagasAuditor:
        return self.entries["ragas_auditor"].obj

    # ---- control -------------------------------------------------------------------
    def set_enabled(self, name: str, enabled: bool) -> None:
        if name in GUARDRAILS:
            raise PluginConfigError(f"{name} is a guardrail: it is shown, not toggled (ADR-012)")
        if name not in TOGGLEABLE:
            raise PluginConfigError(f"unknown plugin {name}")
        self.entries[name].enabled = enabled
        self._rebuild()

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": e.name, "stage": e.stage, "priority": e.priority, "enabled": e.enabled,
                 "locked": e.name in GUARDRAILS, "options": e.options}
                for e in sorted(self.entries.values(), key=lambda x: (x.stage, x.priority))]


def load_plugin_config(path: Path | None = None) -> list[dict[str, Any]]:
    from .config import ROOT

    return json.loads((path or ROOT / "mcp_config.json").read_text())["plugins"]
