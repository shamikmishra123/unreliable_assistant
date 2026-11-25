"""Process-wide wiring. Two modes share the same contracts and the same harness:

- `inproc`: tools bound to local objects (tests, evals, the workbench's lab)
- `mcp`:    tools forwarded over JSON-RPC to the three MCP servers (hybrid-rag, graphify-vault: stdio;
            state-store: Streamable HTTP), each verified against its signed manifest before use

A missing LLM credential fails start-up (no silent stub).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .bulkhead import Pools
from .cache import ResponseCache
from .config import ROOT, Settings, state_token
from .graph import Graph
from .graphify import load_or_build
from .harness import ToolExecutionHarness
from .incident import IncidentManager
from .llm import AnthropicGenerator, Generator, StubGenerator
from .plugins import PluginRegistry, load_plugin_config
from .rag.index import Models, RetrievalIndex, build_index
from .runtime import RuntimeState
from .state import StateStore
from .store import LocalStore, RemoteStore, Store
from .subagents import ExecutionSubagent, RetrievalSubagent, VerificationSubagent
from .tools import build_tools, contract_specs, remote_specs
from .tracing import Tracer

_models_lock = threading.Lock()
_models: dict[tuple[str, str], Models] = {}


def get_models(settings: Settings) -> Models:
    key = (settings.dense_model, settings.rerank_model)
    with _models_lock:
        if key not in _models:
            _models[key] = Models(settings.dense_model, settings.rerank_model)
        return _models[key]


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + n

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


@dataclass
class Services:
    settings: Settings
    mode: str
    store: Store
    harness: ToolExecutionHarness
    generator: Generator
    cache: ResponseCache
    tracer: Tracer
    plugins: PluginRegistry
    pools: Pools
    runtime: RuntimeState
    retrieval: RetrievalSubagent
    execution: ExecutionSubagent
    verification: VerificationSubagent
    incidents: IncidentManager
    state: StateStore | None = None
    index: RetrievalIndex | None = None
    graph: Graph | None = None
    host: Any = None
    metrics: Metrics = field(default_factory=Metrics)
    graph_source: str = ""

    def info(self) -> dict[str, Any]:
        base: dict[str, Any] = {"generator": self.generator.name, "mode": self.mode,
                                "verifier": self.verification.method}
        if self.index is not None:
            base.update({"tenants": sorted(self.index.tenants),
                         "chunks": {t: len(i.chunks) for t, i in self.index.tenants.items()},
                         "quarantined_chunks": len(self.index.load.quarantined), "notes": len(self.index.load.notes),
                         "dense_model": self.index.models.dense_name, "rerank_model": self.index.models.rerank_name})
        elif self.host is not None:
            import json

            try:
                base.update(json.loads(self.host.read_resource("mcp-hybrid-rag", "rag://index/info")))
            except Exception:  # noqa: BLE001 - info must not fail the health view
                base["index"] = "unavailable"
        return base

    def close(self) -> None:
        if hasattr(self.store, "close"):
            self.store.close()          # flushes the write-behind queue first
        if self.host is not None:
            self.host.stop()
        if self.state is not None:
            self.state.close()


def make_generator(settings: Settings) -> Generator:
    if settings.llm_stub:
        return StubGenerator()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError("no Anthropic credentials found: set ANTHROPIC_API_KEY or LLM_STUB=1")
    return AnthropicGenerator(settings.llm_model, settings.llm_step_timeout_s)


def build_services(settings: Settings, *, generator: Generator | None = None, models: Models | None = None,
                   mode: str = "inproc", verifier: str | None = None) -> Services:
    tracer = Tracer()
    plugins = PluginRegistry(load_plugin_config(), tracer)
    config = None
    host = state = index = graph = None
    graph_source = ""
    runtime = RuntimeState()

    if mode == "inproc":
        index = build_index(settings.vault_dir, models or get_models(settings))
        graph, graph_source = load_or_build(ROOT / "graph.json", settings.vault_dir)
        state = StateStore(settings.db_path, settings.approval_ttl_s)
        store: Store = LocalStore(state)
        tools = build_tools(index, graph, state)
    elif mode == "mcp":
        from .mcp_host import McpHost, load_config

        config = load_config()
        host = McpHost(config, state_token=state_token(), extra_env={
            "ASSISTANT_VAULT_DIR": str(settings.vault_dir), "ASSISTANT_DATA_DIR": str(settings.db_path.parent)})
        host.start(["hybrid-rag", "graphify-vault", "state-store"])
        store = RemoteStore(host, settings.db_path.parent / "audit.wal")

        def call(server: str, tool: str, model: BaseModel) -> dict[str, Any]:
            replica = host._conns.get("hybrid-rag-replica")
            key = "hybrid-rag-replica" if (server == "mcp-hybrid-rag" and runtime.tuning.rag_failover
                                           and replica is not None and replica.sessions) else server
            return host.call(key, tool, model.model_dump(mode="json"),
                             on_timing=lambda rtt, handler: tracer.add_span("mcp.client", max(0.0, rtt - handler),
                                                                            server=server, tool=tool))

        tools = remote_specs([s for s in contract_specs() if not s.internal], call)
    else:
        raise ValueError(f"unknown mode {mode}")

    harness = ToolExecutionHarness(tools, audit=store.append_audit, breaker_threshold=plugins.max_identical_calls)

    method = verifier or settings.verifier or ("nli" if mode == "mcp" else "lexical")
    entail = None
    if method == "nli":
        from .nli import get_scorer

        entail = get_scorer(settings.nli_model).entail
    metrics = Metrics()
    pools = Pools(config["bulkheads"] if config else None)
    cache = ResponseCache(settings.cache_ttl_s, settings.cache_jitter_s)
    incidents = IncidentManager(tracer=tracer, runtime=runtime, auditor=plugins.auditor, pools=pools, cache=cache,
                                metrics=metrics, host=host, store=store)
    return Services(
        settings=settings, mode=mode, store=store, harness=harness, generator=generator or make_generator(settings),
        cache=cache, tracer=tracer, plugins=plugins, pools=pools, runtime=runtime,
        retrieval=RetrievalSubagent(harness), execution=ExecutionSubagent(harness),
        verification=VerificationSubagent(entail, store.append_audit), incidents=incidents,
        state=state, index=index, graph=graph, host=host, metrics=metrics, graph_source=graph_source)
