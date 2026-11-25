"""MCP host client (design §0: `mcp.ClientSession`). One background event loop owns every server connection:
stdio children (hybrid-rag, graphify-vault) and a Streamable HTTP service (state-store).

Sync API for the harness (`call`, `read_resource`), because the orchestrator calls tools from worker threads.
Verifies each server's announced tools against its Ed25519-signed manifest before accepting it (Gate 13).
"""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .config import ROOT
from .contracts import ErrorCode, ToolError
from .manifest import ManifestError, verify_server_tools

_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Least privilege: a model-serving child does not need API keys."""
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in _SECRET_HINTS)}
    env.update(extra or {})
    return env


@dataclass
class ServerStats:
    calls: int = 0
    errors: int = 0
    timeouts: int = 0
    connected_at: float | None = None
    last_error: str | None = None
    rtt_ms: deque = field(default_factory=lambda: deque(maxlen=2000))
    overhead_ms: deque = field(default_factory=lambda: deque(maxlen=2000))


@dataclass
class _Conn:
    key: str
    cfg: dict[str, Any]
    sessions: list[ClientSession] = field(default_factory=list)
    task: asyncio.Task | None = None
    stop: asyncio.Event | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    tools: list[types.Tool] = field(default_factory=list)
    stats: ServerStats = field(default_factory=ServerStats)
    rr: Any = None
    manifest_ok: bool = False
    pid: int | None = None


def load_config(path: Path | None = None) -> dict[str, Any]:
    config = json.loads((path or ROOT / "mcp_config.json").read_text())
    if os.environ.get("MCP_STATE_URL"):
        config["mcpServers"]["state-store"]["url"] = os.environ["MCP_STATE_URL"]
    return config


def _leaf_error(err: BaseException) -> BaseException:
    """anyio task groups wrap errors in ExceptionGroup; report the first real cause."""
    while isinstance(err, BaseExceptionGroup) and err.exceptions:
        err = err.exceptions[0]
    return err


class McpHost:
    def __init__(self, config: dict[str, Any], *, verify_manifests: bool = True, state_token: str | None = None,
                 extra_env: dict[str, str] | None = None):
        self.config = config
        self.verify_manifests = verify_manifests
        self.state_token = state_token
        self.extra_env = extra_env or {}
        self._conns: dict[str, _Conn] = {}
        self._payloads: deque = deque(maxlen=300)
        self._lock = threading.Lock()
        self._children: list[subprocess.Popen] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-host-loop", daemon=True)
        self._thread.start()

    # ---- lifecycle -----------------------------------------------------------
    def start(self, keys: list[str] | None = None, connect_timeout_s: float = 240.0) -> None:
        wanted = keys or [k for k, c in self.config["mcpServers"].items() if not c.get("on_demand")]
        for key in wanted:
            self._spawn(key)
        deadline = time.monotonic() + connect_timeout_s
        for key in wanted:
            conn = self._conns[key]
            if not conn.ready.wait(max(0.1, deadline - time.monotonic())):
                raise RuntimeError(f"MCP server {key} did not become ready in {connect_timeout_s:.0f}s")
            if conn.error:
                raise RuntimeError(f"MCP server {key} failed: {conn.error}")

    def _spawn(self, key: str) -> None:
        cfg = self.config["mcpServers"][key]
        conn = _Conn(key=key, cfg=cfg)
        conn.rr = itertools.cycle(range(1))
        with self._lock:
            self._conns[key] = conn
        if cfg["transport"] == "streamable-http":
            self._ensure_http_service(cfg)
        conn.task = asyncio.run_coroutine_threadsafe(self._run(conn), self._loop)  # type: ignore[assignment]

    def _ensure_http_service(self, cfg: dict[str, Any]) -> None:
        auto = cfg.get("autostart")
        if not auto:
            return
        url = cfg["url"]
        try:
            probe = httpx.get(url, timeout=1.0)
        except httpx.HTTPError:
            probe = None
        if probe is not None:
            if probe.status_code == 401 and "unauthorized" in probe.text:
                return                                   # our state server (it answers 401 without a token)
            raise RuntimeError(f"{url} is already served by something that is not mcp-state-store "
                               f"(HTTP {probe.status_code}); set MCP_STATE_URL to a free port")
        (ROOT / "var" / "logs").mkdir(parents=True, exist_ok=True)
        log = open(ROOT / "var" / "logs" / "state-store.log", "ab")
        proc = subprocess.Popen([sys.executable, str(ROOT / auto["args"][0])], cwd=ROOT, stdout=log, stderr=log,
                                env=child_env({"MCP_STATE_TOKEN": self.state_token or "", **self.extra_env}))
        self._children.append(proc)
        for _ in range(100):
            try:
                httpx.get(url, timeout=1.0)
                return
            except httpx.HTTPError:
                time.sleep(0.2)
        raise RuntimeError("state-store did not start")

    async def _run(self, conn: _Conn) -> None:
        conn.stop = asyncio.Event()
        cfg = conn.cfg
        try:
            async with contextlib.AsyncExitStack() as stack:
                n = int(cfg.get("connections", 1))
                for _ in range(n):
                    if cfg["transport"] == "stdio":
                        (ROOT / "var" / "logs").mkdir(parents=True, exist_ok=True)
                        errlog = stack.enter_context(open(ROOT / "var" / "logs" / f"{conn.key}.log", "ab"))
                        params = StdioServerParameters(
                            command=sys.executable if cfg["command"] in ("python", "uv") else cfg["command"],
                            args=[str(ROOT / cfg["args"][-1])], cwd=str(ROOT), env=child_env(self.extra_env))
                        read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
                    else:
                        headers = {"Authorization": f"Bearer {self.state_token}"} if self.state_token else {}
                        read, write, _ = await stack.enter_async_context(
                            streamablehttp_client(cfg["url"], headers=headers))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    conn.sessions.append(session)
                tools = (await conn.sessions[0].list_tools()).tools
                conn.tools = tools
                if self.verify_manifests:
                    verify_server_tools(cfg["server_name"], tools)
                    conn.manifest_ok = True
                conn.rr = itertools.cycle(range(len(conn.sessions)))
                conn.stats.connected_at = time.time()
                conn.ready.set()
                await conn.stop.wait()
        except ManifestError as err:
            conn.error = f"manifest: {err}"
        except BaseException as err:  # noqa: BLE001 - surface any start-up failure to start()
            leaf = _leaf_error(err)
            conn.error = f"manifest: {leaf}" if isinstance(leaf, ManifestError) else f"{type(leaf).__name__}: {leaf}"
            conn.stats.last_error = conn.error
        finally:
            conn.sessions.clear()
            conn.ready.set()

    def stop(self) -> None:
        for conn in list(self._conns.values()):
            self._stop_conn(conn)
        for proc in self._children:
            proc.terminate()
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _stop_conn(self, conn: _Conn) -> None:
        if conn.stop is not None:
            self._loop.call_soon_threadsafe(conn.stop.set)
        if conn.task is not None:
            with contextlib.suppress(Exception):
                conn.task.result(timeout=10)

    def restart(self, key: str, timeout_s: float = 240.0) -> None:
        """Runbook action: restart a worker (hang mitigation)."""
        old = self._conns.get(key)
        if old:
            self._stop_conn(old)
        self._spawn(key)
        conn = self._conns[key]
        if not conn.ready.wait(timeout_s) or conn.error:
            raise RuntimeError(f"restart of {key} failed: {conn.error}")

    # ---- calls -------------------------------------------------------------------
    def _conn(self, key: str) -> _Conn:
        conn = self._conns.get(key)
        if conn is None or not conn.sessions:
            raise ToolError(ErrorCode.UPSTREAM_TIMEOUT, f"MCP server {key} is not connected")
        return conn

    @staticmethod
    def server_key(server_name: str) -> str:
        return server_name.removeprefix("mcp-")

    def call(self, server: str, tool: str, arguments: dict[str, Any], timeout_s: float | None = None,
             on_timing: Any = None) -> dict[str, Any]:
        key = self.server_key(server)
        conn = self._conn(key)
        timeout = timeout_s if timeout_s is not None else conn.cfg.get("timeout_ms", 5000) / 1000
        session = conn.sessions[next(conn.rr)]
        started = time.perf_counter()

        async def go() -> types.CallToolResult:
            return await session.call_tool(tool, arguments, read_timeout_seconds=timedelta(seconds=timeout))

        fut = asyncio.run_coroutine_threadsafe(go(), self._loop)
        try:
            result = fut.result(timeout + 1.0)
        except (TimeoutError, asyncio.TimeoutError, FutureTimeout):
            fut.cancel()
            conn.stats.timeouts += 1
            self._log(key, tool, arguments, {"error": "timeout"}, True, (time.perf_counter() - started) * 1000)
            raise ToolError(ErrorCode.UPSTREAM_TIMEOUT, f"{server} {tool} exceeded {timeout:.1f}s") from None
        except Exception as err:  # transport broke (server crashed, connection reset)
            conn.stats.errors += 1
            conn.stats.last_error = f"{type(err).__name__}: {err}"[:200]
            if "timed out" in str(err).lower():
                conn.stats.timeouts += 1
            self._log(key, tool, arguments, {"error": conn.stats.last_error}, True, (time.perf_counter() - started) * 1000)
            raise ToolError(ErrorCode.UPSTREAM_TIMEOUT, f"{server} unavailable") from None
        rtt = (time.perf_counter() - started) * 1000
        meta = result.meta or {}
        handler_ms = float(meta.get("handler_ms", 0.0))
        conn.stats.calls += 1
        conn.stats.rtt_ms.append(rtt)
        conn.stats.overhead_ms.append(max(0.0, rtt - handler_ms))
        if on_timing is not None:
            on_timing(rtt, handler_ms)
        text = result.content[0].text if result.content and hasattr(result.content[0], "text") else ""
        self._log(key, tool, arguments, result.structuredContent if not result.isError else text, result.isError, rtt)
        if result.isError:
            conn.stats.errors += 1
            raise _to_tool_error(text)
        return result.structuredContent if result.structuredContent is not None else json.loads(text or "{}")

    def read_resource(self, server: str, uri: str, timeout_s: float = 10.0) -> str:
        conn = self._conn(self.server_key(server))
        fut = asyncio.run_coroutine_threadsafe(conn.sessions[0].read_resource(uri), self._loop)  # type: ignore[arg-type]
        try:
            res = fut.result(timeout_s)
        except Exception as err:
            raise ToolError(ErrorCode.UPSTREAM_TIMEOUT, f"resource read failed: {type(err).__name__}") from None
        return res.contents[0].text  # type: ignore[union-attr]

    def list_tools(self, server: str) -> list[types.Tool]:
        return list(self._conn(self.server_key(server)).tools)

    # ---- introspection -------------------------------------------------------------
    def _log(self, key: str, tool: str, request: Any, response: Any, is_error: bool, rtt: float) -> None:
        self._payloads.append({"ts": time.time(), "server": key, "tool": tool, "rtt_ms": round(rtt, 2),
                               "is_error": bool(is_error),
                               "request": {"jsonrpc": "2.0", "method": "tools/call",
                                           "params": {"name": tool, "arguments": request}},
                               "response": json.dumps(response, default=str)[:4000]})

    def payloads(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._payloads)[-limit:][::-1]

    def status(self) -> dict[str, Any]:
        out = {}
        for key, conn in self._conns.items():
            rtt = sorted(conn.stats.rtt_ms)
            over = sorted(conn.stats.overhead_ms)
            pct = lambda xs, q: round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None  # noqa: E731
            out[key] = {
                "transport": conn.cfg["transport"], "connected": bool(conn.sessions), "error": conn.error,
                "tools": [t.name for t in conn.tools], "manifest_verified": conn.manifest_ok,
                "calls": conn.stats.calls, "errors": conn.stats.errors, "timeouts": conn.stats.timeouts,
                "rtt_ms_p50": pct(rtt, 0.5), "rtt_ms_p95": pct(rtt, 0.95),
                "overhead_ms_p50": pct(over, 0.5), "overhead_ms_p95": pct(over, 0.95),
                "timeout_ms": conn.cfg.get("timeout_ms"), "target_timeout_ms": conn.cfg.get("target_timeout_ms"),
                "last_error": conn.stats.last_error,
            }
        return out

    def timeout_rate(self, key: str) -> float:
        s = self._conns[key].stats
        total = s.calls + s.timeouts
        return s.timeouts / total if total else 0.0


try:   # concurrent.futures.TimeoutError is its own class before Python 3.11
    from concurrent.futures import TimeoutError as FutureTimeout
except ImportError:  # pragma: no cover
    FutureTimeout = TimeoutError  # type: ignore[misc,assignment]


def _to_tool_error(text: str) -> ToolError:
    try:
        body = json.loads(text)
        return ToolError(ErrorCode(int(body["code"])), str(body.get("message", "")), dict(body.get("data") or {}))
    except (ValueError, KeyError, TypeError):
        return ToolError(ErrorCode.INTERNAL_ERROR, text[:200])
