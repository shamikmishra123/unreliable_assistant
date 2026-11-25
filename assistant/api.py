"""FastAPI surface (design §1.2.6, ADR-011, ADR-013, ADR-018).

Public:
- POST /api/v1/agent/query         JSON answer. X-Tenant-ID must equal the body tenant_id, else 403.
- POST /api/v1/agent/query/stream  SSE. One `released` event per verified sentence as it is verified, then `done`.
- GET  /api/v1/approvals           reviewer: the tenant's approval queue
- GET  /api/v1/approvals/{id}      tenant-scoped status
- POST /api/v1/approvals/{id}/decide  reviewer only (bearer token from ASSISTANT_REVIEWER_TOKENS)
- GET  /api/v1/audit/verify        reviewer only; recomputes the hash chain
- GET  /healthz, /metrics
Platform engineer (header X-Admin-Token == ASSISTANT_ADMIN_TOKEN; the routes do not exist without a token):
- /admin/overview, /admin/traces, /admin/payloads, /admin/plugins/{name}, /admin/tools/call, /admin/breaker,
  /admin/incidents, /admin/audit
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings, load_settings
from .contracts import APPROVAL_RE, TENANT_RE, USER_RE, CallContext, ErrorCode, ToolError
from .harness import Tier
from .orchestrator import Orchestrator
from .plugins import PluginConfigError
from .services import Services, build_services


class QueryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: str = Field(pattern=TENANT_RE)
    user_id: str = Field(pattern=USER_RE)
    session_id: UUID
    query: str = Field(min_length=1, max_length=2000)


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["APPROVE", "REJECT"]
    reason: str = Field(default="", max_length=512)


class ToolCallIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: str = Field(pattern=TENANT_RE)
    user_id: str = Field(pattern=USER_RE)
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    tier: Literal["TIER_1_READ", "TIER_2_COMPUTE", "TIER_3_MUTATE"] = "TIER_2_COMPUTE"
    idempotency_key: UUID | None = None


class PluginToggleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


def _principal(settings: Settings, authorization: str | None) -> dict[str, Any] | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    presented = authorization[len("Bearer "):].strip()
    for token, principal in settings.reviewer_tokens.items():
        if hmac.compare_digest(token, presented):
            return principal
    return None


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    holder: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or load_settings()
        svc = services or await asyncio.to_thread(build_services, cfg, mode=cfg.mcp_mode)
        holder["svc"] = svc
        app.state.orchestrator = Orchestrator(svc)
        task = None
        if os.environ.get("INCIDENT_RESPONDER", "1") != "0":
            async def watch() -> None:
                while True:
                    await asyncio.sleep(5)
                    await asyncio.to_thread(svc.incidents.evaluate)
            task = asyncio.create_task(watch())
        yield
        if task:
            task.cancel()
        if services is None:
            await asyncio.to_thread(svc.close)
        else:
            svc.store.flush()

    app = FastAPI(title="Unreliable Assistant", version="0.2.0", lifespan=lifespan)

    def svc() -> Services:
        return holder["svc"]

    def require_tenant(header_tenant: str | None, body_tenant: str) -> None:
        if header_tenant is None or header_tenant != body_tenant:
            raise HTTPException(status_code=403, detail="X-Tenant-ID must match tenant_id")

    def admin(token: str | None) -> Services:
        s = svc()
        expected = s.settings.admin_token
        if not expected or not token or not hmac.compare_digest(expected, token):
            raise HTTPException(status_code=404, detail="not found")   # the admin surface is invisible without a token
        return s

    # ---- public ------------------------------------------------------------------------------
    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        s = svc()
        usage = s.generator.usage.snapshot()
        return {"counters": s.metrics.snapshot(), "cache_entries": len(s.cache), "cache_hits": s.cache.hits,
                "cache_misses": s.cache.misses, "usage": usage, "spans": s.tracer.all_stats(),
                "tuning": s.runtime.tuning.snapshot(), "pools": s.pools.stats(), **s.info()}

    @app.post("/api/v1/agent/query")
    async def query(body: QueryIn, x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID")):
        started = time.perf_counter()
        require_tenant(x_tenant_id, body.tenant_id)
        resp = await app.state.orchestrator.handle(
            tenant_id=body.tenant_id, user_id=body.user_id, session_id=body.session_id, query=body.query,
            gateway_ms=(time.perf_counter() - started) * 1000)
        resp.pop("_released", None)
        return resp

    @app.post("/api/v1/agent/query/stream")
    async def query_stream(body: QueryIn, x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID")):
        started = time.perf_counter()
        require_tenant(x_tenant_id, body.tenant_id)
        events = app.state.orchestrator.handle_stream(
            tenant_id=body.tenant_id, user_id=body.user_id, session_id=body.session_id, query=body.query,
            gateway_ms=(time.perf_counter() - started) * 1000)

        async def sse() -> AsyncIterator[str]:
            first = True
            async for kind, payload in events:
                if kind == "released":
                    if first:
                        first = False
                        svc().tracer.add_span("e2e.first_verified", (time.perf_counter() - started) * 1000)
                    yield "event: released\ndata: " + json.dumps(payload) + "\n\n"
                else:
                    payload.pop("_released_fallback", None)
                    yield "event: done\ndata: " + json.dumps(payload) + "\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ---- HITL ---------------------------------------------------------------------------------
    @app.get("/api/v1/approvals")
    async def list_approvals(status: str | None = None, authorization: str | None = Header(default=None)):
        s = svc()
        principal = _principal(s.settings, authorization)
        if principal is None or principal.get("role") != "reviewer":
            raise HTTPException(status_code=403, detail="reviewer role required")
        rows = await asyncio.to_thread(s.store.list_approvals, principal["tenant_id"], status, 100)
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return {"approvals": rows}

    @app.get("/api/v1/approvals/{approval_id}")
    async def get_approval(approval_id: str, x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID")):
        if x_tenant_id is None or not re.match(APPROVAL_RE, approval_id):
            raise HTTPException(status_code=404, detail="not found")
        row = await asyncio.to_thread(svc().store.approval_view, x_tenant_id, approval_id)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")
        return row

    @app.post("/api/v1/approvals/{approval_id}/decide")
    async def decide(approval_id: str, body: DecisionIn, authorization: str | None = Header(default=None)):
        s = svc()
        principal = _principal(s.settings, authorization)
        if principal is None:
            raise HTTPException(status_code=401, detail="reviewer token required")
        if principal.get("role") != "reviewer":
            raise HTTPException(status_code=403, detail="reviewer role required")
        if not re.match(APPROVAL_RE, approval_id):
            raise HTTPException(status_code=404, detail="not found")
        try:
            return await asyncio.to_thread(s.store.decide, principal["tenant_id"], approval_id,
                                           principal["user_id"], body.decision, body.reason)
        except ToolError as err:
            http = err.data.get("http")
            if http == 403 or err.code == ErrorCode.RBAC_DENIED:
                raise HTTPException(status_code=403, detail=err.message) from None
            if http == 409:
                raise HTTPException(status_code=409, detail=err.message) from None
            if err.code == ErrorCode.INVALID_PARAMS:
                raise HTTPException(status_code=404, detail="not found") from None
            raise HTTPException(status_code=503, detail="state store unavailable") from None

    @app.get("/api/v1/audit/verify")
    async def verify_audit(authorization: str | None = Header(default=None)):
        s = svc()
        principal = _principal(s.settings, authorization)
        if principal is None or principal.get("role") != "reviewer":
            raise HTTPException(status_code=403, detail="reviewer role required")
        ok, rows = await asyncio.to_thread(s.store.verify_audit)
        return {"ok": ok, "rows": rows}

    # ---- platform engineer ---------------------------------------------------------------------
    @app.get("/admin/overview")
    def overview(x_admin_token: str | None = Header(default=None)):
        s = admin(x_admin_token)
        return {"info": s.info(), "mcp": s.host.status() if s.host else {}, "plugins": s.plugins.describe(),
                "pools": s.pools.stats(), "tuning": s.runtime.tuning.snapshot(), "incidents": s.incidents.view(),
                "breaker": s.harness.breaker_state(), "counters": s.metrics.snapshot(),
                "spans": s.tracer.all_stats(), "usage": s.generator.usage.snapshot(),
                "cache": {"entries": len(s.cache), "hits": s.cache.hits, "misses": s.cache.misses},
                "otel": {"exporting": s.tracer.exporter is not None}}

    @app.get("/admin/traces")
    def traces(limit: int = 20, x_admin_token: str | None = Header(default=None)):
        return {"traces": admin(x_admin_token).tracer.recent(min(limit, 100))}

    @app.get("/admin/payloads")
    def payloads(limit: int = 30, x_admin_token: str | None = Header(default=None)):
        s = admin(x_admin_token)
        return {"payloads": s.host.payloads(min(limit, 100)) if s.host else []}

    @app.post("/admin/plugins/{name}")
    def toggle_plugin(name: str, body: PluginToggleIn, x_admin_token: str | None = Header(default=None)):
        s = admin(x_admin_token)
        try:
            s.plugins.set_enabled(name, body.enabled)
        except PluginConfigError as err:
            raise HTTPException(status_code=409, detail=str(err)) from None
        return {"plugins": s.plugins.describe()}

    @app.post("/admin/tools/call")
    async def tool_call(body: ToolCallIn, x_admin_token: str | None = Header(default=None)):
        s = admin(x_admin_token)
        ctx = CallContext(tenant_id=body.tenant_id, user_id=body.user_id, session_id=UUID(int=0, version=4),
                          idempotency_key=body.idempotency_key or __import__("uuid").uuid4(),
                          traceparent="00-" + "0" * 31 + "1-" + "0" * 15 + "1-01")
        try:
            result = await asyncio.to_thread(s.harness.call, body.tool, body.params, ctx, Tier[body.tier])
            return {"ok": True, "result": result}
        except ToolError as err:
            return {"ok": False, "error": err.to_jsonrpc()["error"]}

    @app.get("/admin/breaker")
    def breaker(x_admin_token: str | None = Header(default=None)):
        return {"breaker": admin(x_admin_token).harness.breaker_state()}

    @app.post("/admin/breaker/reset")
    def breaker_reset(x_admin_token: str | None = Header(default=None)):
        s = admin(x_admin_token)
        return {"reset": s.harness.reset_breakers()}

    @app.post("/admin/incidents/evaluate")
    async def incidents_evaluate(x_admin_token: str | None = Header(default=None)):
        s = admin(x_admin_token)
        opened = await asyncio.to_thread(s.incidents.evaluate)
        return {"opened": [i.view() for i in opened], "incidents": s.incidents.view()}

    @app.post("/admin/incidents/{incident_id}/resolve")
    def incident_resolve(incident_id: str, x_admin_token: str | None = Header(default=None)):
        s = admin(x_admin_token)
        if not s.incidents.resolve(incident_id):
            raise HTTPException(status_code=404, detail="no such open incident")
        return {"incidents": s.incidents.view()}

    @app.get("/admin/audit")
    async def audit(tenant_id: str | None = None, limit: int = 100, x_admin_token: str | None = Header(default=None)):
        s = admin(x_admin_token)
        events = await asyncio.to_thread(s.store.audit_events, tenant_id, min(limit, 500))
        return {"events": events}

    return app


def __getattr__(name: str) -> Any:  # pragma: no cover - lazily expose `app` for `uvicorn assistant.api:app`
    if name == "app":
        return create_app()
    raise AttributeError(name)
