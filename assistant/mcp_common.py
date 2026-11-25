"""Server-side MCP plumbing (design §1.2, ADR-005): exposes ToolSpecs over JSON-RPC 2.0 with the exact
Pydantic schemas, strict re-validation, tier/HITL metadata, and JSON-RPC style error codes."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from pydantic import ValidationError

from .contracts import ErrorCode, ToolError
from .harness import ToolSpec


def tool_definition(spec: ToolSpec) -> types.Tool:
    return types.Tool(
        name=spec.name, description=(spec.handler.__doc__ or f"{spec.name} ({spec.tier.name})").strip(),
        inputSchema=spec.input_model.model_json_schema(),
        _meta={"tier": spec.tier.name, "hitl_gate": spec.hitl_gate, "internal": spec.internal,
               "server": spec.server})


def error_result(code: ErrorCode, message: str, data: dict[str, Any] | None = None) -> types.CallToolResult:
    body = {"code": int(code), "name": code.name, "message": message, "data": data or {}}
    return types.CallToolResult(isError=True, content=[types.TextContent(type="text", text=json.dumps(body))])


def build_server(name: str, specs: list[ToolSpec], *, resource_reader: Callable[[str], str] | None = None,
                 resource_templates: list[types.ResourceTemplate] | None = None,
                 resource_info: dict[str, Callable[[], str]] | None = None) -> Server:
    server: Server = Server(name)
    by_name = {s.name: s for s in specs}

    @server.list_tools()
    async def _list() -> list[types.Tool]:
        return [tool_definition(s) for s in specs]

    @server.call_tool(validate_input=False)
    async def _call(tool: str, arguments: dict[str, Any]):
        spec = by_name.get(tool)
        if spec is None:
            return error_result(ErrorCode.METHOD_NOT_FOUND, f"unknown tool {tool}")
        try:   # JSON-mode strict validation: UUID/str coercions allowed only as JSON allows them
            model = spec.input_model.model_validate_json(json.dumps(arguments))
        except ValidationError as err:
            return error_result(ErrorCode.SCHEMA_VALIDATION_FAILED, "arguments failed validation",
                                {"fields": [".".join(str(p) for p in e["loc"]) for e in err.errors()][:5]})
        started = time.perf_counter()
        try:
            result = await anyio.to_thread.run_sync(spec.handler, model)
        except ToolError as err:
            return error_result(err.code, err.message, err.data)
        except Exception:   # never leak internals across the transport
            return error_result(ErrorCode.INTERNAL_ERROR, "internal error")
        handler_ms = (time.perf_counter() - started) * 1000
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(result, default=str))],
                                    structuredContent=json.loads(json.dumps(result, default=str)),
                                    _meta={"handler_ms": handler_ms})

    if resource_reader is not None:
        @server.list_resource_templates()
        async def _templates() -> list[types.ResourceTemplate]:
            return resource_templates or []

        @server.list_resources()
        async def _resources() -> list[types.Resource]:
            return [types.Resource(uri=uri, name=uri.rsplit("/", 1)[-1], mimeType="application/json")
                    for uri in (resource_info or {})]

        @server.read_resource()
        async def _read(uri: Any):
            from mcp.server.lowlevel.helper_types import ReadResourceContents

            text = await anyio.to_thread.run_sync(resource_reader, str(uri))
            return [ReadResourceContents(content=text, mime_type="application/json")]

    return server
