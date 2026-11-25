"""mcp-state-store (Streamable HTTP, http://127.0.0.1:8091/mcp). Public tools + internal host-only tools.

Every request needs `Authorization: Bearer <token>` (env MCP_STATE_TOKEN, else var/state_token, generated on first start).
"""
import _bootstrap  # noqa: F401

import contextlib
import hmac
import os
import secrets
import sys

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from assistant.config import load_settings, state_token
from assistant.mcp_common import build_server
from assistant.rag.corpus import ChunkCatalog, load_vault
from assistant.security import is_poisoned
from assistant.state import StateStore
from assistant.state_internal import build_internal_tools
from assistant.tools import build_state_tools


def make_app(token: str) -> Starlette:
    settings = load_settings()
    state = StateStore(settings.db_path, settings.approval_ttl_s)
    catalog = ChunkCatalog(load_vault(settings.vault_dir, is_poisoned))
    server = build_server("mcp-state-store", build_state_tools(state, catalog.has_chunk) + build_internal_tools(state))
    manager = StreamableHTTPSessionManager(app=server, json_response=False, stateless=False)

    class Authenticated:
        async def __call__(self, scope, receive, send):
            headers = dict(scope["headers"])
            presented = headers.get(b"authorization", b"").decode()
            if not hmac.compare_digest(presented, f"Bearer {token}"):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
            await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield
        state.close()

    return Starlette(routes=[Route("/mcp", endpoint=Authenticated())], lifespan=lifespan)


if __name__ == "__main__":
    port = int(os.environ.get("MCP_STATE_PORT", "8091"))
    uvicorn.run(make_app(state_token()), host="127.0.0.1", port=port, log_level="warning")
