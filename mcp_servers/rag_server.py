"""mcp-hybrid-rag (stdio). Tool: hybrid_retrieve. Resources: rag://chunks/{chunk_hash}?tenant_id=..., rag://index/info.

    python mcp_servers/rag_server.py          # serve over stdio
    python mcp_servers/rag_server.py build    # index-time scan: report chunks and quarantined chunks (ADR-010)
"""
import _bootstrap  # noqa: F401  (sets sys.path, logging to stderr)

import json
import sys
from urllib.parse import parse_qs, urlparse

import anyio
from mcp import types
from mcp.server.stdio import stdio_server

from assistant.config import load_settings
from assistant.contracts import CHUNK_RE, TENANT_RE, ErrorCode, ToolError
from assistant.mcp_common import build_server
from assistant.rag.index import build_index
from assistant.services import get_models
from assistant.tools import build_rag_tools
import re


def make_reader(index):
    def read(uri: str) -> str:
        parsed = urlparse(uri)
        if uri.startswith("rag://index/info"):
            return json.dumps({"tenants": sorted(index.tenants),
                               "chunks": {t: len(i.chunks) for t, i in index.tenants.items()},
                               "quarantined_chunks": len(index.load.quarantined), "notes": len(index.load.notes),
                               "dense_model": index.models.dense_name, "rerank_model": index.models.rerank_name})
        if parsed.scheme == "rag" and parsed.netloc == "chunks":
            digest = parsed.path.lstrip("/")
            tenant = (parse_qs(parsed.query).get("tenant_id") or [""])[0]
            if not re.match(CHUNK_RE, digest) or not re.match(TENANT_RE, tenant):
                raise ToolError(ErrorCode.INVALID_PARAMS, "resource needs a chunk hash and tenant_id")
            chunk = index.chunk(tenant, digest)            # tenant ACL on every read (design §1.2.2)
            return json.dumps({"chunk_hash": chunk.chunk_hash, "uri": chunk.uri, "text": chunk.text,
                               "note_id": chunk.note_id})
        raise ToolError(ErrorCode.INVALID_PARAMS, "unknown resource")
    return read


async def serve() -> None:
    settings = load_settings()
    index = build_index(settings.vault_dir, get_models(settings))
    server = build_server(
        "mcp-hybrid-rag", build_rag_tools(index), resource_reader=make_reader(index),
        resource_templates=[types.ResourceTemplate(uriTemplate="rag://chunks/{chunk_hash}", name="chunk",
                                                   mimeType="application/json")],
        resource_info={"rag://index/info": lambda: ""})
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def build_report() -> int:
    from assistant.rag.corpus import load_vault
    from assistant.security import is_poisoned

    load = load_vault(load_settings().vault_dir, is_poisoned)
    print(json.dumps({"notes": len(load.notes), "chunks": len(load.chunks), "quarantined": load.quarantined}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        sys.exit(build_report())
    anyio.run(serve)
