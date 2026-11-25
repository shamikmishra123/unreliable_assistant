"""mcp-graphify-vault (stdio). Tool: graph_subgraph. Resource: graph://stats. Loads the precomputed graph.json."""
import _bootstrap  # noqa: F401

import json

import anyio
from mcp.server.stdio import stdio_server

from assistant.config import ROOT, load_settings
from assistant.graphify import load_or_build
from assistant.mcp_common import build_server
from assistant.tools import build_graph_tools


async def serve() -> None:
    settings = load_settings()
    graph, source = load_or_build(ROOT / "graph.json", settings.vault_dir)

    def read(uri: str) -> str:
        return json.dumps({"source": source, "vault_nodes": len(graph.nodes),
                           "edges": sum(len(v) for v in graph.edges.values()),
                           "code_modules": len(graph.code.get("modules", {}))})

    server = build_server("mcp-graphify-vault", build_graph_tools(graph), resource_reader=read,
                          resource_info={"graph://stats": lambda: ""})
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(serve)
