"""Graphify (Gate 17): build graph.json from ./assistant (code layer) and ./knowledge_vault (vault layer).

    python -m assistant.graphify build assistant knowledge_vault --output graph.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .graph import Graph
from .rag.corpus import parse_note


def vault_digest(vault_dir: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(vault_dir.glob("*.md")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build_code_layer(src_dir: Path) -> dict[str, Any]:
    """Modules, their intra-package imports and Pydantic schema classes, from the AST (no execution)."""
    modules: dict[str, Any] = {}
    root = src_dir.resolve()
    pkg = root.name
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")
        name = ".".join((pkg, *rel.parts)).removesuffix(".__init__")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: list[str] = []
        schemas: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.module is not None:
                base = name.split(".")[:-node.level] if not path.name == "__init__.py" else name.split(".")[:-node.level + 1]
                imports.append(".".join([*base, node.module]))
            elif isinstance(node, ast.ImportFrom) and node.level and node.module is None:
                imports.extend(".".join([*name.split(".")[:-node.level], a.name]) for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(pkg):
                imports.append(node.module)
            elif isinstance(node, ast.ClassDef):
                bases = {getattr(b, "id", getattr(b, "attr", "")) for b in node.bases}
                if bases & {"BaseModel", "_Strict", "_S"}:
                    schemas.append(node.name)
        modules[name] = {"path": str(path.relative_to(root.parent)), "imports": sorted(set(imports)),
                         "schemas": sorted(schemas)}
    return {"modules": modules}


def build(src_dirs: list[Path], vault_dir: Path, output: Path) -> Graph:
    notes = {}
    for path in sorted(vault_dir.glob("*.md")):
        note = parse_note(path)
        notes[note.note_id] = note
    code: dict[str, Any] = {"modules": {}}
    for src in src_dirs:
        code["modules"].update(build_code_layer(src)["modules"])
    graph = Graph.from_notes(notes, code=code, vault_digest=vault_digest(vault_dir))
    output.write_text(json.dumps(graph.to_json(), indent=1, sort_keys=True), encoding="utf-8")
    return graph


def load_or_build(graph_path: Path, vault_dir: Path) -> tuple[Graph, str]:
    """Use graph.json when its vault digest matches; otherwise rebuild the vault layer in memory (never serve stale)."""
    digest = vault_digest(vault_dir)
    if graph_path.exists():
        graph = Graph.from_json(graph_path)
        if graph.vault_digest == digest:
            return graph, "graph.json"
    notes = {n.note_id: n for n in (parse_note(p) for p in sorted(vault_dir.glob("*.md")))}
    return Graph.from_notes(notes, vault_digest=digest), "rebuilt-in-memory (graph.json missing or stale)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="graphify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("src", nargs="+", help="source dirs followed by the vault dir (last positional)")
    b.add_argument("--output", default="graph.json")
    args = ap.parse_args(argv)
    *srcs, vault = [Path(p) for p in args.src]
    graph = build(srcs, vault, Path(args.output))
    print(f"graph.json: {len(graph.nodes)} vault nodes, {sum(len(v) for v in graph.edges.values())} edges, "
          f"{len(graph.code['modules'])} code modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
