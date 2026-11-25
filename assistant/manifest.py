"""Signed MCP server manifests (Gate 13, ADR-014): per server, tool names, tiers, HITL flags and the SHA-256 of each
input schema, signed with Ed25519. The public key and manifests are in git; the private key is in var/ (git-ignored).

    python -m assistant.manifest sign     # key holder only; re-sign after any tool contract change
    python -m assistant.manifest verify    # check the committed manifests against the current contracts
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .config import ROOT
from .harness import ToolSpec

MANIFEST_DIR = ROOT / "mcp_servers" / "manifests"
PUBLIC_KEY = ROOT / "mcp_servers" / "manifest.pub"
PRIVATE_KEY = ROOT / "var" / "manifest_key.pem"


class ManifestError(Exception):
    pass


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def schema_hash(schema: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(schema)).hexdigest()


def entry_from_spec(spec: ToolSpec) -> dict[str, Any]:
    return {"name": spec.name, "tier": spec.tier.name, "hitl_gate": spec.hitl_gate, "internal": spec.internal,
            "input_schema_sha256": schema_hash(spec.input_model.model_json_schema())}


def entry_from_tool(tool: Any) -> dict[str, Any]:
    meta = tool.meta or {}
    return {"name": tool.name, "tier": meta.get("tier"), "hitl_gate": bool(meta.get("hitl_gate")),
            "internal": bool(meta.get("internal")), "input_schema_sha256": schema_hash(tool.inputSchema)}


def build_manifest(server: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"server": server, "tools": sorted(entries, key=lambda e: e["name"])}


def sign_manifest(manifest: dict[str, Any], key: Ed25519PrivateKey) -> dict[str, Any]:
    return {"manifest": manifest, "algorithm": "ed25519", "signature": key.sign(canonical(manifest)).hex()}


def load_public_key(path: Path = PUBLIC_KEY) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(path.read_text().strip()))


def verify_signed(signed: dict[str, Any], public_key: Ed25519PublicKey) -> dict[str, Any]:
    try:
        public_key.verify(bytes.fromhex(signed["signature"]), canonical(signed["manifest"]))
    except (InvalidSignature, KeyError, ValueError):
        raise ManifestError("manifest signature is invalid") from None
    return signed["manifest"]


def verify_server_tools(server: str, tools: list[Any], manifest_dir: Path = MANIFEST_DIR,
                        public_key_path: Path = PUBLIC_KEY) -> None:
    """Refuse a server whose announced tools differ from the signed contract (added, removed or drifted)."""
    path = manifest_dir / f"{server}.json"
    if not path.exists() or not public_key_path.exists():
        raise ManifestError(f"{server}: no signed manifest")
    signed_manifest = verify_signed(json.loads(path.read_text()), load_public_key(public_key_path))
    live = build_manifest(server, [entry_from_tool(t) for t in tools])
    if canonical(live) != canonical(signed_manifest):
        want = {e["name"]: e for e in signed_manifest["tools"]}
        have = {e["name"]: e for e in live["tools"]}
        drift = sorted(set(want) ^ set(have)) + sorted(n for n in want.keys() & have.keys() if want[n] != have[n])
        raise ManifestError(f"{server}: tool contract drifted from the signed manifest: {drift}")


def _load_or_create_key(create: bool) -> Ed25519PrivateKey:
    if PRIVATE_KEY.exists():
        return serialization.load_pem_private_key(PRIVATE_KEY.read_bytes(), password=None)  # type: ignore[return-value]
    if not create:
        raise ManifestError(f"no private key at {PRIVATE_KEY}")
    key = Ed25519PrivateKey.generate()
    PRIVATE_KEY.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_KEY.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption()))
    PRIVATE_KEY.chmod(0o600)
    return key


def specs_by_server() -> dict[str, list[ToolSpec]]:
    from .tools import contract_specs

    grouped: dict[str, list[ToolSpec]] = {}
    for spec in contract_specs():
        grouped.setdefault(spec.server, []).append(spec)
    return grouped


def sign_all(create_key: bool = True) -> list[str]:
    key = _load_or_create_key(create_key)
    PUBLIC_KEY.write_text(key.public_key().public_bytes(serialization.Encoding.Raw,
                                                        serialization.PublicFormat.Raw).hex() + "\n")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for server, specs in specs_by_server().items():
        signed = sign_manifest(build_manifest(server, [entry_from_spec(s) for s in specs]), key)
        (MANIFEST_DIR / f"{server}.json").write_text(json.dumps(signed, indent=1, sort_keys=True) + "\n")
        written.append(server)
    return written


def verify_all() -> list[str]:
    """Signed manifests vs the current in-code contracts. Returns problems (empty = in sync)."""
    problems = []
    pub = load_public_key()
    for server, specs in specs_by_server().items():
        path = MANIFEST_DIR / f"{server}.json"
        if not path.exists():
            problems.append(f"{server}: manifest missing")
            continue
        try:
            signed = verify_signed(json.loads(path.read_text()), pub)
        except ManifestError as err:
            problems.append(f"{server}: {err}")
            continue
        live = build_manifest(server, [entry_from_spec(s) for s in specs])
        if canonical(live) != canonical(signed):
            problems.append(f"{server}: contract differs from the signed manifest (re-sign)")
    return problems


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "sign":
        print("signed:", ", ".join(sign_all()))
    else:
        issues = verify_all()
        print("manifests OK" if not issues else "\n".join(issues))
        sys.exit(1 if issues else 0)
