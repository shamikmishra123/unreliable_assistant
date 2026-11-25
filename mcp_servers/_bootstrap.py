"""Shared start-up for server scripts: make the repo importable and keep stdout clean (stdio = JSON-RPC)."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
