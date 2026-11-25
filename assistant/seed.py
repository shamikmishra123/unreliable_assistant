"""Demo lot records. Run: python -m assistant.seed  (idempotent upserts)."""
from __future__ import annotations

from .config import load_settings
from .state import StateStore

LOTS = [
    ("tenant_a", "LOT-00044", "AVAILABLE", "PROT-SAF-7"),
    ("tenant_a", "LOT-88392", "AVAILABLE", "PROT-SAF-7"),
    ("tenant_a", "LOT-00099", "QUARANTINED", None),
    ("tenant_a", "LOT-00120", "ON_HOLD", "PROT-SAF-7"),
    ("tenant_b", "LOT-00044", "RELEASED", "PROT-RIV-2"),
    ("tenant_b", "LOT-00500", "AVAILABLE", "PROT-RIV-2"),
]


def seed(state: StateStore) -> int:
    for tenant, lot, status, protocol in LOTS:
        state.upsert_lot(tenant, lot, status, protocol)
    return len(LOTS)


if __name__ == "__main__":
    settings = load_settings()
    store = StateStore(settings.db_path)
    print(f"seeded {seed(store)} lots into {settings.db_path}")
    store.close()
