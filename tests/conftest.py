import dataclasses
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.config import Settings  # noqa: E402
from assistant.llm import StubGenerator  # noqa: E402
from assistant.seed import seed  # noqa: E402
from assistant.services import build_services, get_models  # noqa: E402

REVIEWERS = {
    "tok-rev-a": {"user_id": "user_reva", "tenant_id": "tenant_a", "role": "reviewer"},
    "tok-rev-b": {"user_id": "user_revb", "tenant_id": "tenant_b", "role": "reviewer"},
    "tok-op": {"user_id": "user_op1", "tenant_id": "tenant_a", "role": "operator"},
}


@pytest.fixture(scope="session")
def models():
    # Loads bge-large and the MiniLM reranker once per test session (both cached locally).
    return get_models(Settings(vault_dir=ROOT / "knowledge_vault"))


@pytest.fixture
def make_services(tmp_path, models):
    created = []

    def _make(generator=None, name="state.db"):
        settings = dataclasses.replace(
            Settings(vault_dir=ROOT / "knowledge_vault", db_path=tmp_path / name,
                     approval_ttl_s=24 * 3600.0, verifier="lexical"),
            reviewer_tokens=REVIEWERS)
        svc = build_services(settings, generator=generator or StubGenerator(), models=models)
        seed(svc.state)
        created.append(svc)
        return svc

    yield _make
    for svc in created:
        svc.state.close()


@pytest.fixture
def svc(make_services):
    return make_services()
