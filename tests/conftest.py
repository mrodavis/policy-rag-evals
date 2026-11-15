import datetime as dt
from pathlib import Path

import pytest

from policy_rag.index import build_index

ROOT = Path(__file__).resolve().parents[1]
TODAY = dt.date(2026, 11, 15)


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    return ROOT / "corpus"


@pytest.fixture(scope="session")
def index(corpus_dir):
    return build_index(corpus_dir, TODAY)
