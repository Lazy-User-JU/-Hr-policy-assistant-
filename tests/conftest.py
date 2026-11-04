import pytest

from app import config, store
from app.ingest import index_document


@pytest.fixture(autouse=True)
def fresh_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "hashing")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    store.init()


def index(name: str, text: str) -> None:
    index_document(name, text.encode())
