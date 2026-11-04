import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, ingest, store
from app.main import app

ADMIN = {"X-Role": "admin"}
V1 = b"# Leave\n\n## Carry forward\n\nA maximum of 8 days of casual leave may be carried forward.\n"
V2 = b"# Leave\n\n## Carry forward\n\nA maximum of 5 days of casual leave may be carried forward.\n"
QUESTION = {"question": "How many days of casual leave may be carried forward?"}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def upload(client, name="leave.md", data=V1, headers=ADMIN):
    return client.post("/documents", files={"file": (name, data)}, headers=headers)


def test_upload_then_ask_returns_structured_answer(client):
    assert upload(client).status_code == 201
    body = client.post("/ask", json=QUESTION).json()
    assert body["status"] == "answered"
    assert "8 days" in body["answer"]
    assert body["citations"][0]["document"] == "leave.md"
    assert body["citations"][0]["section"] == "Carry forward"


def test_employee_cannot_upload(client):
    assert upload(client, headers={}).status_code == 403


def test_same_file_twice_is_not_indexed_again(client):
    upload(client)
    body = upload(client).json()
    assert (body["status"], body["version"]) == ("unchanged", 1)


def test_new_version_replaces_the_old_one(client):
    upload(client)
    assert upload(client, data=V2).json()["version"] == 2
    body = client.post("/ask", json=QUESTION).json()
    assert "5 days" in body["answer"]
    assert body["citations"][0]["version"] == 2
    assert len(client.get("/documents").json()) == 1


def test_failed_embedding_keeps_the_old_version_live(client, monkeypatch):
    upload(client)

    def broken(texts):
        raise RuntimeError("model missing")

    monkeypatch.setattr(ingest, "embed", broken)
    assert upload(client, data=V2).status_code == 503
    body = client.post("/ask", json=QUESTION).json()
    assert "8 days" in body["answer"]


def test_changing_the_embedding_model_rebuilds_vectors(client, monkeypatch):
    upload(client)
    monkeypatch.setattr(store, "embed", lambda texts: np.ones((len(texts), 3), dtype="float32"))
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "another-model")
    store.init()
    assert np.frombuffer(store.active_chunks()[0]["embedding"], dtype="float32").shape == (3,)


def test_deleted_document_is_no_longer_searched(client):
    upload(client)
    assert client.delete("/documents/leave.md", headers=ADMIN).status_code == 204
    assert client.post("/ask", json=QUESTION).json()["refusal_reason"] == "no_documents"
    assert client.delete("/documents/leave.md", headers=ADMIN).status_code == 404


@pytest.mark.parametrize("question", ["", "    "])
def test_empty_question_is_rejected(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_too_long_question_is_rejected(client):
    assert client.post("/ask", json={"question": "a" * 1001}).status_code == 422


@pytest.mark.parametrize("name, data, code", [
    ("photo.png", b"data", 415),
    ("empty.md", b"", 422),
    ("binary.txt", b"\x00\x01", 422),
    ("broken.pdf", b"not a pdf", 422),
    ("blank-table.md", b"| a |\n| --- |\n|  |\n", 422),
])
def test_bad_uploads_get_a_clear_error(client, name, data, code):
    assert upload(client, name=name, data=data).status_code == code


def test_home_page_loads(client):
    assert "HR Policy Assistant" in client.get("/").text
