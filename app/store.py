import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

from app import config
from app.chunking import Chunk
from app.embeddings import embed

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    title TEXT NOT NULL,
    applies_to TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    active INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL,
    UNIQUE (name, version)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents (id),
    section TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    search_text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    search_text, chunk_id UNINDEXED, tokenize = 'porter unicode61'
);
"""

CHUNK_COLUMNS = """
    c.id, c.section, c.kind, c.text, c.search_text, c.embedding,
    d.name AS document, d.title, d.applies_to, d.version
"""


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init() -> None:
    with connect() as db:
        db.executescript(SCHEMA)
        saved = db.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
        rows = db.execute("SELECT id, search_text FROM chunks").fetchall()

        if saved and saved["value"] != config.EMBEDDING_MODEL and rows:
            vectors = embed([row["search_text"] for row in rows])
            db.executemany(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                [(vector.tobytes(), row["id"]) for row, vector in zip(rows, vectors)],
            )
        db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_model', ?)",
            (config.EMBEDDING_MODEL,),
        )


def active_version(name: str):
    with connect() as db:
        return db.execute(
            "SELECT * FROM documents WHERE name = ? AND active = 1", (name,)
        ).fetchone()


def save_document(
    name: str, title: str, applies_to: str, sha256: str, chunks: list[Chunk], vectors: np.ndarray
) -> int:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect() as db:
        latest = db.execute(
            "SELECT MAX(version) AS version FROM documents WHERE name = ?", (name,)
        ).fetchone()
        version = (latest["version"] or 0) + 1
        db.execute("UPDATE documents SET active = 0 WHERE name = ?", (name,))
        document_id = db.execute(
            "INSERT INTO documents (name, title, applies_to, version, sha256, active, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?)",
            (name, title, applies_to, version, sha256, now),
        ).lastrowid
        for chunk, vector in zip(chunks, vectors):
            chunk_id = db.execute(
                "INSERT INTO chunks (document_id, section, kind, text, search_text, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (document_id, chunk.section, chunk.kind, chunk.text, chunk.search_text, vector.tobytes()),
            ).lastrowid
            db.execute(
                "INSERT INTO chunks_fts (search_text, chunk_id) VALUES (?, ?)",
                (chunk.search_text, chunk_id),
            )
    return version


def list_documents() -> list[dict]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT d.name, d.title, d.version, d.uploaded_at, COUNT(c.id) AS chunks
            FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.active = 1 GROUP BY d.id ORDER BY d.name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def deactivate_document(name: str) -> bool:
    with connect() as db:
        return db.execute(
            "UPDATE documents SET active = 0 WHERE name = ? AND active = 1", (name,)
        ).rowcount > 0


def active_chunks() -> list[dict]:
    with connect() as db:
        rows = db.execute(
            f"SELECT {CHUNK_COLUMNS} FROM chunks c "
            "JOIN documents d ON d.id = c.document_id WHERE d.active = 1"
        ).fetchall()
    return [dict(row) for row in rows]


def keyword_search(words: list[str], limit: int) -> list[int]:
    if not words:
        return []
    query = " OR ".join(f'"{word}"' for word in words)
    with connect() as db:
        rows = db.execute(
            """
            SELECT f.chunk_id FROM chunks_fts f
            JOIN chunks c ON c.id = f.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ? AND d.active = 1
            ORDER BY bm25(chunks_fts) LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    return [row["chunk_id"] for row in rows]
