"""Vector store: SQLite + numpy. Zero extra infrastructure.

The corpus is small (~a dozen documents, a few thousand chunks), so brute-force
cosine search over in-memory embeddings is faster than any vector database's
network round-trip — and there is nothing new to deploy or pay for.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id);
"""


@dataclass
class Hit:
    doc_id: str
    title: str
    category: str
    seq: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.title} [{self.doc_id}#{self.seq}]"


class RagStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._matrix: np.ndarray | None = None
        self._rows: list[tuple] | None = None

    def add(self, doc_id: str, title: str, category: str, seq: int,
            text: str, embedding: np.ndarray) -> None:
        self._conn.execute(
            "INSERT INTO chunks (doc_id, title, category, seq, text, embedding) VALUES (?,?,?,?,?,?)",
            (doc_id, title, category, seq, text,
             np.asarray(embedding, dtype=np.float32).tobytes()),
        )
        self._matrix = None  # invalidate cache

    def delete_doc(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        self._matrix = None

    def commit(self) -> None:
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def doc_ids(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT DISTINCT doc_id FROM chunks")}

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT doc_id, title, category, seq, text, embedding FROM chunks"
        ).fetchall()
        self._rows = rows
        if rows:
            mat = np.vstack([np.frombuffer(r[5], dtype=np.float32) for r in rows])
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            self._matrix = mat / np.clip(norms, 1e-9, None)
        else:
            self._matrix = np.zeros((0, 1), dtype=np.float32)

    def search(self, query_embedding: np.ndarray, top_k: int = 4,
               category: str | None = None) -> list[Hit]:
        if self._matrix is None:
            self._load()
        assert self._rows is not None and self._matrix is not None
        if not self._rows:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        scores = self._matrix @ q
        order = np.argsort(-scores)
        hits: list[Hit] = []
        for i in order:
            doc_id, title, cat, seq, text, _ = self._rows[int(i)]
            if category and cat != category:
                continue
            hits.append(Hit(doc_id, title, cat, seq, text, float(scores[int(i)])))
            if len(hits) >= top_k:
                break
        return hits

    def close(self) -> None:
        self._conn.close()
