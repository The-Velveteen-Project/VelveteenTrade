"""Retrieval facade for the pipeline: graceful when the corpus isn't built yet."""
from __future__ import annotations

import logging
from pathlib import Path

from .store import Hit, RagStore

log = logging.getLogger(__name__)


class Retriever:
    """Returns canon excerpts for a query, or [] if the RAG db doesn't exist.

    The pipeline must work with or without the corpus — a missing rag.db
    degrades to un-grounded (but still schema-valid) analysis, never to a crash.
    """

    def __init__(self, db_path: Path) -> None:
        self._store: RagStore | None = None
        if db_path.exists():
            try:
                store = RagStore(db_path)
                if store.count() > 0:
                    self._store = store
                else:
                    store.close()
            except Exception as exc:
                log.warning("RAG store unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._store is not None

    def excerpts(self, query: str, top_k: int = 3, category: str | None = None) -> list[str]:
        """Formatted excerpts ready to inject into an agent prompt."""
        hits = self.search(query, top_k=top_k, category=category)
        return [f"[{h.doc_id}#{h.seq}] ({h.title}): {h.text[:900]}" for h in hits]

    def search(self, query: str, top_k: int = 4, category: str | None = None) -> list[Hit]:
        if self._store is None:
            return []
        try:
            return self._store.search(self._embed_cached(query), top_k=top_k, category=category)
        except Exception as exc:
            log.warning("RAG search failed (continuing without canon): %s", exc)
            return []

    def _embed_cached(self, query: str):
        if not hasattr(self, "_qcache"):
            self._qcache: dict[str, object] = {}
        if query not in self._qcache:
            from .ingest import embed_query

            self._qcache[query] = embed_query(query)
        return self._qcache[query]
