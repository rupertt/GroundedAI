from __future__ import annotations

import logging
import sys
from typing import Any

from app.rag import DOC_SOURCE, index_if_needed, retrieve_with_scores

logger = logging.getLogger(__name__)


def apply_sqlite_fix_for_chroma() -> None:
    """
    CrewAI imports Chroma internally, which can fail on some WSL distros due to old sqlite3.

    This mirrors the defensive sqlite shim used in app/rag.py, but must run BEFORE importing crewai.
    """
    try:
        import sqlite3

        def _t(v: str) -> tuple[int, int, int]:
            parts = (v.split(".") + ["0", "0", "0"])[:3]
            return int(parts[0]), int(parts[1]), int(parts[2])

        if _t(sqlite3.sqlite_version) < (3, 35, 0):
            import pysqlite3

            sys.modules["sqlite3"] = pysqlite3
            logger.info("Applied pysqlite3 shim for sqlite3 compatibility.")
    except Exception:
        # If it fails, CrewAI/Chroma may still raise a clear RuntimeError; we don't hide it.
        logger.exception("Failed to apply sqlite3 compatibility shim.")


def make_retrieve_tool(store: dict[str, Any]):
    """
    Create a per-request CrewAI tool function with signature:
      retrieve_chunks(query: str, top_k: int) -> list[dict]

    The tool appends all retrieved chunks into `store` for later citation extraction/debug.
    """

    # Import crewai tooling lazily so app startup does not require CrewAI.
    from crewai.tools import tool

    @tool("retrieve_chunks")
    def retrieve_chunks(query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Retrieve relevant chunks from the persistent Chroma index.

        Returns a list of dicts with: {chunk_id, text, source, score}.
        """
        # Defensive: keep top_k within reasonable bounds.
        try:
            top_k_int = int(top_k)
        except Exception:
            top_k_int = 4
        top_k_int = max(1, min(20, top_k_int))

        index_if_needed()
        results = retrieve_with_scores(query, top_k=top_k_int)

        out: list[dict[str, Any]] = []
        for doc, score in results:
            chunk_id = str(doc.metadata.get("chunk_id", "chunk-??"))
            source = str(doc.metadata.get("source", DOC_SOURCE))
            item = {
                "chunk_id": chunk_id,
                "text": doc.page_content,
                "source": source,
                "score": float(score),
            }
            out.append(item)
            # Store latest version per (source, chunk_id) (last write wins).
            # This avoids collisions in multi-document mode where every file has chunk-00, chunk-01, ...
            store.setdefault("chunks", {})
            store["chunks"][f"{source}#{chunk_id}"] = item

        # Keep a linear history too (useful for debugging ordering).
        store.setdefault("calls", [])
        store["calls"].append({"query": query, "top_k": top_k_int, "results": out})

        # Optional debug logging controlled by the per-request store flag.
        if store.get("debug"):
            logger.info(
                "Agent tool retrieve_chunks: query=%r top_k=%s results=%s chunk_ids=%s",
                query,
                top_k_int,
                len(out),
                [x.get("chunk_id") for x in out[:5]],
            )

        return out

    return retrieve_chunks


