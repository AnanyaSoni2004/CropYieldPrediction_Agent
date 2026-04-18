"""
Vector Store (ChromaDB)
------------------------
Manages the ChromaDB collection that backs the RAG Knowledge Agent.
Uses sentence-transformers for local embeddings (no API key needed).
"""
import os
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH  = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION   = "agroagent_knowledge"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # fast, small, runs fully offline


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self):
        self._ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self._client = chromadb.PersistentClient(path=CHROMA_PATH)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_populated(self) -> bool:
        return self._col.count() > 0

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """
        Upsert a list of {'id', 'text', 'metadata'} dicts into the collection.
        """
        self._col.upsert(
            ids       =[d["id"]       for d in docs],
            documents =[d["text"]     for d in docs],
            metadatas =[d["metadata"] for d in docs],
        )

    def query(self, text: str, n_results: int = 3) -> list[dict[str, Any]]:
        """
        Semantic search.  Returns a list of result dicts with keys:
        'text', 'metadata', 'distance'.
        """
        results = self._col.query(
            query_texts=[text],
            n_results=min(n_results, self._col.count() or 1),
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"text": doc, "metadata": meta, "distance": dist})
        return output

    def count(self) -> int:
        return self._col.count()
