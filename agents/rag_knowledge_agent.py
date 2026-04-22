"""
RAG Knowledge Agent
--------------------
Answers farming questions by retrieving relevant passages from ChromaDB
and synthesising them with an LLM (or a simple extractive fallback).
"""
import os
from typing import Any

from rag.vector_store import VectorStore
from rag.knowledge_base import iter_chunks

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

RAG_SYSTEM_PROMPT = """You are an expert agricultural knowledge assistant.
Answer the farmer's question using ONLY the context passages provided.
Be concise, practical, and farmer-friendly. If the context does not contain
enough information, say so honestly."""


class RAGKnowledgeAgent:
    """Retrieval-Augmented Generation agent backed by ChromaDB."""

    def __init__(self):
        self._store = VectorStore()
        self._ensure_populated()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def query(self, question: str, n_results: int = 3) -> dict[str, Any]:
        """
        Answer *question* using RAG.

        Returns
        -------
        {
            "question":   str,
            "answer":     str,
            "sources":    [{"topic": str, "snippet": str, "score": float}],
            "context":    str,   # raw context passed to LLM
        }
        """
        hits     = self._store.query(question, n_results=n_results)
        context  = self._build_context(hits)
        answer   = self._generate_answer(question, context)
        sources  = [
            {
                "topic":   h["metadata"].get("topic", "General"),
                "snippet": h["text"][:200].replace("\n", " "),
                "score":   round(1 - h["distance"], 4),  # cosine similarity
            }
            for h in hits
        ]
        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
            "context":  context,
        }

    def retrieve_for_crop(self, crop_name: str) -> str:
        """
        Convenience: return a brief context paragraph about *crop_name*.
        Used by the Decision Agent to enrich LLM prompts.
        """
        hits = self._store.query(f"cultivation practices for {crop_name}", n_results=2)
        return self._build_context(hits)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _ensure_populated(self) -> None:
        """Ingest knowledge base on first run (idempotent via upsert)."""
        if not self._store.is_populated():
            print("RAG: Ingesting agricultural knowledge base into ChromaDB...")
            docs = list(iter_chunks())
            self._store.add_documents(docs)
            print(f"RAG: Stored {self._store.count()} document chunks.")

    @staticmethod
    def _build_context(hits: list[dict]) -> str:
        return "\n\n---\n\n".join(h["text"] for h in hits)

    def _generate_answer(self, question: str, context: str) -> str:
        if GROQ_API_KEY:
            try:
                from groq import Groq
                client = Groq(api_key=GROQ_API_KEY)
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system",  "content": RAG_SYSTEM_PROMPT},
                        {"role": "user",    "content": f"Context:\n{context}\n\nQuestion: {question}"},
                    ],
                    temperature=0.2,
                    max_tokens=400,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                pass

        # Extractive fallback: return the most relevant passage
        return (
            context.split("\n\n---\n\n")[0][:600]
            + "\n\n(Note: LLM not configured – showing raw retrieval result.)"
        )
