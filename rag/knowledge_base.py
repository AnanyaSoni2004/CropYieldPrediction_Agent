"""
Knowledge Base Builder
-----------------------
Parses agricultural_knowledge.txt into labelled chunks for ingestion
into the ChromaDB vector store.
"""
import os
import re
from typing import Iterator

KNOWLEDGE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "agricultural_knowledge.txt"
)


def iter_chunks(
    chunk_size: int = 400,
    overlap: int = 50,
) -> Iterator[dict]:
    """
    Yield {'id': str, 'text': str, 'metadata': dict} dicts.

    The file is split on section headers (=== ... ===).  Sections longer
    than *chunk_size* words are further split with a rolling *overlap*-word
    window to preserve context across chunk boundaries.
    """
    with open(KNOWLEDGE_FILE, encoding="utf-8") as fh:
        raw = fh.read()

    sections = re.split(r"={3,}.*?={3,}", raw)
    headers  = re.findall(r"={3,}(.*?)={3,}", raw)

    doc_id = 0
    for header, body in zip(headers, sections[1:]):
        topic = header.strip().title()
        words = body.split()

        if len(words) <= chunk_size:
            yield {
                "id":       f"doc_{doc_id}",
                "text":     f"{topic}\n{body.strip()}",
                "metadata": {"topic": topic, "chunk": 0},
            }
            doc_id += 1
        else:
            # sliding-window chunking for long sections
            start  = 0
            chunk_idx = 0
            while start < len(words):
                end   = start + chunk_size
                chunk = " ".join(words[start:end])
                yield {
                    "id":       f"doc_{doc_id}",
                    "text":     f"{topic}\n{chunk}",
                    "metadata": {"topic": topic, "chunk": chunk_idx},
                }
                doc_id    += 1
                chunk_idx += 1
                start      = end - overlap
