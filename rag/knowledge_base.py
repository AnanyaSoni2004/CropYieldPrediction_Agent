"""
Knowledge Base Builder
-----------------------
Reads agronomic knowledge documents from rag/documents/ and produces
labelled chunks for ingestion into the ChromaDB vector store.

Each .txt file in documents/ is a crop-specific guide covering ideal growing
conditions, soil requirements, pests, diseases, and best practices.
"""
import os
from typing import Iterator

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")


def iter_chunks() -> Iterator[dict]:
    """
    Yield {'id': str, 'text': str, 'metadata': dict} dicts.

    Each document is split into section chunks (split on '## ') so that
    vector search retrieves focused, relevant passages rather than entire files.
    """
    doc_id = 0

    for filename in sorted(os.listdir(DOCUMENTS_DIR)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(DOCUMENTS_DIR, filename)
        crop_name = filename.replace(".txt", "").capitalize()

        with open(filepath, encoding="utf-8") as fh:
            content = fh.read()

        # Extract topic and crop from frontmatter lines
        topic = crop_name
        for line in content.splitlines()[:5]:
            if line.startswith("topic:"):
                topic = line.split(":", 1)[1].strip()
                break

        # Split into sections on '## ' headers; keep header with its content
        raw_sections = content.split("\n## ")
        sections = []
        for i, section in enumerate(raw_sections):
            # Re-attach the '## ' prefix for all but the first block
            text = section if i == 0 else "## " + section
            text = text.strip()
            if text:
                sections.append(text)

        for section in sections:
            yield {
                "id":       f"doc_{doc_id}",
                "text":     section,
                "metadata": {
                    "crop":  crop_name,
                    "topic": topic,
                    "chunk_type": "knowledge",
                    "source": filename,
                },
            }
            doc_id += 1
