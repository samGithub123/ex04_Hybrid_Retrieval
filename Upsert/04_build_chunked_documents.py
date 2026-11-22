# -*- coding: utf-8 -*-
"""
04_build_chunked_documents.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Step 4: the loader/chunker
orchestrator. Loops over every raw document (02_load_source_documents.py),
chunks each one (03_semantic_chunking.py), and returns the final flat list
of chunk-level dicts that Step 5 (BM25) and Step 6 (dense embeddings) both
consume.

What it does
LOOP over every raw document loaded from ../data/.
  Split its text into semantic chunks.
  LOOP over every chunk produced for that document.
    Stamp it with a stable id ("<filename>_chunk_<i>") and carry the
    parent document's source path along for citation purposes.
Returns one flat list, e.g.:
    {"id": "home_loan.txt_chunk_0", "text": "...", "source": ".../home_loan.txt"}
"""

from importlib import import_module

from sentence_transformers import SentenceTransformer

_loader = import_module("02_load_source_documents")
_chunker = import_module("03_semantic_chunking")


def build_chunked_documents(chunk_model: SentenceTransformer = None) -> list[dict]:
    """Reads every raw document and splits it into chunk-level dicts:
        {"id": "<filename>_chunk_<i>", "text": "<chunk text>", "source": "<file path>"}
    """
    model = chunk_model or SentenceTransformer(_chunker.CHUNK_MODEL_NAME)

    raw_docs = _loader.load_raw_documents()
    print(f"[Upsert/04] Chunking {len(raw_docs)} raw document(s) ...")

    docs = []
    for raw_doc in raw_docs:
        chunks = _chunker.semantic_chunk_text_embedding(raw_doc["text"], model=model)
        print(f"[Upsert/04]   '{raw_doc['id']}' -> {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            docs.append({
                "id": f"{raw_doc['id']}_chunk_{i}",
                "text": chunk,
                "source": raw_doc["source"],
            })

    return docs


if __name__ == "__main__":
    docs = build_chunked_documents()

    print(f"\n[Upsert/04] Built {len(docs)} chunk document(s) in total.")
    print("[Upsert/04] First chunk document:")
    print(docs[0])
