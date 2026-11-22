# -*- coding: utf-8 -*-
"""
02_load_source_documents.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Step 2: load the raw source
documents that will later be split into chunks (03_semantic_chunking.py /
04_build_chunked_documents.py), BM25-encoded (05_bm25_sparse_encoding.py),
dense-embedded and upserted (06/07).

The corpus here is a flat folder of plain-text loan product/FAQ documents
(../data/*.txt) - no metadata catalog is needed, unlike the richer
appliance-manual project this folder's conventions were borrowed from.

What it does
Lists every file under ../data/.
Reads each file's full text.
Packages {id, text, source} per document, ready for chunking.
"""

import os

# data/ lives in the project root (one level up from this script's Upsert/
# folder), not next to this file. Computed locally (instead of importing it
# from Shared/hybrid_pinecone_common.py) so this script has no import-order
# dependency on another script having already put Shared/ on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# Organize your docs like this - each raw document is just {id, text, source}.
# Sample only, not for sale!
SAMPLE_DOCS = [
    {"id": "doc1", "text": "Pandas is a Python library for data analysis.", "source": "sample"},
    {"id": "doc2", "text": "Pinecone is a vector database for semantic search.", "source": "sample"},
    {"id": "doc3", "text": "Spark enables distributed data processing.", "source": "sample"},
]


def load_raw_documents(data_dir: str = _DATA_DIR) -> list[dict]:
    """Reads every file in `data_dir` and returns a list of raw (not yet
    chunked) documents: {"id": filename, "text": file contents, "source": path}.
    Falls back to SAMPLE_DOCS if the data folder is missing/empty so the rest
    of the pipeline still has something to work with.
    """
    if not os.path.isdir(data_dir):
        print(f"[Upsert/02] '{data_dir}' not found - using SAMPLE_DOCS instead.")
        return SAMPLE_DOCS

    files = sorted(os.listdir(data_dir))
    print(f"[Upsert/02] Found {len(files)} file(s) in '{data_dir}': {files}")

    if not files:
        print(f"[Upsert/02] '{data_dir}' is empty - using SAMPLE_DOCS instead.")
        return SAMPLE_DOCS

    docs = []
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"id": filename, "text": text, "source": file_path})

    return docs


if __name__ == "__main__":
    docs = load_raw_documents()
    print(f"[Upsert/02] Loaded {len(docs)} raw document(s) from '{_DATA_DIR}'.")

    # Peek at the first document, a quick sanity check before wiring up chunking.
    first_doc = docs[0]
    print(f"\n[Upsert/02] First document id: {first_doc['id']}")
    print(f"[Upsert/02] First document length: {len(first_doc['text'])} chars")
    print(f"[Upsert/02] First document preview:\n{first_doc['text'][:400]}")
