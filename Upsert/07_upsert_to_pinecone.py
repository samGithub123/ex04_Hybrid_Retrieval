# -*- coding: utf-8 -*-
"""
07_upsert_to_pinecone.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Step 7 (final): the ingestion
orchestrator. Wires together every earlier Upsert/ step in order and pushes
the resulting hybrid vectors into Pinecone in batches.

    01_pinecone_index_setup.py        -> ensure the destination index exists
    02_load_source_documents.py       -> read ../data/*.txt
    03_semantic_chunking.py           -> chunking functions
    04_build_chunked_documents.py     -> loop: chunk every raw document
    05_bm25_sparse_encoding.py        -> fit BM25, save params for reuse
    06_dense_embedding_and_vector_build.py -> dense-embed + assemble records
    07_upsert_to_pinecone.py (this file)   -> batch-upsert to Pinecone

Requires: pip install pinecone pinecone-text sentence-transformers

What it does
Runs the full ingestion pipeline end-to-end (see main() below).
Batches the vector list (Pinecone accepts ~100 vectors / ~2MB per request)
   and upserts one batch at a time, printing progress after each batch.
Prints the resulting index stats so you can see the vector count that
   the Retrieval pipeline will be searching against.
"""

import os
import sys
from importlib import import_module

from sentence_transformers import SentenceTransformer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED_DIR = os.path.join(_PROJECT_ROOT, "Shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

from hybrid_pinecone_common import EMBEDDING_MODEL_NAME, INDEX_NAME

_index_setup = import_module("01_pinecone_index_setup")
_doc_builder = import_module("04_build_chunked_documents")
_bm25_step = import_module("05_bm25_sparse_encoding")
_vector_builder = import_module("06_dense_embedding_and_vector_build")

BATCH_SIZE = 100


def upsert_vectors(index, vectors: list[dict], batch_size: int = BATCH_SIZE) -> None:
    """Slices `vectors` into batches and upserts each one, printing progress
    so a long ingestion run stays observable."""
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start:start + batch_size]
        index.upsert(vectors=batch)
        print(f"[Upsert/07] Upserted batch: {len(batch)} vector(s) "
              f"({min(start + batch_size, len(vectors))}/{len(vectors)}).")


def main() -> None:
    print("=" * 70)
    print("[Upsert/07] STEP 1 - Ensure Pinecone index exists")
    print("=" * 70)
    index = _index_setup.get_index()
    print(f"[Upsert/07] Using index handle: {index}")

    print("\n" + "=" * 70)
    print("[Upsert/07] STEP 2-4 - Load, chunk, and stamp every document")
    print("=" * 70)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    docs = _doc_builder.build_chunked_documents(chunk_model=model)
    print(f"[Upsert/07] Total chunk documents built: {len(docs)}")

    print("\n" + "=" * 70)
    print("[Upsert/07] STEP 5 - Fit BM25 sparse encoder + save params")
    print("=" * 70)
    bm25, sparse_embeddings = _bm25_step.fit_bm25_on_corpus(docs)
    _bm25_step.save_bm25_params(bm25)

    print("\n" + "=" * 70)
    print("[Upsert/07] STEP 6 - Dense-embed chunks + build hybrid vectors")
    print("=" * 70)
    dense_embeddings = _vector_builder.embed_chunks(docs, model)
    vectors = _vector_builder.build_hybrid_vectors(docs, dense_embeddings, sparse_embeddings)

    print("\n" + "=" * 70)
    print(f"[Upsert/07] STEP 7 - Upsert {len(vectors)} hybrid vector(s) into '{INDEX_NAME}'")
    print("=" * 70)
    upsert_vectors(index, vectors)

    print(f"\n[Upsert/07] Done. Upserted {len(vectors)} hybrid record(s).")
    print(f"[Upsert/07] Index stats: {index.describe_index_stats()}")


if __name__ == "__main__":
    main()
