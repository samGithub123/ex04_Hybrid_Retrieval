# -*- coding: utf-8 -*-
"""
01_pinecone_and_model_setup.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Retrieval Step 1: connect to the
already-populated Pinecone index, load the same dense embedding model used
at upsert time, load the SAME fitted BM25 encoder that
Upsert/05_bm25_sparse_encoding.py saved to ../json/bm25_params.json, and get
an LLM handle for query editing (Step 2) and final answer generation
(Step 6).

Creates nothing - it assumes Upsert/07_upsert_to_pinecone.py has already run.

The actual Pinecone/embedding-model/BM25/LLM setup is shared with the
Upsert/ pipeline - see ../Shared/hybrid_pinecone_common.py.

Requires: pip install pinecone pinecone-text sentence-transformers langchain-openai

What it does
Connects to Pinecone using the API key from keys.py/.env.
Returns a handle to the existing 'vector-loan-hybrid' index (raises
   Pinecone's own error if it hasn't been created/upserted into yet).
Loads 'paraphrase-MiniLM-L6-v2' - the SAME model used to embed the stored
   chunks, so queries land in the same vector space as the documents.
Loads the fitted BM25 encoder from disk - reusing it (never re-fitting) is
   what keeps sparse query vectors compatible with the sparse vectors
   already stored in Pinecone.
Loads the 'gpt-4o-mini' chat LLM used by Step 2 (query editing) and Step 6
   (final answer generation).
"""

import os
import sys

from pinecone import Pinecone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED_DIR = os.path.join(_PROJECT_ROOT, "Shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

from hybrid_pinecone_common import (
    BM25_PARAMS_PATH,
    EMBEDDING_MODEL_NAME,   # must match the model used to upsert the vectors
    INDEX_NAME,             # must match ../Upsert/01_pinecone_index_setup.py
    LLM_MODEL_NAME,
    get_embedding_model,
    get_llm,
    get_pinecone_client,
    load_bm25_encoder,
)
from hybrid_pinecone_common import get_index as _get_index


def get_index(pc: Pinecone = None, index_name: str = INDEX_NAME):
    """Connects to the (already-existing) hybrid Pinecone index. Raises
    Pinecone's own error if the index hasn't been created/upserted into yet."""
    return _get_index(pc, index_name, ensure_exists=False)


def connect_everything():
    """Convenience helper bundling all of Retrieval Step 1 into one call:
    returns (pc, index, embedding_model, bm25_encoder, llm)."""
    pc = get_pinecone_client()
    index = get_index(pc)
    embedding_model = get_embedding_model()
    bm25 = load_bm25_encoder()
    llm = get_llm()
    return pc, index, embedding_model, bm25, llm


def _demo_reload_bm25_from_scratch() -> None:
    """Sanity check that loading BM25 params from disk into a BRAND NEW,
    never-fitted encoder reproduces usable sparse vectors - exactly what
    Retrieval/03_hybrid_retrieval.py relies on for every query."""
    print("\n[Retrieval/01] Sanity check: load BM25 from JSON into a fresh encoder")
    fresh_bm25 = load_bm25_encoder(BM25_PARAMS_PATH)

    doc_sparse_vector = fresh_bm25.encode_documents("The brown fox is quick")
    query_sparse_vector = fresh_bm25.encode_queries("Which fox is brown?")

    print(f"[Retrieval/01] Sample doc sparse vector:   {doc_sparse_vector}")
    print(f"[Retrieval/01] Sample query sparse vector: {query_sparse_vector}")


if __name__ == "__main__":
    print("[Retrieval/01] Connecting to Pinecone ...")
    pc = get_pinecone_client()

    index = get_index(pc)
    print(f"[Retrieval/01] Connected to index: {index}")
    print(f"[Retrieval/01] Index stats: {index.describe_index_stats()}")

    model = get_embedding_model()
    print(f"[Retrieval/01] Loaded embedding model: {EMBEDDING_MODEL_NAME}")

    bm25 = load_bm25_encoder()
    print(f"[Retrieval/01] Loaded BM25 encoder from '{BM25_PARAMS_PATH}'")

    llm = get_llm()
    print(f"[Retrieval/01] Loaded LLM: {LLM_MODEL_NAME}")
    print(f"[Retrieval/01] LLM sanity check: {llm.invoke('Say hello in one short sentence.').content}")

    _demo_reload_bm25_from_scratch()
