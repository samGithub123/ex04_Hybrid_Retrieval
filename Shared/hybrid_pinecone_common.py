# -*- coding: utf-8 -*-
"""
hybrid_pinecone_common.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - shared Pinecone / embedding-model
/ BM25 / LLM setup used by both the Upsert/ and Retrieval/ pipelines. Keeping
this in one place guarantees the index name, embedding model, vector
dimension, similarity metric, BM25 params file path and LLM stay in sync
between the scripts that build+upsert the hybrid vectors and the scripts
that later query, rerank and answer from them.

This module lives in Shared/, one level below the project root where
keys.py, .env, data/ and json/ live - see _PROJECT_ROOT below.

Requires: pip install pinecone pinecone-text sentence-transformers langchain-openai
"""

import os
import sys
import time

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

# keys.py lives in the project root (one level up from Shared/), not next to
# this script, so it must be added to sys.path before it can be imported.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import keys

# --- Project-wide constants -------------------------------------------------
# Every script under Upsert/ and Retrieval/ imports these instead of hardcoding
# them, so the two pipelines can never drift apart (e.g. one side upserting
# with a 384-dim model while the other queries with a different one).
INDEX_NAME = "vector-loan-hybrid"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Must match paraphrase-MiniLM-L6-v2's output size

# Hybrid records carry BOTH a dense "values" vector and a BM25 "sparse_values"
# vector on every record. Pinecone requires "dotproduct" for that combination
# - an existing "cosine" index cannot be converted in place, so this project
# always creates its own fresh dotproduct index rather than reusing a
# cosine one from another exercise.
HYBRID_METRIC = "dotproduct"

# Drives query editing and final answer generation in the Retrieval pipeline
# (see Retrieval/02_query_editing.py and Retrieval/06_prompt_rendering_and_full_chain.py).
LLM_MODEL_NAME = "gpt-4o-mini"

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
BM25_PARAMS_DIR = os.path.join(_PROJECT_ROOT, "json")
BM25_PARAMS_PATH = os.path.join(BM25_PARAMS_DIR, "bm25_params.json")


def get_pinecone_client() -> Pinecone:
    """Connects to Pinecone using the API key from keys.py/.env."""
    return Pinecone(api_key=keys.require("PINECONE_API_KEY", keys.PINECONE_API_KEY))


def ensure_index_exists(pc: Pinecone, index_name: str = INDEX_NAME) -> None:
    """Creates `index_name` as a serverless (AWS us-east-1) dotproduct index
    if it doesn't exist yet. Safe to call repeatedly - it only creates the
    index once and simply reuses it on every later run."""
    existing = [idx.name for idx in pc.list_indexes()]
    print(f"[Shared] Existing Pinecone indexes: {existing}")

    if index_name in existing:
        print(f"[Shared] Index '{index_name}' already exists - reusing it.")
        return

    print(
        f"[Shared] Creating hybrid index '{index_name}' "
        f"(dim={EMBEDDING_DIMENSION}, metric={HYBRID_METRIC}) ..."
    )
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric=HYBRID_METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Wait until the serverless index is ready before anything tries to use it.
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print(f"[Shared] Index '{index_name}' created and ready.")


def get_index(pc: Pinecone = None, index_name: str = INDEX_NAME, ensure_exists: bool = False):
    """Connects to the Pinecone index and returns a ready-to-use Index handle.

    Pass ensure_exists=True (Upsert side) to create the index first if it's
    missing. Otherwise (Retrieval side) Pinecone's own error is raised if the
    index hasn't been created/upserted into yet.
    """
    pc = pc or get_pinecone_client()
    if ensure_exists:
        ensure_index_exists(pc, index_name)
    return pc.Index(index_name)


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """The same dense sentence-embedding model used to build the vectors that
    were upserted - queries must be embedded with this same model so they
    land in the same vector space as the stored documents."""
    print(f"[Shared] Loading dense embedding model '{model_name}' ...")
    return SentenceTransformer(model_name)


def get_bm25_encoder() -> BM25Encoder:
    """Returns a fresh, UNFITTED BM25Encoder. The Upsert pipeline fits this on
    the chunk corpus (see Upsert/05_bm25_sparse_encoding.py); the Retrieval
    pipeline must NOT fit a new one - it should call load_bm25_encoder()
    below so it reuses the exact vocabulary/statistics used at upsert time."""
    return BM25Encoder()


def load_bm25_encoder(params_path: str = BM25_PARAMS_PATH) -> BM25Encoder:
    """Loads the BM25 encoder that Upsert/05_bm25_sparse_encoding.py fitted on
    the document corpus and saved to disk. Reusing this exact encoder (instead
    of fitting a new one) is mandatory - a freshly-fit encoder would assign
    different term indices, and the resulting sparse query vector would not
    line up with the sparse vectors already stored in Pinecone.
    """
    if not os.path.isfile(params_path):
        raise RuntimeError(
            f"BM25 params file not found at '{params_path}'. Run the Upsert "
            "pipeline first (Upsert/05_bm25_sparse_encoding.py) so it can be "
            "fitted and saved before the Retrieval pipeline tries to load it."
        )
    bm25 = BM25Encoder()
    bm25.load(params_path)
    print(f"[Shared] Loaded fitted BM25 encoder from '{params_path}'.")
    return bm25


def get_llm(model_name: str = LLM_MODEL_NAME) -> ChatOpenAI:
    """Chat LLM reused for query editing and final answer generation.
    temperature=0 keeps query rewrites and answers deterministic/grounded."""
    keys.require("OPENAI_API_KEY", keys.OPENAI_API_KEY)
    print(f"[Shared] Loading LLM '{model_name}' ...")
    return ChatOpenAI(model=model_name, temperature=0)


if __name__ == "__main__":
    print("[Shared] Sanity check of hybrid_pinecone_common.py")
    print(f"[Shared] INDEX_NAME          = {INDEX_NAME}")
    print(f"[Shared] EMBEDDING_MODEL     = {EMBEDDING_MODEL_NAME} (dim={EMBEDDING_DIMENSION})")
    print(f"[Shared] HYBRID_METRIC       = {HYBRID_METRIC}")
    print(f"[Shared] LLM_MODEL_NAME      = {LLM_MODEL_NAME}")
    print(f"[Shared] DATA_DIR            = {DATA_DIR}")
    print(f"[Shared] BM25_PARAMS_PATH    = {BM25_PARAMS_PATH}")
