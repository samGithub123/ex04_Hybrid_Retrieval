# -*- coding: utf-8 -*-
"""
01_pinecone_index_setup.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Step 1: connect to Pinecone and
create the hybrid (dotproduct) index this whole folder's pipeline upserts
into, if it doesn't already exist.

The actual Pinecone/embedding-model/BM25 setup is shared with the Retrieval/
pipeline - see ../Shared/hybrid_pinecone_common.py.

Requires: pip install pinecone

What it does
Connects to the Pinecone vector database using the API key from keys.py/.env.
Checks whether the 'vector-loan-hybrid' index already exists.
If it doesn't exist, creates it as a 384-dim, dotproduct, AWS-serverless
   index (dotproduct is required because hybrid records carry both dense
   "values" and BM25 "sparse_values").
Ensures later scripts can store (upsert) and retrieve vectors from the
   same index.
"""

import os
import sys

from pinecone import Pinecone

# The shared setup module lives in Shared/, a sibling of this Upsert/ folder,
# not next to this script, so it must be added to sys.path first.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED_DIR = os.path.join(_PROJECT_ROOT, "Shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

from hybrid_pinecone_common import (
    INDEX_NAME,
    EMBEDDING_DIMENSION,   # Must match the paraphrase-MiniLM-L6-v2 embedding model used throughout this folder
    HYBRID_METRIC,
    get_pinecone_client,
    ensure_index_exists,
)
from hybrid_pinecone_common import get_index as _get_index


def get_index(pc: Pinecone = None, index_name: str = INDEX_NAME):
    """Convenience helper: ensures the hybrid index exists and returns a
    ready-to-use Pinecone Index handle, so downstream scripts can do just:
        index = _pinecone_setup.get_index()
    """
    return _get_index(pc, index_name, ensure_exists=True)


if __name__ == "__main__":
    print("[Upsert/01] Connecting to Pinecone ...")
    pc = get_pinecone_client()

    print(f"[Upsert/01] Ensuring index '{INDEX_NAME}' exists "
          f"(dim={EMBEDDING_DIMENSION}, metric={HYBRID_METRIC}) ...")
    ensure_index_exists(pc)

    index = get_index(pc)
    print(f"[Upsert/01] Ready to use index handle: {index}")
    print(f"[Upsert/01] Index stats: {index.describe_index_stats()}")
