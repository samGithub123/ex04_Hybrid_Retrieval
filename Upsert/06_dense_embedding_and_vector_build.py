# -*- coding: utf-8 -*-
"""
06_dense_embedding_and_vector_build.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Step 6: embed every chunk from
04_build_chunked_documents.py with the dense sentence-embedding model, then
combine each chunk's dense vector with its BM25 sparse vector
(05_bm25_sparse_encoding.py) into the exact record shape Pinecone expects:

    {
        "id": "file.txt_chunk_0",
        "values": [0.12, -0.08, 0.31, ...],          # dense embedding
        "sparse_values": {                            # BM25 output
            "indices": [104, 919, 4501],
            "values": [0.42, 0.77, 0.31]
        },
        "metadata": {
            "text": "One document chunk",
            "source": "file.txt"
        }
    }

Requires: pip install sentence-transformers

What it does
Loads the same embedding model used for chunking similarity decisions.
Encodes ALL chunk texts into dense vectors in one batch call, normalized
   so dot-product search behaves like cosine similarity.
Zips each chunk's {id, text, source}, dense vector, and BM25 sparse vector
   together into one Pinecone-ready vector record.
"""

import os
import sys
from importlib import import_module

from sentence_transformers import SentenceTransformer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED_DIR = os.path.join(_PROJECT_ROOT, "Shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

from hybrid_pinecone_common import EMBEDDING_MODEL_NAME

_doc_builder = import_module("04_build_chunked_documents")
_bm25_step = import_module("05_bm25_sparse_encoding")


def embed_chunks(docs: list[dict], model: SentenceTransformer):
    """Encodes every chunk's text into a normalized dense vector, in one
    batch call. Normalizing lets a dotproduct index (required for hybrid
    search) behave like cosine similarity on the dense side."""
    texts = [d["text"] for d in docs]
    print(f"[Upsert/06] Dense-embedding {len(texts)} chunk(s) with '{model!r}' ...")

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"[Upsert/06] Embedding shape: {embeddings.shape}")
    print(f"[Upsert/06] First embedding's first 5 dims: {embeddings[0][:5]}")

    return embeddings


def build_hybrid_vectors(docs: list[dict], dense_embeddings, sparse_embeddings) -> list[dict]:
    """Zips {id, text, source} + dense vector + BM25 sparse vector into the
    {id, values, sparse_values, metadata} shape index.upsert() expects."""
    vectors = []
    for d, dense_emb, sparse_emb in zip(docs, dense_embeddings, sparse_embeddings):
        vectors.append({
            "id": d["id"],
            "values": dense_emb.tolist(),
            "sparse_values": {
                "indices": [int(i) for i in sparse_emb["indices"]],
                "values": [float(v) for v in sparse_emb["values"]],
            },
            "metadata": {
                "text": d["text"],
                "source": d["source"],
            },
        })

    print(f"[Upsert/06] Built {len(vectors)} hybrid vector record(s).")
    print(f"[Upsert/06] First vector record: {vectors[0]}")
    return vectors


if __name__ == "__main__":
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    docs = _doc_builder.build_chunked_documents(chunk_model=model)
    bm25, sparse_embeddings = _bm25_step.fit_bm25_on_corpus(docs)

    dense_embeddings = embed_chunks(docs, model)
    vectors = build_hybrid_vectors(docs, dense_embeddings, sparse_embeddings)
