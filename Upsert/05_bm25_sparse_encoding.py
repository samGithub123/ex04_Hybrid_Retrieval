# -*- coding: utf-8 -*-
"""
05_bm25_sparse_encoding.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Step 5: build the BM25 sparse
representation for the same chunks produced by 04_build_chunked_documents.py.

Each chunk will later be stored (Step 7) as one Pinecone record containing:
    - a dense vector in "values"           (Step 6)
    - a BM25 sparse vector in "sparse_values"  (this step)
    - text and source in "metadata"

BM25 must be FITTED once on the complete chunk corpus, then saved to disk as
JSON (not pickle - pickle is not portable across machines/versions) so the
Retrieval pipeline can load the exact same fitted encoder later. Re-fitting a
new encoder on the query side would assign different term indices and break
retrieval silently.

Requires: pip install pinecone-text

What it does
Fits a BM25Encoder on every chunk's text (the full corpus).
Encodes every chunk into a compressed {indices, values} sparse vector.
Saves the fitted encoder's statistics to ../json/bm25_params.json for reuse
   by the Retrieval pipeline.
"""

import os
import sys
from importlib import import_module

from pinecone_text.sparse import BM25Encoder

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED_DIR = os.path.join(_PROJECT_ROOT, "Shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

from hybrid_pinecone_common import BM25_PARAMS_DIR, BM25_PARAMS_PATH

_doc_builder = import_module("04_build_chunked_documents")


def fit_bm25_on_corpus(docs: list[dict]) -> tuple[BM25Encoder, list[dict]]:
    """Fits a BM25Encoder on every chunk's text and returns
    (fitted encoder, sparse_embeddings) where sparse_embeddings[i] is the
    {"indices": [...], "values": [...]} sparse vector for docs[i].
    """
    corpus = [d["text"] for d in docs]
    print(f"[Upsert/05] Fitting BM25 on a corpus of {len(corpus)} chunk(s) ...")

    bm25 = BM25Encoder()
    bm25.fit(corpus)

    sparse_embeddings = bm25.encode_documents(corpus)
    print(f"[Upsert/05] Encoded {len(sparse_embeddings)} sparse document vector(s).")
    print(f"[Upsert/05] First sparse vector: {sparse_embeddings[0]}")

    return bm25, sparse_embeddings


def save_bm25_params(bm25: BM25Encoder, params_path: str = BM25_PARAMS_PATH) -> str:
    """Persists the fitted BM25 corpus statistics to disk as JSON so the
    Retrieval pipeline can load this exact encoder later (see
    Shared/hybrid_pinecone_common.py::load_bm25_encoder)."""
    os.makedirs(BM25_PARAMS_DIR, exist_ok=True)
    bm25.dump(params_path)
    print(f"[Upsert/05] Saved fitted BM25 params to '{params_path}'.")
    return params_path


if __name__ == "__main__":
    docs = _doc_builder.build_chunked_documents()
    bm25, sparse_embeddings = fit_bm25_on_corpus(docs)
    save_bm25_params(bm25)
