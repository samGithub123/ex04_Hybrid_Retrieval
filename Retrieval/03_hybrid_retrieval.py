# -*- coding: utf-8 -*-
"""
03_hybrid_retrieval.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Retrieval Step 3: "Retrieving".
Embeds the edited query (Step 2) into BOTH a dense vector and a BM25 sparse
vector, combines them at a chosen alpha weighting, and searches the hybrid
Pinecone index. Reshapes the raw Pinecone matches into plain candidate dicts
that Step 4 (reranking) and Step 5 (context building) consume.

    alpha = 0    -> pure sparse (BM25 keyword) search
    alpha = 0.5  -> equal weight - dense and sparse each scaled by 0.5 (default)
    alpha = 1    -> pure dense (semantic) search

Requires: pip install pinecone pinecone-text sentence-transformers

What it does
Embeds the edited query with the SAME dense model used to embed the stored
   chunks (Step 1).
Encodes the edited query with the SAME fitted BM25 encoder used to encode
   the stored chunks (Step 1) - never a freshly-fit one.
Scales both vectors via hybrid_convex_scale(alpha) and runs ONE
   index.query(vector=..., sparse_vector=...) call against Pinecone.
Reshapes Pinecone's ScoredVector matches into plain dicts carrying every
   field Steps 4-6 need: {id, text, source, score, rank}.
"""

from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer

DEFAULT_ALPHA = 0.5
DEFAULT_TOP_K = 8


def encode_dense_query(query: str, embedding_model: SentenceTransformer) -> list[float]:
    """Embeds the query with the SAME model used to embed the stored chunks."""
    dense_query = embedding_model.encode(query, normalize_embeddings=True).tolist()
    print(f"[Retrieval/03] Dense query vector: {len(dense_query)} dims, "
          f"first 5 = {dense_query[:5]}")
    return dense_query


def encode_sparse_query(query: str, bm25: BM25Encoder) -> dict:
    """Encodes the query with the fitted BM25 encoder loaded in Step 1."""
    sparse_query = bm25.encode_queries(query)
    print(f"[Retrieval/03] Sparse query vector: "
          f"{len(sparse_query['indices'])} non-zero term(s) -> {sparse_query}")
    return sparse_query


def encode_hybrid_query(
    query: str,
    embedding_model: SentenceTransformer,
    bm25: BM25Encoder,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[list[float], dict]:
    """Encodes `query` twice (dense + BM25 sparse), then scales both vectors
    with hybrid_convex_scale so Pinecone can search on both signals in one
    query. Returns (hybrid_dense, hybrid_sparse) ready for index.query().
    """
    print(f"[Retrieval/03] Encoding query {query!r} at alpha={alpha} "
          f"(0=pure sparse, 1=pure dense) ...")

    dense_query = encode_dense_query(query, embedding_model)
    sparse_query = encode_sparse_query(query, bm25)

    hybrid_dense, hybrid_sparse = hybrid_convex_scale(dense_query, sparse_query, alpha=alpha)
    print(f"[Retrieval/03] Scaled dense vector first 5 dims: {hybrid_dense[:5]}")
    print(f"[Retrieval/03] Scaled sparse vector:              {hybrid_sparse}")

    return hybrid_dense, hybrid_sparse


def retrieve_documents(
    edited_query: str,
    index,
    embedding_model: SentenceTransformer,
    bm25: BM25Encoder,
    alpha: float = DEFAULT_ALPHA,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """"Retrieving" stage: hybrid-searches `index` with `edited_query` and
    returns a candidate list of plain dicts (not Pinecone ScoredVector
    objects) with every field the later stages need: {id, text, source,
    score, rank}."""
    hybrid_dense, hybrid_sparse = encode_hybrid_query(edited_query, embedding_model, bm25, alpha=alpha)

    results = index.query(
        vector=hybrid_dense,
        sparse_vector=hybrid_sparse,
        top_k=top_k,
        include_metadata=True,
    )

    documents = []
    for rank, match in enumerate(results.matches, start=1):
        metadata = match.metadata or {}
        documents.append({
            "id": match.id,
            "score": match.score,
            "rank": rank,
            "text": metadata.get("text", ""),
            "source": metadata.get("source"),
        })

    print(f"[Retrieval/03] Retrieved {len(documents)} candidate(s) "
          f"(top_k={top_k}, alpha={alpha}):")
    for d in documents:
        preview = d["text"][:80].replace("\n", " ")
        print(f"[Retrieval/03]   [{d['rank']}] {d['id']} score={d['score']:.4f} text={preview!r}...")

    return documents


if __name__ == "__main__":
    from importlib import import_module

    _setup = import_module("01_pinecone_and_model_setup")
    _query_editing = import_module("02_query_editing")

    pc, index, embedding_model, bm25, llm = _setup.connect_everything()

    question = "Can I take money against my house without selling the property?"
    edited_query = _query_editing.edit_query(question, llm)

    documents = retrieve_documents(edited_query, index, embedding_model, bm25)
    print(f"\n[Retrieval/03] Done. {len(documents)} candidate(s) ready for reranking (Step 4).")
