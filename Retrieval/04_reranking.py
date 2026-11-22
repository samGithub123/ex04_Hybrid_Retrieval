# -*- coding: utf-8 -*-
"""
04_reranking.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Retrieval Step 4: "Re-ranking".
Reorders Step 3's hybrid-search candidates against the ORIGINAL user
question (not the edited query) using a Cross-Encoder, which - unlike the
dense/sparse vector search in Step 3 - scores the question and each chunk
JOINTLY, so it's more accurate at judging true relevance.

Scoring against the ORIGINAL question (not the edited one) matters: the
edited query is optimized for recall (cast a wide net across dense+sparse
search), while reranking is optimized for precision against what the user
actually asked.

Requires: pip install sentence-transformers

What it does
Loads a cross-encoder/ms-marco-MiniLM-L-6-v2 Cross-Encoder model.
Scores every (original_question, candidate_text) pair jointly.
Sorts all candidates by that score, best first, and keeps only the top_k
   (default 3) - these are what Step 5 renders into the final context.
"""

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 3


def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    print(f"[Retrieval/04] Loading cross-encoder '{CROSS_ENCODER_MODEL_NAME}' ...")
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


def rerank_documents(question: str, retrieved_documents: list[dict], cross_encoder, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """"Re-ranking" stage: reorders `retrieved_documents` (as returned by
    03_hybrid_retrieval.retrieve_documents) against the ORIGINAL `question`
    using the Cross-Encoder's relevance score for each (question, chunk)
    pair, and returns the top_k, sorted best-first."""
    if not retrieved_documents:
        print("[Retrieval/04] No candidates to rerank.")
        return []

    print(f"[Retrieval/04] Reranking {len(retrieved_documents)} candidate(s) "
          f"against ORIGINAL question: {question!r}")

    pairs = [[question, doc["text"]] for doc in retrieved_documents]
    scores = cross_encoder.predict(pairs)

    for doc, score in zip(retrieved_documents, scores):
        doc["cross_encoder_score"] = float(score)

    reranked = sorted(retrieved_documents, key=lambda d: d["cross_encoder_score"], reverse=True)
    kept = reranked[:top_k]

    print(f"[Retrieval/04] Cross-encoder scores (all {len(reranked)} candidate(s), best first):")
    for d in reranked:
        marker = "KEEP" if d in kept else "drop"
        print(f"[Retrieval/04]   [{marker}] {d['id']} cross_encoder_score={d['cross_encoder_score']:.4f}")

    print(f"[Retrieval/04] Kept top {len(kept)} (top_k={top_k}) for context building (Step 5).")
    return kept


if __name__ == "__main__":
    from importlib import import_module

    _setup = import_module("01_pinecone_and_model_setup")
    _query_editing = import_module("02_query_editing")
    _retrieval = import_module("03_hybrid_retrieval")

    pc, index, embedding_model, bm25, llm = _setup.connect_everything()

    question = "Can I take money against my house without selling the property?"
    edited_query = _query_editing.edit_query(question, llm)
    retrieved_documents = _retrieval.retrieve_documents(edited_query, index, embedding_model, bm25)

    cross_encoder = get_cross_encoder()
    reranked_documents = rerank_documents(question, retrieved_documents, cross_encoder)

    print("\n[Retrieval/04] Final reranked list:")
    for d in reranked_documents:
        print(f"  {d['id']} (cross_encoder_score={d['cross_encoder_score']:.4f})")
