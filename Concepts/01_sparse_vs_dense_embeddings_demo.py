# -*- coding: utf-8 -*-
"""
01_sparse_vs_dense_embeddings_demo.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - standalone teaching demo, kept
separate from the real Upsert/ and Retrieval/ pipelines because it runs on a
tiny, hard-coded toy corpus (not the real loan-product documents in data/)
and needs none of Pinecone/keys.py/the embedding model.

This is the "build sparse vectors from scratch" walkthrough referenced in
../Hybrid_RAG_Sparse_Dense_Embeddings.md (sections 6-8): it shows, with
actual printed numbers, why raw word-count vectors need TF-IDF weighting
before they become a useful sparse representation - the same problem BM25
(used for real in Upsert/05_bm25_sparse_encoding.py) solves more robustly.

Requires: pip install scikit-learn pandas

What it does
1. Builds a Count Vector (bag-of-words) matrix for a 4-sentence toy corpus.
2. Encodes a query with the SAME fitted vectorizer and scores it against
   every document via a simple dot product.
3. Rebuilds the same corpus as a TF-IDF matrix and prints the weighted
   scores for a few key terms, to contrast against the raw counts.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Vocabulary => exhaustive list of all words in the corpus.
corpus = [
    "card payment failed card declined",
    "cash withdrawal failed at atm",
    "how to lock my lost card",
    "card payment completed but merchant refund pending",
]

query = "failed card payment"


def demo_count_vectors() -> None:
    """Step 1: raw word-count ("bag of words") sparse vectors."""
    print("=" * 70)
    print("STEP 1: Count Vectors (CountVectorizer)")
    print("=" * 70)

    count_vectorizer = CountVectorizer()

    # Fitting builds the vocabulary AND the per-document count matrix in one call.
    document_count_matrix = count_vectorizer.fit_transform(corpus)
    vocabulary = count_vectorizer.get_feature_names_out()
    print(f"Vocabulary ({len(vocabulary)} terms): {list(vocabulary)}")

    count_df = pd.DataFrame(document_count_matrix.toarray(), columns=vocabulary)
    print("\nDocument x term count matrix:")
    print(count_df)

    # Important: the query MUST be encoded with the same fitted vectorizer,
    # never a freshly-fit one, or the term<->index mapping will not match.
    query_count_vector = count_vectorizer.transform([query])
    print(f"\nQuery: {query!r}")
    print("Query count vector (compressed indices/values):")
    print(query_count_vector)
    print("Query count vector (dense, for readability only):")
    print(query_count_vector.toarray())

    # A plain dot product between each document row and the query vector is
    # the simplest possible "sparse similarity score".
    scores = (document_count_matrix @ query_count_vector.T).toarray().ravel()
    print("\nDot-product score of each document against the query:")
    for doc_text, score in zip(corpus, scores):
        print(f"  score={score}  ->  {doc_text!r}")


def demo_tfidf_vectors() -> None:
    """Step 2: TF-IDF - down-weights words that repeat a lot within one
    document, and words that are common across the WHOLE corpus."""
    print("\n" + "=" * 70)
    print("STEP 2: TF-IDF (TfidfVectorizer)")
    print("=" * 70)

    tfidf_vectorizer = TfidfVectorizer(norm=None)
    document_tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    tfidf_terms = tfidf_vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(document_tfidf_matrix.toarray(), columns=tfidf_terms)
    print("TF-IDF weights for a few key terms (rounded to 3 decimals):")
    print(tfidf_df[["card", "payment", "failed", "refund"]].round(3))

    print("\nFull TF-IDF matrix (dense, for readability only):")
    print(document_tfidf_matrix.toarray())

    print(
        "\nNote: 'card' appears in 3 of the 4 documents, so its IDF weight is "
        "low (common word); 'refund' appears in only 1 document, so it gets "
        "a much higher, more distinctive weight."
    )


if __name__ == "__main__":
    demo_count_vectors()
    demo_tfidf_vectors()
    print(
        "\n" + "=" * 70 +
        "\nTF-IDF still only measures literal word overlap - it has no notion "
        "of MEANING. The real pipeline in Upsert/ uses BM25 (a refined "
        "TF-IDF) for exactly this kind of sparse/keyword signal, combined "
        "with dense embeddings for semantic matching. See Upsert/05_bm25_"
        "sparse_encoding.py and ../upsert.md.\n" + "=" * 70
    )
