# -*- coding: utf-8 -*-
"""
main.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - single entry point that runs the
whole project end-to-end, non-interactively:

    Concepts/01_sparse_vs_dense_embeddings_demo.py  (optional, educational)
    Upsert/01..07      (ingestion: index -> load -> chunk -> BM25 -> dense -> upsert)
    Retrieval/01..06   (query-time: connect -> edit -> retrieve -> rerank ->
                         build context -> render prompt -> generate answer)

This file used to contain ALL of that logic inline as one long script. It has
since been reorganized into the Concepts/, Upsert/, Retrieval/ and Shared/
folders (see README.md, upsert.md, retrieval.md) so each stage can be read,
run and tested on its own - `python main.py` still reproduces the original
end-to-end run, just by calling into the now-organized pieces.

Note: Retrieval/06_prompt_rendering_and_full_chain.py's own `main()` is an
INTERACTIVE question loop (it reads from stdin until you type 'exit'/'bye').
Running it here directly would make `python main.py` hang waiting for
keyboard input, so this file instead builds the same chain and runs it
non-interactively on one demo question - run
`python Retrieval/06_prompt_rendering_and_full_chain.py` yourself for the
interactive Q&A experience.

Requires: pip install pinecone pinecone-text sentence-transformers scikit-learn pandas langchain-core langchain-openai
"""

import os
import sys
from importlib import import_module

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Each numbered script imports its own siblings via import_module(), which
# only works if that script's own folder is on sys.path - so every pipeline
# folder used here is added up front.
for _sub in ("Concepts", "Upsert", "Retrieval", "Shared"):
    _dir = os.path.join(_PROJECT_ROOT, _sub)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)


def run_concepts_demo() -> None:
    """Optional educational detour: sparse (count/TF-IDF) vector basics on a
    tiny toy corpus. Does not touch Pinecone or the real loan-document data."""
    print("\n" + "#" * 70)
    print("# STAGE 0: CONCEPTS DEMO (Concepts/01_sparse_vs_dense_embeddings_demo.py)")
    print("#" * 70)
    concepts_demo = import_module("01_sparse_vs_dense_embeddings_demo")
    concepts_demo.demo_count_vectors()
    concepts_demo.demo_tfidf_vectors()


def run_upsert_pipeline() -> None:
    """Ingestion: create the index, load+chunk the loan documents, fit BM25,
    dense-embed, and upsert every hybrid vector into Pinecone."""
    print("\n" + "#" * 70)
    print("# STAGE 1: UPSERT PIPELINE (Upsert/01_pinecone_index_setup.py -> Upsert/07_upsert_to_pinecone.py)")
    print("#" * 70)
    upsert_main = import_module("07_upsert_to_pinecone")
    upsert_main.main()


def run_retrieval_demo() -> None:
    """Query-time: connect to the now-populated index, build the full
    query-editing -> hybrid-retrieval -> reranking -> context-building ->
    prompt-rendering -> answer-generation chain, and run it non-interactively
    on one demo question, printing every intermediate value."""
    print("\n" + "#" * 70)
    print("# STAGE 2: RETRIEVAL PIPELINE (Retrieval/01_pinecone_and_model_setup.py -> Retrieval/06_prompt_rendering_and_full_chain.py)")
    print("#" * 70)
    retrieval_setup = import_module("01_pinecone_and_model_setup")
    retrieval_reranking = import_module("04_reranking")
    retrieval_chain = import_module("06_prompt_rendering_and_full_chain")

    pc, index, embedding_model, bm25, llm = retrieval_setup.connect_everything()
    cross_encoder = retrieval_reranking.get_cross_encoder()
    chain = retrieval_chain.build_chain(index, embedding_model, bm25, llm, cross_encoder)

    retrieval_chain.demo_intermediate_output(chain, retrieval_chain.TEST_QUESTIONS[0])


if __name__ == "__main__":
    run_concepts_demo()
    run_upsert_pipeline()
    run_retrieval_demo()

    print("\n" + "#" * 70)
    print("# DONE - see README.md / upsert.md / retrieval.md for a step-by-step breakdown.")
    print("#" * 70)
