# -*- coding: utf-8 -*-
"""
05_context_building.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Retrieval Step 5: "Build
Context". Turns Step 4's reranked chunks into what the LLM actually sees
(a single prompt-ready text block), plus a citation list that doesn't
depend on the LLM repeating sources correctly.

Requires: none beyond the standard library

What it does
render_context()  - concatenates the reranked chunks into one text block,
   each chunk labelled with a [Source N] header + its source filename, under
   a character budget (default 1600 chars) so the prompt stays a bounded
   size regardless of how large the reranked chunks are.
extract_sources()  - walks the reranked chunks and returns a deduplicated
   {source} list, in first-seen (i.e. strongest match first) order -
   independent of whether the LLM's own answer text faithfully repeats it.
"""

import os

DEFAULT_MAX_CHARS = 1600


def _source_label(doc: dict) -> str:
    """Loan documents only carry a file path in metadata (no title/model
    fields like the appliance-manual reference), so the citation label is
    just the source file's basename, e.g. 'home_loan.txt'."""
    source = doc.get("source")
    return os.path.basename(source) if source else "unknown source"


def render_context(reranked_documents: list[dict], max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """"Build Context" stage: concatenates `reranked_documents` into a single
    context block, each chunk preceded by a header naming its source and
    retrieval/rerank scores, truncated to a `max_chars` budget. This is what
    the answer-generation prompt (Step 6) actually receives - not the full
    corpus, only these selected chunks."""
    print(f"[Retrieval/05] Building context from {len(reranked_documents)} reranked chunk(s), "
          f"max_chars={max_chars} ...")

    parts = []
    total_chars = 0

    for i, doc in enumerate(reranked_documents, start=1):
        header = (
            f"[Source {i}] {_source_label(doc)} "
            f"(retrieval_score={doc.get('score', 0):.3f}, "
            f"rerank_score={doc.get('cross_encoder_score', 0):.3f})"
        )
        block = f"{header}\n{doc['text'].strip()}"

        remaining_chars = max_chars - total_chars
        if remaining_chars <= 0:
            print(f"[Retrieval/05]   budget exhausted before Source {i} - stopping.")
            break
        if len(block) > remaining_chars:
            print(f"[Retrieval/05]   Source {i} truncated to fit remaining budget "
                  f"({len(block)} -> {remaining_chars} chars).")
            block = block[:remaining_chars]
        else:
            print(f"[Retrieval/05]   Source {i} added as-is ({len(block)} chars).")

        parts.append(block)
        total_chars += len(block)

    rendered = "\n\n---\n\n".join(parts)
    print(f"[Retrieval/05] Final rendered context: {len(rendered)} chars total.")
    return rendered


def extract_sources(reranked_documents: list[dict]) -> list[dict]:
    """Deduplicated {source} list for the chunks actually used, in the order
    they first appear (i.e. strongest match first)."""
    seen = set()
    sources = []
    for doc in reranked_documents:
        key = doc.get("source")
        if key and key not in seen:
            seen.add(key)
            sources.append({"source": key, "source_name": _source_label(doc)})

    print(f"[Retrieval/05] Deduplicated sources ({len(sources)}): "
          f"{[s['source_name'] for s in sources]}")
    return sources


if __name__ == "__main__":
    from importlib import import_module

    _setup = import_module("01_pinecone_and_model_setup")
    _query_editing = import_module("02_query_editing")
    _retrieval = import_module("03_hybrid_retrieval")
    _reranking = import_module("04_reranking")

    pc, index, embedding_model, bm25, llm = _setup.connect_everything()

    question = "Can I take money against my house without selling the property?"
    edited_query = _query_editing.edit_query(question, llm)
    retrieved_documents = _retrieval.retrieve_documents(edited_query, index, embedding_model, bm25)

    cross_encoder = _reranking.get_cross_encoder()
    reranked_documents = _reranking.rerank_documents(question, retrieved_documents, cross_encoder)

    rendered_context = render_context(reranked_documents)
    sources = extract_sources(reranked_documents)

    print("\n[Retrieval/05] Rendered context:\n")
    print(rendered_context)
    print("\n[Retrieval/05] Sources:")
    print(sources)
