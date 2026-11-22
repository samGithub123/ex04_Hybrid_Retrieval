# -*- coding: utf-8 -*-
"""
06_prompt_rendering_and_full_chain.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Retrieval Step 6 (final):
"Prompt Rendering", plus wiring every previous stage together into one
transparent LangChain chain:

    Input question
    -> assign edited_query        (02_query_editing.edit_query)
    -> assign retrieved_documents  (03_hybrid_retrieval.retrieve_documents)
    -> assign reranked_documents   (04_reranking.rerank_documents, scored vs ORIGINAL question)
    -> assign rendered_context     (05_context_building.render_context)
    -> assign sources               (05_context_building.extract_sources)
    -> assign prompt                (render_prompt, below)
    -> assign answer                (llm.invoke(prompt).content)

Built with RunnablePassthrough.assign() so the original `question` key
survives every step and every intermediate value (edited_query,
retrieved_documents, reranked_documents, rendered_context, sources, prompt,
answer) is present in the final result dict - nothing is hidden behind a
single retriever call.

Requires: pip install langchain-core langchain-openai sentence-transformers pinecone pinecone-text

What it does
render_prompt()  - "Prompt rendering": builds the final system+context+
   question messages the LLM sees - strict grounding rules first, then the
   Step 5 context block, then the original question.
build_chain()     - wires Steps 2-6 into one inspectable LCEL pipeline.
demo_intermediate_output() - runs one question through the chain and prints
   EVERY intermediate value in order, so the whole pipeline stays inspectable.
main()             - connects everything, builds the chain, and runs an
   interactive question loop (type 'exit' or 'bye' to quit).
"""

import os
import sys
from importlib import import_module
from time import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED_DIR = os.path.join(_PROJECT_ROOT, "Shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

_setup = import_module("01_pinecone_and_model_setup")
_query_editing = import_module("02_query_editing")
_retrieval = import_module("03_hybrid_retrieval")
_reranking = import_module("04_reranking")
_context = import_module("05_context_building")

# Grounding rules for a loan-products assistant: never invent rates/terms,
# say so when evidence is missing, cite sources, no side-effecting actions.
_ANSWER_SYSTEM_PROMPT = """
You are the BrightBridge Finance Loan Assistant. Answer using ONLY the
retrieved context below, which comes from official BrightBridge loan
product documents, eligibility guides and FAQs.

Rules:
- Use only the retrieved context. Never use outside knowledge and never guess.
- Never invent specific interest rates, fees, tenors or eligibility numbers
  that are not explicitly present in the context.
- If the context does not contain enough information to answer, say so
  plainly - do not invent an answer.
- If the context mixes guidance from more than one loan product and the
  question doesn't specify which one, say so and ask the user to confirm
  the exact product instead of blending guidance.
- Prefer short, direct bullet points or steps.
- This is a RAG assistant only - do not offer to take any action (no loan
  application submission, no fund transfer, no ticket creation).
"""


def render_prompt(question: str, rendered_context: str):
    """"Prompt Rendering" stage: renders the system+context+question
    messages. Returns a LangChain PromptValue, `.invoke()`-able directly by
    the LLM."""
    print("[Retrieval/06] Rendering final prompt (system rules + context + question) ...")
    prompt_template = ChatPromptTemplate([
        ("system", _ANSWER_SYSTEM_PROMPT),
        ("system", "Context:\n{rendered_context}"),
        ("human", "{question}"),
    ])
    prompt = prompt_template.invoke({"rendered_context": rendered_context, "question": question})
    print(f"[Retrieval/06] Rendered prompt messages:\n{prompt.to_messages()}")
    return prompt


def format_answer_with_sources(result: dict) -> str:
    """Appends a deterministic Sources list (from Step 5's extract_sources,
    computed in the chain below, not from the LLM's own text) after the
    LLM's answer, so citations are always accurate even if the model
    forgets to repeat a source."""
    lines = [result["answer"]]
    if result["sources"]:
        lines.append("")
        lines.append("Sources:")
        for s in result["sources"]:
            lines.append(f"- {s['source_name']}")
    return "\n".join(lines)


def build_chain(index, embedding_model, bm25, llm, cross_encoder, alpha: float = _retrieval.DEFAULT_ALPHA):
    """Builds the end-to-end LCEL RAG chain: query editing -> hybrid
    retrieval -> reranking -> context building -> prompt rendering ->
    answer generation. `index`/`embedding_model`/`bm25`/`llm` come from
    01_pinecone_and_model_setup.py, `cross_encoder` drives reranking (04)."""
    chain = (
        RunnablePassthrough.assign(
            edited_query=RunnableLambda(lambda d: _query_editing.edit_query(d["question"], llm))
        )
        .assign(
            retrieved_documents=RunnableLambda(
                lambda d: _retrieval.retrieve_documents(d["edited_query"], index, embedding_model, bm25, alpha=alpha)
            )
        )
        .assign(
            # Reranked against the ORIGINAL question, not the edited query -
            # see the note in 04_reranking.py.
            reranked_documents=RunnableLambda(
                lambda d: _reranking.rerank_documents(d["question"], d["retrieved_documents"], cross_encoder)
            )
        )
        .assign(
            rendered_context=RunnableLambda(lambda d: _context.render_context(d["reranked_documents"]))
        )
        .assign(
            sources=RunnableLambda(lambda d: _context.extract_sources(d["reranked_documents"]))
        )
        .assign(
            prompt=RunnableLambda(lambda d: render_prompt(d["question"], d["rendered_context"]))
        )
        .assign(
            answer=RunnableLambda(lambda d: llm.invoke(d["prompt"]).content)
        )
    )
    return chain


def demo_intermediate_output(chain, question: str) -> dict:
    """Runs `question` through the chain and prints EVERY intermediate
    value - original query, edited query, retrieved chunks, reranked
    chunks, rendered context, sources and final answer - so the pipeline
    stays fully inspectable end to end."""
    print("\n" + "#" * 70)
    print(f"# QUESTION: {question}")
    print("#" * 70)

    result = chain.invoke({"question": question})

    print("\n" + "-" * 70)
    print("[Retrieval/06] STAGE SUMMARY")
    print("-" * 70)

    print("\n1) edited_query (Step 2):")
    print(f"   {result['edited_query']}")

    print(f"\n2) retrieved_documents (Step 3, {len(result['retrieved_documents'])} candidate(s)):")
    for d in result["retrieved_documents"]:
        print(f"   [{d['rank']}] {d['id']}  score={d['score']:.4f}")

    print(f"\n3) reranked_documents (Step 4, {len(result['reranked_documents'])} kept):")
    for d in result["reranked_documents"]:
        print(f"   {d['id']}  cross_encoder_score={d['cross_encoder_score']:.4f}")

    print("\n4) rendered_context (Step 5):")
    print(result["rendered_context"])

    print("\n5) sources (Step 5):")
    print(result["sources"])

    print("\n6) final answer (Step 6):")
    print(format_answer_with_sources(result))

    return result


# A few representative loan questions, plus one deliberately outside the
# corpus, to exercise the "missing evidence" grounding rule.
TEST_QUESTIONS = [
    "Can I take money against my house without selling the property?",
    "What documents do I need for a personal loan?",
    "Is No-Cost EMI really interest free?",
    "Does BrightBridge offer student loans for studying abroad?",  # not in the corpus
]


def main() -> None:
    print("=" * 70)
    print("[Retrieval/06] STEP 1 - Connect to Pinecone + embedding model + BM25 + LLM")
    print("=" * 70)
    pc, index, embedding_model, bm25, llm = _setup.connect_everything()
    print(f"[Retrieval/06] Connected. Index stats: {index.describe_index_stats()}")

    cross_encoder = _reranking.get_cross_encoder()
    chain = build_chain(index, embedding_model, bm25, llm, cross_encoder)
    print("[Retrieval/06] Chain built: question -> edited_query -> retrieved_documents "
          "-> reranked_documents -> rendered_context -> sources -> prompt -> answer")

    print("\nAsk a loan-related question (type 'exit' or 'bye' to quit).")
    print(f"Example questions: {TEST_QUESTIONS}\n")

    while True:
        question = input("Q: ").strip()

        if question.lower() in ("exit", "bye"):
            print("Goodbye!")
            break
        if not question:
            continue

        start = time()
        demo_intermediate_output(chain, question)
        elapsed = time() - start
        print(f"\n({elapsed:.2f}s)\n")


if __name__ == "__main__":
    main()
