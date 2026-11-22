# -*- coding: utf-8 -*-
"""
02_query_editing.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Retrieval Step 2: "Query
editing". Rewrites the user's informal question into a clearer search query
before it reaches Step 3 (retrieval) - both the dense embedding AND the BM25
sparse encoding get this SAME edited text, so a clearer query improves both
retrieval signals at once.

Unlike the appliance-manual reference this was modelled on
(../../miniProj/ApplianceComunicationSystem/Retrival/02_query_editing.py),
there is no "Prefiltering" stage here: the loan document corpus is a flat
folder of .txt files with no per-document metadata catalog (no model/
product_type/document_type fields to filter on), so this file only covers
the query-rewrite half of the reference's Step 2.

Requires: pip install pydantic langchain-core langchain-openai

What it does
Sends the user's original question to the LLM with a system prompt asking
   for a clearer search-query rewrite (same intent, expanded wording).
Uses a Pydantic structured-output schema so the LLM's reply always parses
   as a single clean string, never free-form chatter.
Never invents details or changes the question's intent - it only rewrites.
"""

from pydantic import BaseModel, Field


class EditedQueryOutput(BaseModel):
    edited_query: str = Field(
        description="The user's question rewritten as a clearer search query, same intent, "
                    "with informal wording expanded into the terms loan documents actually use."
    )


_EDIT_SYSTEM_PROMPT = """
You are a search-query rewriting assistant for a loan-products RAG system
(personal loans, home loans, loan against property, two-wheeler loans,
no-cost EMI, eligibility, FAQs).

Rewrite the user's question into a clearer search query for retrieving
relevant loan-document content.

Rules:
- Preserve the original intent exactly - do not answer the question.
- Expand informal or everyday wording into the terms the loan documents
  actually use (e.g. "money against my house" -> "loan against property,
  secured loan using residential property as collateral").
- Keep any product names, amounts or timeframes exactly as given.
- Do not add a loan product or claim that was not in the original question.
- Keep it short - one sentence.
"""


def edit_query(query: str, llm) -> str:
    """"Query editing" stage: rewrites `query` into a clearer search query
    without changing its intent. Returns the edited query string."""
    print(f"[Retrieval/02] Editing question: {query!r}")

    edit_llm = llm.with_structured_output(EditedQueryOutput)
    response = edit_llm.invoke([
        ("system", _EDIT_SYSTEM_PROMPT),
        ("human", query),
    ])

    print(f"[Retrieval/02] Edited query:      {response.edited_query!r}")
    return response.edited_query


if __name__ == "__main__":
    from importlib import import_module

    _setup = import_module("01_pinecone_and_model_setup")
    llm = _setup.get_llm()

    sample_query = "Can I take money against my house without selling the property?"
    edited = edit_query(sample_query, llm)

    print(f"\n[Retrieval/02] Original: {sample_query!r}")
    print(f"[Retrieval/02] Edited:   {edited!r}")
