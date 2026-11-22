# Hybrid Retrieval (Dense + BM25 Sparse) RAG — Loan Documents

> Companion notes: [Hybrid_RAG_Sparse_Dense_Embeddings.md](Hybrid_RAG_Sparse_Dense_Embeddings.md)
> (session write-up: why hybrid search, dense vs. sparse, BM25 theory)

## What this exercise does, in one paragraph

This folder turns a handful of plain-text loan product/FAQ documents
(`data/*.txt`) into a **hybrid** Pinecone index — every stored vector carries
both a dense (semantic) embedding *and* a BM25 (keyword) sparse vector — then
runs a full retrieval chain on top of it: query editing, hybrid retrieval,
reranking, context building, and prompt rendering + grounded answer
generation. The code is organized into four folders so each concept from the
session notes has its own runnable, self-contained script instead of one
long inline notebook-style file.

## Folder layout

| Folder | Purpose |
|---|---|
| [`Concepts/`](Concepts/) | Standalone teaching demo: Count Vectors vs. TF-IDF on a toy corpus. No Pinecone/keys needed. |
| [`Shared/`](Shared/) | One place for the index name, embedding model, vector dimension, metric, BM25 params path, and LLM — used by both pipelines below so they can never drift apart. See [`hybrid_pinecone_common.py`](Shared/hybrid_pinecone_common.py). |
| [`Upsert/`](Upsert/) | Ingestion pipeline (7 numbered scripts, run in order 01→07). Details: [upsert.md](upsert.md). |
| [`Retrieval/`](Retrieval/) | Query-time pipeline (6 numbered scripts, run in order 01→06): query editing → hybrid retrieval → reranking → context building → prompt rendering → answer generation. Details: [retrieval.md](retrieval.md). |
| [`data/`](data/) | Source `.txt` documents that get chunked, embedded and upserted. |
| [`json/`](json/) | `bm25_params.json` — the fitted BM25 encoder's statistics, written by `Upsert/05_bm25_sparse_encoding.py`, read by `Retrieval/01_pinecone_and_model_setup.py`. Must never be hand-edited. |

`keys.py`, `.env` / `.env.example` live at the project root and are shared by
every script in every folder above (see `keys.py`'s own docstring).

## Running it

### VS Code setup

Create the project-local environment and install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `.env` with the Pinecone and OpenAI credentials, then select
`.venv\Scripts\python.exe` using **Python: Select Interpreter** in VS Code.
The included VS Code launch and task configurations run with this environment.

```bash
# One-shot, end-to-end, non-interactive (ingests + runs one demo question):
python main.py

# Or run each stage on its own, in order:
python Upsert/01_pinecone_index_setup.py
python Upsert/02_load_source_documents.py
python Upsert/03_semantic_chunking.py
python Upsert/04_build_chunked_documents.py
python Upsert/05_bm25_sparse_encoding.py
python Upsert/06_dense_embedding_and_vector_build.py
python Upsert/07_upsert_to_pinecone.py      # <- actually writes to Pinecone

python Retrieval/01_pinecone_and_model_setup.py
python Retrieval/02_query_editing.py
python Retrieval/03_hybrid_retrieval.py
python Retrieval/04_reranking.py
python Retrieval/05_context_building.py
python Retrieval/06_prompt_rendering_and_full_chain.py  # <- interactive Q&A loop (type 'exit'/'bye' to quit)
```

Every script prints a `[FolderName/NN]`-prefixed trace of what it's doing at
each step (files found, chunk counts, vector shapes, batch progress, edited
queries, retrieval/rerank scores, rendered context, sources, and the final
grounded answer) so the pipeline stays inspectable end to end.

## Why a dedicated `dotproduct` index

A single Pinecone record here carries **both** a dense `values` vector and a
BM25 `sparse_values` vector. Pinecone requires the `dotproduct` metric for
that combination — an existing `cosine` index cannot be converted in place —
so `Upsert/01_pinecone_index_setup.py` always creates its own index
(`vector-loan-hybrid`) rather than reusing one from another exercise.

## Why BM25 params are saved to `json/bm25_params.json`

The BM25 encoder must be **fitted once** on the full chunk corpus at upsert
time, then reused as-is at query time. Re-fitting a fresh encoder on the
query side would assign different term→index numbers, silently breaking
retrieval (the sparse query vector would no longer line up with the sparse
vectors already stored in Pinecone). Saving as JSON (not pickle) keeps the
file portable across machines — see `Upsert/05_bm25_sparse_encoding.py` and
`Shared/hybrid_pinecone_common.py::load_bm25_encoder`.
