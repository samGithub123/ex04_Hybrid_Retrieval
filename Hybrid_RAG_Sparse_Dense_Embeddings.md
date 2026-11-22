# Hybrid RAG: Sparse & Dense Embeddings
### Session 9 — Agentic Workflow Manager, Cohort 2

---

## Table of Contents
1. [Overview](#1-overview)
2. [The Problem with Pre-Filtering](#2-the-problem-with-pre-filtering)
3. [Dense vs. Sparse Embeddings](#3-dense-vs-sparse-embeddings)
4. [The Exam Analogy for Hybrid Matching](#4-the-exam-analogy-for-hybrid-matching)
5. [Agent Architecture — AWS AgentCore](#5-agent-architecture--aws-agentcore)
6. [Building Sparse Vectors from Scratch — Count Vectors](#6-building-sparse-vectors-from-scratch--count-vectors)
7. [Compression of Sparse Vectors](#7-compression-of-sparse-vectors)
8. [From Count Vectors to TF-IDF](#8-from-count-vectors-to-tf-idf)
9. [BM25 — An Advanced TF-IDF](#9-bm25--an-advanced-tf-idf)
10. [Hands-On: Hybrid Pipeline in Pinecone](#10-hands-on-hybrid-pipeline-in-pinecone)
11. [Retrieval with Hybrid Search](#11-retrieval-with-hybrid-search)
12. [Other Topics Covered](#12-other-topics-covered)
13. [Homework](#13-homework)
14. [Next Session Preview](#14-next-session-preview)

---

## 1. Overview

This session covered **hybrid retrieval-augmented generation (RAG)** — combining *sparse* (keyword-based) and *dense* (semantic) embeddings to improve retrieval quality over either method alone. It also included a walkthrough of production agent architecture using AWS AgentCore as a reference model, and a hands-on build of a hybrid retrieval pipeline in Pinecone.

---

## 2. The Problem with Pre-Filtering

Pre-filtering shortlists documents (e.g., resumes) based on **keyword presence** before semantic/vector matching runs.

**Two core failure modes:**

| Failure Mode | Example |
|---|---|
| **False exclusion** | A strong candidate is filtered out for lacking an exact keyword — e.g., resume says *"LLM Developer"* but the filter looks for *"GenAI"* |
| **False inclusion** | A weak candidate passes the filter by matching just one keyword — e.g., only has *"Python"* |

**Conclusion:** Pre-filtering is useful for **reducing search space**, but is a poor tool for **precise key-term matching**. It should be paired with additional scoring layers rather than used as a standalone gate.

---

## 3. Dense vs. Sparse Embeddings

| Type | Captures | Behavior |
|---|---|---|
| **Dense embeddings** | Meaning / semantics | Similar *meaning* → higher score, even if exact words differ |
| **Sparse embeddings** | Literal word overlap | Similar *words* → higher score, regardless of meaning (closer to keyword/lexical search — e.g., "Ctrl+F" or early Google search, though not identical to pure lexical search) |

**Key clarification:** Sparse embeddings have **no semantic role** — they are purely word-match based. Sparse and dense embeddings are generated **independently and in parallel** from the same input text; sparse output is *not* fed into the dense model. The two are combined only at the scoring/retrieval stage.

---

## 4. The Exam Analogy for Hybrid Matching

A recurring analogy (school exams / recruitment exams) was used to explain the shift from **sequential filtering** to **hybrid scoring**:

1. **Exam 1 — Pre-filtering:** A sequential gate to shrink the candidate pool.
   *Example rule: "must contain at least one of: Python, GenAI, Data Science."*
2. **Exam 2 — Sparse matching:** Run on all candidates who passed Exam 1. Produces a sparse score (e.g., `0.35`).
3. **Exam 3 — Dense matching:** Run on the same candidates in parallel. Produces a dense score (e.g., `0.77`).

Instead of eliminating candidates sequentially, the **scores from Exam 2 and Exam 3 are combined (weighted/averaged)** — this combination *is* Hybrid Matching. The weighting between sparse and dense is **tunable per use case**, allowing candidates who underperform on one axis to still succeed via the other.

---

## 5. Agent Architecture — AWS AgentCore

AWS AgentCore was used as a reference diagram for what a **production-grade agentic system** requires. The core formula:

> **Model + Harness = Agent**

### 5.1 Harness — the four pillars

| Pillar | Components |
|---|---|
| **Models** | Bedrock (Anthropic, OpenAI, Nova, Llama, Qwen…), Direct APIs (Anthropic, OpenAI, Gemini), LiteLLM |
| **Tools** | MCP servers, AgentCore Gateway & Policy, AgentCore Browser, AgentCore Code Interpreter, web search, inline functions |
| **Memory** | Short-term & long-term memory (managed by default, or bring-your-own) |
| **Skills** | AWS skills, Git source, S3 source, container-baked skills |

### 5.2 Runtime
- Deployed via **Firecracker microVM**
- Filesystem: managed session storage, or bring-your-own (S3, Files, EFS)
- Environment: default or bring-your-own container
- Supports shell command execution

### 5.3 Identity
- Inbound authentication
- Outbound authentication
- OAuth
- API keys
- Governs **who can call your agent** and **what your agent is allowed to call**

### 5.4 Observability
Tracked via CloudWatch traces, logs, and metrics (unified harness view):
- Compute uptime/downtime
- Memory fill/empty rate
- Tool call rate
- Token consumption

### 5.5 Evaluation & Optimization
- Built-in and custom evaluators
- A/B testing
- Recommendations, versioning, and endpoints
- Needed to **statistically validate** whether a change (e.g., loosening pre-filtering) actually improves results

> **Key takeaway:** Skipping any one of these five layers (Harness, Runtime, Identity, Observability, Evaluation) limits you to building **POCs**, not production-grade agents.

**Note:** *Deep Agents* (LangChain's newer framework) were mentioned briefly as a bundled harness with more built-in autonomy (sandboxed filesystem/tools) — to be covered in a future session.

---

## 6. Building Sparse Vectors from Scratch — Count Vectors

- **Vocabulary** = the exhaustive list of unique words across the entire document corpus.
- **Count Vector** = for each document, a vector recording the count of every vocabulary word present in it (`0` if absent).

**Implementation:** `sklearn.feature_extraction.text.CountVectorizer`

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(corpus)   # sparse matrix
dense_array = count_matrix.toarray()              # for visualization only
```

> **Important:** Query vectors must be generated using the **same fitted vectorizer** used on the corpus. Save and reuse it (e.g., via pickle) — a freshly-fit vectorizer will produce an incompatible vocabulary/index mapping.

---

## 7. Compression of Sparse Vectors

Large vocabularies (potentially millions of words) make storing full dense arrays impractical. Sparse vectors are instead stored in **compressed format**: only the **positions (indices)** and their **values** are retained — zeros are not stored.

**Analogy:** Like assigning roll numbers to students — you only record *which* roll numbers were present, not a full list padded with zeros for everyone absent.

This is the principle underlying **CSR (Compressed Sparse Row)** storage.

---

## 8. From Count Vectors to TF-IDF

### 8.1 The Problem
A word repeated many times in one document (e.g., *"card"* appearing 20 times) artificially inflates that document's vector weight — causing a poor match against a query where the same word appears only once, even if the two are topically aligned.

### 8.2 The Fix — TF-IDF
**TF-IDF = Term Frequency (TF) × Inverse Document Frequency (IDF)**

**Term Frequency (TF):**
```
TF(doc, term) = (count of term in doc) / (total words in doc)
```
*Refinement from the worked example: TF is **not** the raw count — it must be normalized by document length. (Raw count of "card" in Doc 1 = 2 → normalized by total word count in Doc 1.)*

**Document Frequency (df):**
The number of documents in the corpus containing the term.
*Worked example: "card" appears in 3 documents → df("card") = 3.*

**Inverse Document Frequency (IDF):**
```
IDF(t) = log(N / df(t))
```
where `N` = total number of documents in the corpus, and `df(t)` = number of documents containing term `t`.

IDF **down-weights common words** that appear across many documents (less distinctive terms).

> **Analogy:** Like the word *"awesome"* losing its impact once everyone overuses it (Gen Z slang example).

> **Worked example (from the whiteboard):**
> - Term: `"card"`, Document: `Doc1`
> - Raw count of `"card"` in `Doc1` = `2`
> - `TF(Doc1, "card") = 2 / (total number of words in Doc1)` — the raw count of 2 must be divided by document length; it is **not** used as-is.
> - `"card"` appears across `3` documents in the corpus → `df("card") = 3`
> - Applying the IDF formula: `IDF("card") = log(N / 3)`, where `N` is the total number of documents in the corpus.

**Implementation:** `sklearn.feature_extraction.text.TfidfVectorizer` — same workflow as `CountVectorizer`, but outputs weighted scores instead of raw counts.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
```

### 8.3 Limitation
TF-IDF **does not capture context or meaning**. It is never used alone in practice — always combined with dense embeddings in a hybrid setup.

---

## 9. BM25 — An Advanced TF-IDF

- Considered an improvement over plain TF-IDF.
- Internal algorithmic details were **not** covered in depth in this session — flagged as self-study material and a common interview topic.
- Pinecone provides a dedicated BM25 encoder via the `pinecone_text` package, which automatically handles stop-word removal and NLP preprocessing.

```python
from pinecone_text.sparse import BM25Encoder

bm25 = BM25Encoder()
bm25.fit(corpus)                  # corpus = 43 documents/chunks in this session
sparse_vectors = bm25.encode_documents(corpus)
```

- Each sparse embedding is represented as **indices + values** (compressed format).
- BM25 is available both in scikit-learn and in Pinecone's own library — Pinecone's implementation integrates more directly with its vector DB storage format.

---

## 10. Hands-On: Hybrid Pipeline in Pinecone

Step-by-step pipeline (extends prior sessions' pipeline with sparse embeddings):

1. **Create a Pinecone index** named `vector-loan-hybrid` to store both dense and sparse embeddings.
2. **Access the index.**
3. **Load and chunk documents** using the same semantic chunking code from previous sessions.
4. **Generate sparse embeddings** via the BM25 encoder (fit on the corpus).
5. **Generate dense embeddings** via the sentence transformer (same model as prior sessions).
6. **Combine into a vector record:**
   ```python
   {
     "id": doc_id,
     "values": dense_embedding,        # dense vector
     "sparse_values": {                # BM25 output
         "indices": [...],
         "values": [...]
     },
     "metadata": {
         "text": chunk_text,
         "source": source_name
     }
   }
   ```
7. **Upsert in batches** to the Pinecone index (43 vectors in this session; batching recommended as best practice for larger datasets).
8. **Save the trained BM25 model as JSON** (not pickle — pickle is not portable across systems) to a local project folder (e.g., `/models`). This exact model must be reused later to encode queries, otherwise sparse and dense vector spaces will mismatch and produce inconsistent retrieval results.

> **Environment note:** When working locally (e.g., VS Code, rather than Colab), you must explicitly specify the correct local file path for the BM25 JSON file to save properly.

---

## 11. Retrieval with Hybrid Search

At query time, the query is encoded **twice**:
1. Via the **dense model** → dense query vector
2. Via the **loaded BM25 model** → sparse query vector

Pinecone's `hybrid_convex_scale` function combines the two using an **alpha (α) weighting parameter**:

| Alpha value | Effect |
|---|---|
| `alpha = 0` | Pure sparse search |
| `alpha = 0.5` | Equal weight — dense and sparse each scaled by 0.5 |
| `alpha = 1` | Pure dense search |

```python
from pinecone_text.hybrid import hybrid_convex_scale

dense_vec, sparse_vec = hybrid_convex_scale(dense_query_vec, sparse_query_vec, alpha=0.5)

results = index.query(
    vector=dense_vec,
    sparse_vector=sparse_vec,
    top_k=5,
    include_metadata=True
)
```

The session demonstrated combined hybrid scores (e.g., `0.24`) versus dense-only scores (e.g., `0.35`) to illustrate how score combination shifts ranking behavior.

### 11.1 Worked Example (from the whiteboard)

Given a query encoded into both a dense vector and a sparse vector:

```
Query → Dense  → [0.2, 0.3, 0.4]
Query → Sparse → [0.5, 0.7, 0.8]
```

Calling `hybrid_convex_scale(dense_query, sparse_query, alpha=0.5)` **scales each vector independently** before it is sent to the index:

```
scaled_dense  = dense_vector  × alpha        →  [0.1,  0.15, 0.2]
scaled_sparse = sparse_vector × (1 − alpha)  →  [0.25, 0.35, 0.4]
```

| Term | Formula | Result |
|---|---|---|
| Scaled dense | `Dense × α` | `[0.2, 0.3, 0.4] × 0.5 = [0.1, 0.15, 0.2]` |
| Scaled sparse | `Sparse × (1 − α)` | `[0.5, 0.7, 0.8] × 0.5 = [0.25, 0.35, 0.4]` |

At `alpha = 0.5`, both vectors are scaled down equally (each by a factor of `0.5`), giving dense and sparse **equal influence** on the final combined score. Lowering `alpha` shifts more weight onto the sparse (keyword) vector; raising it shifts more weight onto the dense (semantic) vector.

---

## 12. Other Topics Covered

- **Namespaces in Pinecone:** Function like partitions in a traditional database, defined at index creation. Useful for agents that need to search across separate data domains.
- **HyDE (Hypothetical Document Embeddings):** Lets the LLM generate a hypothetical answer *before* looking at the data, then uses that answer to build the retrieval query. **Caution:** avoid using HyDE for open-ended questions due to hallucination risk.
- **Knowledge Fabric / Knowledge Graphs / Structured Memory:** Related concepts for organizing information; knowledge graphs will be covered in a future session.
- **MHQE (Multi-Hop Query Expansion):** Previously covered; noted as more challenging than HyDE.

---

## 13. Homework

Build the **complete retrieval pipeline** with sparse embeddings included — not just the retrieval snippet demonstrated in class, but the full flow:

```
Query enrichment/editing → Retrieval → Reranking → Context building → Prompt rendering
```

---

## 14. Next Session Preview

- Complete the **LLM-as-a-judge** framework
- **Monitoring and observability** in RAG systems
- Key evaluation metrics: **precision, recall, correctness, relevance** — closing the gap on knowing whether a RAG system's final answer is actually correct
- Introduction to **RAGAS metrics** and **G-Eval**
