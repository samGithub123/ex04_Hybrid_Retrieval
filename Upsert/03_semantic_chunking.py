# -*- coding: utf-8 -*-
"""
03_semantic_chunking.py

Hybrid Retrieval (Dense + BM25 Sparse) RAG - Step 3: split each raw
document's text into chunks that are small enough to embed usefully, but
each still make sense on their own when retrieved in isolation.

Logic:
- split into paragraph units (or sentence units if there are no real
  paragraphs)
- embed each unit with the dense embedding model
- compare adjacent units using cosine similarity
- merge units while they stay semantically close, or the chunk is still
  small/below target size
- close the chunk and start a new one once it gets too large or the topic
  clearly shifts
- force-split (hard_split_large_text) any chunk that still ends up too long

Requires: pip install sentence-transformers numpy

What it does
Splits each document's text into paragraph/sentence units.
Embeds each unit and merges adjacent ones while their meaning stays similar.
Starts a new chunk once similarity drops or size limits are hit.
Guarantees no chunk exceeds hard_max_chunk_chars via a fallback hard split.
"""

import re

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_MODEL_NAME = "paraphrase-MiniLM-L6-v2"  # 384-dim - must match EMBEDDING_DIMENSION in Shared/hybrid_pinecone_common.py


def split_into_semantic_units(text: str) -> list[str]:
    """First try paragraph-level splitting.
    If the text has no real paragraphs, fall back to sentence-like splitting.
    Returns: list[str]
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def hard_split_large_text(text: str, hard_max_chunk_chars: int = 1200) -> list[str]:
    """Safety fallback if one chunk becomes too large: slices it into
    fixed-size pieces so nothing downstream ever sees an oversized chunk.
    Returns: list[str]
    """
    if len(text) <= hard_max_chunk_chars:
        return [text]

    pieces = []
    start = 0
    while start < len(text):
        end = start + hard_max_chunk_chars
        pieces.append(text[start:end].strip())
        start = end

    return [p for p in pieces if p]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D numpy vectors:
        cosine_similarity(A, B) = (A . B) / (||A|| * ||B||)
    where A . B is the dot product and ||A||, ||B|| are the vector lengths.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def semantic_chunk_text_embedding(
    text: str,
    model: SentenceTransformer,
    similarity_threshold: float = 0.65,
    min_chunk_chars: int = 300,
    target_chunk_chars: int = 700,
    hard_max_chunk_chars: int = 1200,
) -> list[str]:
    """
    Embedding-based semantic chunking.

    Logic:
    - split into paragraph/sentence units
    - embed each unit
    - compare adjacent units using cosine similarity
    - merge if semantically close
    - split if topic seems to shift
    - force split if chunk gets too large

    Returns: list[str]
    """
    units = split_into_semantic_units(text)
    if not units:
        return []

    if len(units) == 1:
        return hard_split_large_text(units[0], hard_max_chunk_chars)

    # Encode all units once
    unit_embeddings = model.encode(units)

    chunks = []
    current_chunk = units[0]
    current_chunk_emb = unit_embeddings[0]

    for i in range(1, len(units)):
        next_unit = units[i]
        next_emb = unit_embeddings[i]

        sim = cosine_similarity(current_chunk_emb, next_emb)
        proposed_chunk = current_chunk + "\n\n" + next_unit

        # If current chunk is too small, keep growing it first
        if len(current_chunk) < min_chunk_chars and len(proposed_chunk) <= hard_max_chunk_chars:
            current_chunk = proposed_chunk
            current_chunk_emb = model.encode([current_chunk])[0]
            continue

        # If it becomes too large, close current chunk
        if len(proposed_chunk) > hard_max_chunk_chars:
            chunks.extend(hard_split_large_text(current_chunk, hard_max_chunk_chars))
            current_chunk = next_unit
            current_chunk_emb = next_emb
            continue

        # If still below target size, bias toward merging
        if len(current_chunk) < target_chunk_chars:
            current_chunk = proposed_chunk
            current_chunk_emb = model.encode([current_chunk])[0]
            continue

        # Main semantic decision
        if sim >= similarity_threshold:
            current_chunk = proposed_chunk
            current_chunk_emb = model.encode([current_chunk])[0]
        else:
            chunks.extend(hard_split_large_text(current_chunk, hard_max_chunk_chars))
            current_chunk = next_unit
            current_chunk_emb = next_emb

    if current_chunk.strip():
        chunks.extend(hard_split_large_text(current_chunk, hard_max_chunk_chars))

    return [c.strip() for c in chunks if c.strip()]


if __name__ == "__main__":
    from importlib import import_module

    _loader = import_module("02_load_source_documents")

    print(f"[Upsert/03] Loading chunking model '{CHUNK_MODEL_NAME}' ...")
    model = SentenceTransformer(CHUNK_MODEL_NAME)

    sample_doc = _loader.load_raw_documents()[0]
    print(f"[Upsert/03] Chunking sample document '{sample_doc['id']}' "
          f"({len(sample_doc['text'])} chars) ...")

    chunks = semantic_chunk_text_embedding(sample_doc["text"], model=model)

    print(f"[Upsert/03] Split '{sample_doc['id']}' into {len(chunks)} chunk(s):\n")
    for i, chunk in enumerate(chunks):
        print(f"--- chunk {i} ({len(chunk)} chars) ---")
        print(chunk)
        print()
