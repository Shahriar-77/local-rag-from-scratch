
from typing import List, Dict
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


# -----------------------------
# Models
# -----------------------------

_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Utilities
# -----------------------------

def cosine_similarity(a, b):
    return (a @ b) / (norm(a) * norm(b))


def lexical_score(text: str, query: str) -> float:
    """
    Simple term overlap score.
    Intentionally naive but deterministic.
    """
    text_tokens = set(text.lower().split())
    query_tokens = set(query.lower().split())
    return len(text_tokens & query_tokens)


def min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores

    min_s, max_s = min(scores), max(scores)
    if min_s == max_s:
        return [1.0] * len(scores)

    return [(s - min_s) / (max_s - min_s) for s in scores]


# -----------------------------
# Embeddings
# -----------------------------

def embed_chunks(chunks: List[Dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    return _model.encode(texts, show_progress_bar=True)


# -----------------------------
# Simple Retrieval
# -----------------------------



def search_corpus(query: str, corpus: list[dict], top_k: int = 5) -> list[dict]:
    scored = []

    for chunk in corpus:
        score = lexical_score(chunk["text"], query)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

# -----------------------------
# Semantic Retrieval
# -----------------------------


def semantic_search(query: str, chunks: list[dict], embeddings: np.ndarray, top_k: int = 5):
    query_emb = model.encode(query)

    scores = []
    for emb, chunk in zip(embeddings, chunks):
        score = cosine_similarity(query_emb, emb)
        scores.append((score, chunk))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scores[:top_k]]


# -----------------------------
# Hybrid Retrieval
# -----------------------------

def hybrid_search(
    query: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    alpha: float = 0.6,
    beta: float = 0.4,
    top_k: int = 5
):
    # Lexical
    lexical_scores = [lexical_score(c["text"], query) for c in chunks]
    lexical_norm = min_max_normalize(lexical_scores)

    # Semantic
    query_emb = _model.encode(query)
    semantic_scores = [
        cosine_similarity(query_emb, emb)
        for emb in embeddings
    ]
    semantic_norm = min_max_normalize(semantic_scores)

    # Combine
    combined = []
    for i, chunk in enumerate(chunks):
        score = alpha * semantic_norm[i] + beta * lexical_norm[i]
        combined.append((score, chunk))

    combined.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in combined[:top_k]]
