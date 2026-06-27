"""
Interest scorer — Phase 1.

Embeds article titles and scores them against an interest profile using
cosine similarity. The profile in Phase 1 is a plain string; from Phase 3
onward it's a persisted vector that drifts via the feedback loop.

Design decisions:
- We embed titles only (not full text) in Phase 1 — fast, good enough.
  Full-text embedding happens in Phase 4 when a click-through is detected.
- Batch embedding: sentence-transformers is most efficient when encoding
  all titles in one call rather than one-at-a-time.
- The model (all-MiniLM-L6-v2) is 80MB, 384 dims, ~0.8s for 300 titles on CPU.
  It's downloaded once to ~/.cache/huggingface on first run.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import Article

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("Loading sentence-transformer model: %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def embed_text(text: str) -> np.ndarray:
    """Embed a single string. Returns a normalised 384-dim float32 vector."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.astype(np.float32)  # float 32 is the standard for many ml libs


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of strings. Returns shape (N, 384) float32.
    Batch encoding is ~10x faster than calling embed_text() in a loop.
    """
    model = _get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
        batch_size=64,
    )
    return vecs.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two normalised vectors.
    Since both are L2-normalised by sentence-transformers, this is just a dot product.
    Returns a float in [-1, 1]; higher = more similar.
    """
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def score_articles(
    articles: list[Article],
    profile_embedding: np.ndarray,
    top_n: int = 10,
) -> list[Article]:
    """
    Score and rank articles against a profile embedding vector.

    - Embeds all article titles in a single batch call.
    - Computes cosine similarity against the profile vector.
    - Attaches `.score` and `.embedding` to each article.
    - Returns the top_n highest-scoring articles, sorted descending.

    Args:
        articles:          Raw articles from the fetcher.
        profile_embedding: 384-dim normalised vector representing your interests.
        top_n:             How many to return (default 10 for the digest).

    Returns:
        List of up to top_n Articles with .score filled in, highest first.
    """
    if not articles:
        logger.warning("score_articles: received empty article list")
        return []

    titles = [a.title for a in articles]
    logger.info("Embedding %d article titles...", len(titles))
    embeddings = embed_texts(titles)  # shape (N, 384)

    # Cosine similarity: dot product of normalised vectors = (N,) array
    scores = embeddings @ profile_embedding  # matrix × vector

    for article, score, embedding in zip(articles, scores, embeddings):
        article.score = float(score)
        article.embedding = embedding

    ranked = sorted(articles, key=lambda a: a.score, reverse=True)

    logger.info(
        "Scored %d articles. Top score: %.3f | Median: %.3f | Bottom: %.3f",
        len(ranked),
        ranked[0].score,
        ranked[len(ranked) // 2].score,
        ranked[-1].score,
    )

    return ranked[:top_n]


def build_profile_from_string(interest_string: str) -> np.ndarray:
    """
    Turn a plain-English interest description into a 384-dim profile vector.

    Example:
        build_profile_from_string("Rust, distributed systems, AI infrastructure, not crypto")

    In Phase 3 this vector is saved to SQLite. In Phase 4 it drifts via
    click-through feedback. For now it's just embedded and used directly.
    """
    logger.info("Building profile from: %r", interest_string)
    return embed_text(interest_string)