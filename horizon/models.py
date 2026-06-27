"""
Shared data models for Horizon.
A single Article dataclass is the unit of currency through the whole pipeline:
  fetch → dedup → score → render → deliver
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    """One item fetched from any source."""

    title: str
    url: str
    source: str                        # e.g. "hackernews", "arxiv", "lobsters"
    published_at: Optional[datetime] = None
    summary: str = ""                  # short excerpt from the feed (not AI-generated yet)
    score: float = 0.0                 # cosine similarity vs. interest profile (filled by scorer)
    embedding: Optional[object] = None # numpy array — filled by scorer, not stored long-term

    def __repr__(self) -> str:
        score_str = f"{self.score:.3f}" if self.score else "unscored"
        return f"Article({score_str}, [{self.source}] {self.title[:60]})"