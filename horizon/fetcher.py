"""
Async feed fetcher — Phase 1.

Fetches 5 sources concurrently using aiohttp + asyncio. Each source
is a plain async function that returns a list of Articles. The public
entrypoint `fetch_all()` runs them all in parallel and merges the results.

Adding a new source in Phase 5 is one function + one line in SOURCES.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Coroutine

import aiohttp
import feedparser

from .models import Article

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency guard — never hammer a single host with unbounded parallelism
# ---------------------------------------------------------------------------
_SEMAPHORE_LIMIT = 10  # max simultaneous open HTTP connections


# ---------------------------------------------------------------------------
# Individual source fetchers
# ---------------------------------------------------------------------------

async def _fetch_rss(
    session: aiohttp.ClientSession,
    url: str,
    source_name: str,
    sem: asyncio.Semaphore,
    limit: int = 30,
) -> list[Article]:
    """Generic RSS/Atom fetcher. Used by most sources."""
    async with sem: # Grab a semaphore
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status() # Did the request succeed?
                raw = await resp.text() # Safe to use the response
        except Exception as exc:
            logger.warning("Failed to fetch %s (%s): %s", source_name, url, exc)
            return []

    feed = feedparser.parse(raw)
    articles: list[Article] = []

    for entry in feed.entries[:limit]:
        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

        summary = ""
        if hasattr(entry, "summary"):
            # strip HTML tags crudely — good enough for a 2-sentence blurb
            import re
            summary = re.sub(r"<[^>]+>", "", entry.summary)[:400]

        articles.append(Article(
            title=entry.get("title", "").strip(),
            url=entry.get("link", "").strip(),
            source=source_name,
            published_at=published,
            summary=summary,
        ))

    logger.debug("Fetched %d items from %s", len(articles), source_name)
    return articles


async def fetch_hackernews(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore
) -> list[Article]:
    """HN top stories via the official Firebase API — returns titles + links."""
    async with sem:
        try:
            async with session.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                story_ids: list[int] = (await resp.json())[:40] 
                """
                Tells Python to wait for the response body to download completely and parse it as a JSON list
                """
        except Exception as exc:
            logger.warning("HN top stories failed: %s", exc)
            return []

    async def _fetch_item(story_id: int) -> Article | None:
        async with sem:
            try:
                async with session.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    item = await resp.json()
            except Exception:
                return None

        if not item or item.get("type") != "story":
            return None
        url = item.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
        return Article(
            title=item.get("title", "").strip(),
            url=url,
            source="hackernews",
            published_at=datetime.fromtimestamp(
                item.get("time", 0), tz=timezone.utc
            ),
            summary=f"HN score: {item.get('score', 0)} | comments: {item.get('descendants', 0)}",
        )

    tasks = [_fetch_item(sid) for sid in story_ids]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def fetch_lobsters(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore
) -> list[Article]:
    return await _fetch_rss(
        session,
        "https://lobste.rs/rss",
        source_name="lobsters",
        sem=sem,
        limit=25,
    )


async def fetch_arxiv_cs(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore
) -> list[Article]:
    """
    arXiv CS feed — covers cs.AI, cs.DC (distributed computing), cs.PL (prog languages).
    Each entry has a proper abstract as its summary.
    """
    # Combined query: recent papers across three relevant CS categories
    url = (
        "https://export.arxiv.org/rss/cs.AI+cs.DC+cs.PL"
    )
    return await _fetch_rss(session, url, source_name="arxiv", sem=sem, limit=20)


async def fetch_devto(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore
) -> list[Article]:
    """dev.to top articles from the last week via their public API."""
    async with sem:
        try:
            async with session.get(
                "https://dev.to/api/articles?per_page=30&top=7",
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"Accept": "application/json"},
            ) as resp:
                resp.raise_for_status()
                items = await resp.json()
        except Exception as exc:
            logger.warning("dev.to fetch failed: %s", exc)
            return []

    articles = []
    for item in items:
        articles.append(Article(
            title=item.get("title", "").strip(),
            url=item.get("url", "").strip(),
            source="devto",
            summary=item.get("description", "")[:400],
        ))
    return articles


async def fetch_reddit_rust(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore
) -> list[Article]:
    """r/rust hot posts — a good proxy for what the Rust community is talking about."""
    return await _fetch_rss(
        session,
        "https://www.reddit.com/r/rust/hot/.rss?limit=25",
        source_name="reddit_rust",
        sem=sem,
        limit=25,
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

# Registry — add new fetchers here in Phase 5
_SOURCE_FETCHERS: list[
    Callable[[aiohttp.ClientSession, asyncio.Semaphore], Coroutine]
] = [
    fetch_hackernews,
    fetch_lobsters,
    fetch_arxiv_cs,
    fetch_devto,
    fetch_reddit_rust,
]


async def fetch_all() -> list[Article]:
    """
    Fetch all sources concurrently.

    Returns a flat list of Articles with no dedup and no scoring —
    those are handled by downstream pipeline stages.
    """
    sem = asyncio.Semaphore(_SEMAPHORE_LIMIT)
    headers = {
        "User-Agent": "Horizon-digest/0.1 (personal news aggregator; [EMAIL_ADDRESS])"
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [fetcher(session, sem) for fetcher in _SOURCE_FETCHERS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles: list[Article] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Source %d raised: %s", i, result)
        else:
            all_articles.extend(result)

    logger.info("Fetched %d total articles from %d sources", len(all_articles), len(_SOURCE_FETCHERS))
    return all_articles