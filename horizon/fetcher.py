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

FALLBACK_URLS: dict[str, str] = {
    "stripe": "https://stripe.com/blog/feed.xml",
    "shopify": "https://shopify.engineering/feed",
    "uber": "https://www.uber.com/blog/engineering/rss/",
    "acmqueue": "https://queue.acm.org/rss/feeds/queuecontent.xml",
}


async def _fetch_rss(
    session: aiohttp.ClientSession,
    url: str,
    source_name: str,
    sem: asyncio.Semaphore,
    limit: int = 30,
) -> list[Article]:
    """Generic RSS/Atom fetcher. Used by most sources."""
    headers = None
    if "reddit" in source_name:
        headers = {"User-Agent": "python:horizon.digest:v0.1 (by /u/horizon_news)"}
    elif source_name == "uber":
        headers = {"Accept": "*/*"}

    async with sem:
        raw = None
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    raw = await resp.text()
                elif source_name in FALLBACK_URLS:
                    fallback = FALLBACK_URLS[source_name]
                    async with session.get(fallback, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as f_resp:
                        f_resp.raise_for_status()
                        raw = await f_resp.text()
                else:
                    resp.raise_for_status()
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


# Additional RSS/Atom engineering blogs and tech news feeds
RSS_SOURCES: list[tuple[str, str]] = [
    ("https://blog.cloudflare.com/rss",          "cloudflare"),
    ("https://netflixtechblog.com/feed",          "netflix"),
    ("https://eng.uber.com/feed",                 "uber"),
    ("https://slack.engineering/rss",             "slack"),
    ("https://engineering.fb.com/feed",           "meta"),
    ("https://github.blog/engineering.atom",      "github"),
    ("https://shopify.engineering/rss",           "shopify"),
    ("https://stripe.com/blog/engineering.rss",   "stripe"),
    ("https://martinfowler.com/feed.atom",        "martinfowler"),
    ("https://brooker.co.za/blog/rss.xml",        "marc_brooker"),   # AWS distributed systems
    ("https://matklad.github.io/feed.xml",        "matklad"),        # Rust, compilers
    ("https://fasterthanli.me/index.xml",         "fasterthanli"),   # deep Rust posts
    ("https://export.arxiv.org/rss/cs.LG",       "arxiv_ml"),
    ("https://export.arxiv.org/rss/cs.CL",       "arxiv_nlp"),
    ("https://huggingface.co/blog/feed.xml",      "huggingface"),
    ("https://hnrss.org/frontpage?points=100",    "hackernews_top"),  # HN with 100+ points filter
    ("https://www.reddit.com/r/golang/hot/.rss",  "reddit_golang"),
    ("https://www.reddit.com/r/MachineLearning/hot/.rss", "reddit_ml"),
    ("https://www.reddit.com/r/programming/hot/.rss",     "reddit_prog"),
    ("https://arstechnica.com/feed",              "arstechnica"),
    ("https://feeds.feedburner.com/ACMQueue-LatestArticles", "acmqueue"),
]

# Registry — custom API/feed fetchers
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
        # Custom API & custom RSS fetchers
        tasks = [fetcher(session, sem) for fetcher in _SOURCE_FETCHERS]

        # Additional RSS feed sources
        for url, name in RSS_SOURCES:
            tasks.append(_fetch_rss(session, url, name, sem=sem))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles: list[Article] = []
    total_sources = len(_SOURCE_FETCHERS) + len(RSS_SOURCES)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Source %d raised: %s", i, result)
        else:
            all_articles.extend(result)

    logger.info("Fetched %d total articles from %d sources", len(all_articles), total_sources)
    return all_articles