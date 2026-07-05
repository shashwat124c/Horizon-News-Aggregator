"""
Horizon CLI — Phase 1.

Entry point: `horizon run` (or `python -m horizon.cli run`)

Usage:
    horizon run                         # use HORIZON_INTERESTS from .env
    horizon run --interests "Rust, AI"  # override inline
    horizon run --top 20                # return top 20 instead of 10
    horizon run --verbose               # show DEBUG logs
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from .redis_client import is_digest_sent_today, mark_digest_sent
import click
from dotenv import load_dotenv
from .dedup import dedup_by_url, dedup_by_similarity

load_dotenv()

DEFAULT_INTERESTS = os.getenv(
    "HORIZON_INTERESTS",
    "Rust, distributed systems, AI infrastructure, compiler design, systems programming",
)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # silence noisy third-party loggers unless verbose
    if not verbose:
        for noisy in ("urllib3", "httpx", "asyncio", "sentence_transformers", "transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


@click.group()
def cli() -> None:
    """Horizon — personalized tech news digest."""
    pass


@cli.command()
@click.option(
    "--interests", "-i",
    default=DEFAULT_INTERESTS,
    show_default=True,
    help="Plain-English interest description (overrides HORIZON_INTERESTS in .env).",
)
@click.option("--top", "-n", default=10, show_default=True, help="Number of articles to show.")
@click.option("--verbose", "-v", is_flag=True, help="Show DEBUG logs.")
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Ignore the check for whether a digest has already been sent today and run anyway.",
)
@click.option(
    "--no-dedup",
    is_flag=True,
    help="Ignore the check for previously seen URLs (do not query or update Redis seen history).",
)
def run(interests: str, top: int, verbose: bool, force: bool, no_dedup: bool) -> None:
    """
    Fetch, score, and print today's digest to the console.

    Phase 1 entry point — no Redis, no Telegram, no DB. Just the
    fetch → embed → rank → print pipeline.
    """
    _configure_logging(verbose)
    logger = logging.getLogger(__name__)

    if not force and is_digest_sent_today():
        click.echo("Digest already sent today — skipping.")
        return

    # Lazy imports here so CLI --help is instant even before deps are installed
    from .fetcher import fetch_all
    from .scorer import build_profile_from_string, score_articles

    click.echo(f"\n{'─' * 60}")
    click.echo(f"  🌅  Horizon  —  {datetime.now().strftime('%A, %d %B %Y')}")
    click.echo(f"{'─' * 60}")
    click.echo(f"  Interests: {interests}")
    click.echo(f"{'─' * 60}\n")

    # Step 1: build profile embedding
    click.echo("⏳ Building interest profile...")
    profile_vec = build_profile_from_string(interests)
    click.echo(f"   Profile vector shape: {profile_vec.shape}, norm: {float((profile_vec**2).sum()**0.5):.4f}\n")

    # Step 2: fetch all sources concurrently
    click.echo("⏳ Fetching feeds...")
    articles = asyncio.run(fetch_all())
    click.echo(f"   Fetched {len(articles)} articles total\n")

    if not articles:
        click.echo("❌ No articles fetched — check your network connection.", err=True)
        sys.exit(1)

    articles = dedup_by_url(articles, ignore_seen=no_dedup)
    click.echo(f"   After URL dedup: {len(articles)} articles\n")

    # Step 3: score and rank
    click.echo(f"⏳ Scoring against your profile (top {top})...")
    ranked = score_articles(articles, profile_vec, top_n=top)
    
    _print_digest(ranked)
    
    mark_digest_sent()

def _print_digest(articles: list) -> None:
    """Render the digest as a terminal-friendly table."""
    import textwrap

    click.echo(f"\n{'═' * 60}")
    click.echo("  📰  TODAY'S DIGEST")
    click.echo(f"{'═' * 60}\n")

    source_colors = {
        "hackernews":  "yellow",
        "lobsters":    "cyan",
        "arxiv":       "blue",
        "devto":       "green",
        "reddit_rust": "red",
    }

    for i, article in enumerate(articles, 1):
        color = source_colors.get(article.source, "white")

        # Header line: rank, score, source badge
        source_badge = click.style(f"[{article.source}]", fg=color, bold=True)
        score_badge  = click.style(f"{article.score:.3f}", fg="bright_white", bold=True)
        click.echo(f"  {i:>2}.  {score_badge}  {source_badge}")

        # Title
        title_lines = textwrap.wrap(article.title, width=54)
        for j, line in enumerate(title_lines):
            prefix = "       " if j > 0 else "       "
            click.echo(f"{prefix}{click.style(line, bold=(j == 0))}")

        # Summary (if any) — wrapped, dimmed
        if article.summary:
            summary = article.summary.replace("\n", " ").strip()
            for line in textwrap.wrap(summary, width=52)[:2]:  # max 2 lines
                click.echo(f"       {click.style(line, dim=True)}")

        # URL
        click.echo(f"       {click.style(article.url, fg='bright_blue', underline=True)}")
        click.echo()

    click.echo(f"{'─' * 60}")
    click.echo(f"  {len(articles)} articles  •  Phase 1  •  Run `horizon run --help` for options")
    click.echo(f"{'─' * 60}\n")


if __name__ == "__main__":
    cli()