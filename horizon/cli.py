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
from .delivery import send_telegram
from .database import init_db, save_profile, load_profile
from .scorer import build_profile_from_string
from .scheduler import start_scheduler
from .server import app as flask_app
import threading


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

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


# ---------------------------------------------------------------------------
# Topic probes used to interpret the profile vector in human-readable terms.
# Add or remove topics freely — they're only used for display.
# ---------------------------------------------------------------------------
_PROBE_TOPICS = [
    "Rust programming",
    "distributed systems",
    "AI infrastructure",
    "compiler design",
    "systems programming",
    "machine learning",
    "Python backend",
    "web development",
    "databases",
    "DevOps / Kubernetes",
    "security / cryptography",
    "open source",
    "startups / business",
    "mathematics / algorithms",
]


def _render_profile(clear: bool = False) -> None:
    """Load both profiles and print a drift summary to the terminal."""
    from .database import load_profile
    from .scorer import embed_texts, cosine_similarity

    current  = load_profile("default")
    original = load_profile("original")

    if current is None or original is None:
        click.echo("No profile found. Run `horizon init` first.")
        return

    cur_vec  = current["embedding"]
    orig_vec = original["embedding"]

    drift = 1.0 - cosine_similarity(cur_vec, orig_vec)   # 0 = identical, 2 = opposite

    if clear:
        click.clear()

    click.echo(f"\n{'═' * 58}")
    click.echo(f"  🧭  Horizon Profile  —  {datetime.now().strftime('%H:%M:%S')}")
    click.echo(f"{'═' * 58}")
    click.echo(f"  Original interests : {original['interests'][:54]}")
    click.echo(f"  Profile drift      : {drift:.4f}  {'(no clicks yet)' if drift < 0.001 else ''}")
    click.echo(f"{'─' * 58}\n")

    # Score every probe against both vectors and show the delta
    probe_vecs = embed_texts(_PROBE_TOPICS)   # shape (N, 384)
    orig_scores = (probe_vecs @ orig_vec).tolist()
    cur_scores  = (probe_vecs @ cur_vec).tolist()

    rows = sorted(
        zip(_PROBE_TOPICS, orig_scores, cur_scores),
        key=lambda r: r[2],
        reverse=True,
    )

    click.echo(f"  {'Topic':<30}  {'Original':>8}  {'Current':>8}  {'Delta':>8}")
    click.echo(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*8}")
    for topic, orig_s, cur_s in rows:
        delta = cur_s - orig_s
        delta_str = f"{delta:+.4f}"
        color = "green" if delta > 0.002 else ("red" if delta < -0.002 else "white")
        click.echo(
            f"  {topic:<30}  {orig_s:>8.4f}  {cur_s:>8.4f}  "
            + click.style(f"{delta_str:>8}", fg=color)
        )

    click.echo(f"\n{'═' * 58}\n")


@cli.command()
@click.option("--watch", "-w", is_flag=True, help="Poll every N seconds and refresh.")
@click.option("--interval", "-n", default=10, show_default=True, help="Refresh interval in seconds (with --watch).")
def profile(watch: bool, interval: int) -> None:
    """Show how your interest profile has drifted after clicks."""
    import time
    if watch:
        click.echo(f"Watching profile (refreshing every {interval}s — Ctrl+C to stop)...")
        try:
            while True:
                _render_profile(clear=True)
                time.sleep(interval)
        except KeyboardInterrupt:
            click.echo("Stopped.")
    else:
        _render_profile()



@cli.command()
@click.option(
    "--interests", "-i",
    default=DEFAULT_INTERESTS,
    show_default=True,
    help="Plain-English interest description (overrides HORIZON_INTERESTS in .env).",
)
@click.option("--name", default="default", help="Profile name.")
def init(interests: str, name: str):
    """Create or overwrite your interest profile."""
    init_db()

    click.echo(f"Building profile from: {interests!r}")
    embedding = build_profile_from_string(interests)

    save_profile("default", embedding, interests)
    save_profile("original", embedding, interests)

    click.echo(f"Profile '{name}' saved to database. Shape: {embedding.shape}")

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
    fetch -> embed -> rank -> print pipeline.
    """
    _configure_logging(verbose)
    logger = logging.getLogger(__name__)

    if not force and is_digest_sent_today():
        click.echo("Digest already sent today — skipping.")
        return

    # Lazy imports here so CLI --help is instant even before deps are installed
    from .fetcher import fetch_all
    from .scorer import build_profile_from_string, score_articles

    # Step 1: build profile embedding
    if interests != DEFAULT_INTERESTS:
        click.echo("⏳ Building interest profile from override...")
        profile_vec = build_profile_from_string(interests)
    else:
        click.echo("⏳ Loading interest profile...")
        profile = load_profile()
        if profile is None:
            click.echo("No profile found. Run `horizon init --interests '...'` first.")
            return
        profile_vec = profile["embedding"]
        interests = profile["interests"]

    click.echo(f"\n{'─' * 60}")
    click.echo(f"  🌅  Horizon  —  {datetime.now().strftime('%A, %d %B %Y')}")
    click.echo(f"{'─' * 60}")
    click.echo(f"  Interests: {interests}")
    click.echo(f"{'─' * 60}\n")

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
    scored = score_articles(articles, profile_vec, top_n=top)

    # now dedup against full embeddings
    deduped = dedup_by_similarity(scored)
    click.echo(f"   After similarity dedup: {len(deduped)} articles\n")
    ranked = deduped[:top]
    _print_digest(ranked)

    click.echo("⏳ Sending to Telegram...")
    send_telegram(ranked)
    click.echo("✅ Digest sent.\n")
    
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

@cli.command()
@click.option("--hour", default=7, help="Hour to run the digest (24h format).")
@click.option("--minute", default=0, help="Minute to run the digest.")
@click.option("--port", default=5000, help="Port for the redirect server.")
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
def serve(hour: int, minute: int, port: int, force: bool, no_dedup: bool):
    """Start the redirect server and background scheduler."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    click.echo(f"Starting background scheduler (scheduled for {hour:02d}:{minute:02d} daily)...")
    sched = start_scheduler(hour=hour, minute=minute, force=force, no_dedup=no_dedup)

    click.echo(f"Redirect server running on http://localhost:{port}")
    try:
        flask_app.run(port=port, debug=False)
    except (KeyboardInterrupt, SystemExit):
        click.echo("\nStopping background scheduler...")
        sched.shutdown()

if __name__ == "__main__":
    cli()