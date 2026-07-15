import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from datetime import datetime

logger = logging.getLogger(__name__)

def run_job(force: bool = False, no_dedup: bool = False):
    """The actual job APScheduler calls every morning."""
    import asyncio
    from .fetcher import fetch_all
    from .scorer import score_articles
    from .database import load_profile
    from .dedup import dedup_by_url, dedup_by_similarity
    from .delivery import send_telegram
    from .redis_client import is_digest_sent_today, mark_digest_sent

    logger.info("Job started")

    if not force and is_digest_sent_today():
        logger.info("Already sent today — skipping")
        return

    profile = load_profile()
    if profile is None:
        logger.error("No profile found — run `horizon init` first")
        return
    profile_vec = profile["embedding"]

    articles = asyncio.run(fetch_all())
    articles = dedup_by_url(articles, ignore_seen=no_dedup)
    scored = score_articles(articles, profile_vec, top_n=None)
    deduped = dedup_by_similarity(scored)
    ranked = deduped[:10]

    send_telegram(ranked)
    mark_digest_sent()

    logger.info("Digest sent successfully")


def start_scheduler(hour: int = 7, minute: int = 0, force: bool = False, no_dedup: bool = False):
    jobstore = SQLAlchemyJobStore(url="sqlite:///horizon.db")

    scheduler = BlockingScheduler(jobstores={"default": jobstore})

    scheduler.add_job(
        run_job,
        trigger="cron",
        # hour=hour,
        # minute=minute,
        hour=datetime.now().hour,
        minute=datetime.now().minute + 1,
        id="daily_digest",
        replace_existing=True,
        kwargs={"force": force, "no_dedup": no_dedup},
    )

    logger.info(f"Scheduler started — digest will run at {hour:02d}:{minute:02d} daily (force={force}, no_dedup={no_dedup})")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped")