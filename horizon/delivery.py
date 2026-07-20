import os
import requests
from .models import Article
from .database import save_article, init_db

TELEGRAM_API = "https://api.telegram.org"
REDIRECT_BASE = os.getenv("REDIRECT_BASE", "http://localhost:5000")


def send_telegram(articles: list[Article]) -> None:
    init_db()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")

    message = _format_message(articles)

    response = requests.post(
        f"{TELEGRAM_API}/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
    )

    if not response.ok:
        raise RuntimeError(f"Telegram delivery failed: {response.text}")


def _format_message(articles: list[Article]) -> str:
    lines = ["<b>Today's Horizon Digest</b>\n"]
    use_direct = os.getenv("USE_DIRECT_LINKS", "false").lower() == "true"

    for i, article in enumerate(articles, 1):
        article_id = save_article(article)
        target_url = article.url if use_direct else f"{REDIRECT_BASE}/click/{article_id}"

        title = article.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        summary = article.summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines.append(
            f"{i}. <a href=\"{target_url}\"><b>{title}</b></a>\n"
            f"   {summary[:120]}...\n"
            f"   <i>[{article.source}]</i>  •  score: {article.score:.2f}\n"
        )

    return "\n".join(lines)