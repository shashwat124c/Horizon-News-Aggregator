import os
import requests
from .models import Article

TELEGRAM_API = "https://api.telegram.org"

def send_telegram(articles: list[Article]) -> None:
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
    lines = ["🌅 <b>Today's Horizon Digest</b>\n"]

    for i, article in enumerate(articles, 1):
        # clean up title — strip any HTML characters that would break Telegram's parser
        title = article.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        summary = article.summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines.append(
            f"{i}. <b>{title}</b>\n"
            f"   {summary[:120]}...\n"
            f"   <a href=\"{article.url}\">{article.source}</a>  •  score: {article.score:.2f}\n"
        )

    return "\n".join(lines)