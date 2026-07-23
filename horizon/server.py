import logging
import threading
from flask import Flask, redirect
from .database import load_article, load_profile, save_profile
from .scorer import embed_text

logger = logging.getLogger(__name__)
app = Flask(__name__)

@app.route("/click/<article_id>")
def handle_click(article_id: str):
    # immediately redirect the user — don't make them wait for embedding
    article = load_article(article_id)
    if article is None:
        return "Not found", 404

    # do the profile update in a background thread so redirect is instant
    threading.Thread(
        target=_update_profile,
        args=(article["url"], article["embedding"]),
        daemon=True,
    ).start()

    return redirect(article["url"])


def _update_profile(url: str, stored_embedding_bytes):
    try:
        import trafilatura
        import numpy as np
        from .database import get_connection

        # fetch and extract clean article text
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)

        if not text:
            logger.warning("Could not extract text from %s", url)
            return

        # embed the article text
        article_embedding = embed_text(text[:2000])  # cap at 2000 chars, enough context

        # load current profile
        current_data = load_profile()
        if current_data is None:
            return
        current_profile = current_data["embedding"]

        # load original profile (anchor) — stored separately
        original_data = load_profile(name="original")
        original_profile = original_data["embedding"] if original_data is not None else current_profile.copy()

        # weighted average nudge
        new_profile = 0.85 * current_profile + 0.15 * article_embedding

        # normalize back to unit length — cosine similarity assumes unit vectors
        new_profile = new_profile / np.linalg.norm(new_profile)

        # anchor blend — pull slightly toward original to prevent drift
        final_profile = 0.90 * new_profile + 0.10 * original_profile
        final_profile = final_profile / np.linalg.norm(final_profile)

        # save updated profile
        conn = get_connection()
        conn.execute("""
            UPDATE profiles SET embedding = ?, updated_at = CURRENT_TIMESTAMP
            WHERE name = 'default'
        """, (final_profile.tobytes(),))
        conn.commit()
        conn.close()

        logger.info("Profile updated after click on %s", url)

    except Exception as e:
        logger.error("Profile update failed: %s", e)