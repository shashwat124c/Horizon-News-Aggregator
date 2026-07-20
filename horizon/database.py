import sqlite3
import os
import numpy as np
from datetime import datetime
import uuid

DB_PATH = os.getenv("HORIZON_DB_PATH", "horizon.db")

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets you access columns by name instead of index
    return conn

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            interests TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            url        TEXT NOT NULL,
            source     TEXT NOT NULL,
            embedding  BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_profile(name: str, embedding: np.ndarray, interests: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO profiles (name, embedding, interests)
        VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            embedding  = excluded.embedding,
            interests  = excluded.interests,
            updated_at = CURRENT_TIMESTAMP
    """, (name, embedding.tobytes(), interests))
    conn.commit()
    conn.close()

def load_profile(name: str = "default") -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT embedding, interests FROM profiles WHERE name = ?", (name,)
    ).fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "embedding": np.frombuffer(row["embedding"], dtype=np.float32).copy(),
        "interests": row["interests"]
    }

def save_article(article) -> str:
    """Persist an article and return its ID."""
    article_id = uuid.uuid4().hex[:8]  # short random string e.g. "a3f9c12b"
    conn = get_connection()
    conn.execute("""
        INSERT OR IGNORE INTO articles (id, title, url, source, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (
        article_id,
        article.title,
        article.url,
        article.source,
        article.embedding.tobytes() if article.embedding is not None else None,
    ))
    conn.commit()
    conn.close()
    return article_id

def load_article(article_id: str):
    """Fetch a stored article by ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM articles WHERE id = ?", (article_id,)
    ).fetchone()
    conn.close()
    return row