import sqlite3
import os
import numpy as np
from datetime import datetime

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