import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from horizon.database import load_profile, init_db
from horizon.scorer import cosine_similarity, embed_text

def inspect():
    init_db()
    default = load_profile("default")
    original = load_profile("original")

    if default is None or original is None:
        print("❌ No profile found in horizon.db! Run `horizon init` first.")
        return

    def_vec = default["embedding"]
    orig_vec = original["embedding"]

    similarity = cosine_similarity(def_vec, orig_vec)

    print("=" * 60)
    print(" 📊 HORIZON PROFILE INSPECTOR")
    print("=" * 60)
    print(f"Original Initial Interests : {original['interests']}")
    print(f"Current Profile Interests  : {default['interests']}")
    print(f"Drift Similarity (Current vs. Original Anchor): {similarity:.4f} / 1.0000")
    print("-" * 60)

    print("\n🔍 Scoring Profile against Target Topics:")
    sample_topics = [
        "Rust programming language",
        "Distributed systems and infrastructure",
        "AI and machine learning models",
        "Compiler design and low-level code",
        "Web development and HTML/CSS",
        "Cryptocurrency and Web3",
        "Python backend frameworks",
    ]

    for topic in sample_topics:
        topic_vec = embed_text(topic)
        orig_score = cosine_similarity(orig_vec, topic_vec)
        curr_score = cosine_similarity(def_vec, topic_vec)
        delta = curr_score - orig_score
        sign = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  • {topic:<42} : {curr_score:.4f} (original: {orig_score:.4f}, change: {sign})")

    print("=" * 60)

if __name__ == "__main__":
    inspect()
