import os
import redis
from datetime import datetime

def get_redis() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.from_url(url, decode_responses=True)

def is_digest_sent_today() -> bool:
    r = get_redis()
    key = datetime.now().strftime("digest:%d-%m-%Y")
    return r.exists(key) == 1

def mark_digest_sent():
    r = get_redis()
    key = datetime.now().strftime("digest:%d-%m-%Y")
    r.set(key, "sent", ex=90000)  # 25 hours in seconds

BLOOM_KEY = "horizon:seen_urls"

def is_url_seen(url: str) -> bool:
    r = get_redis()
    # BF.EXISTS returns 1 if probably seen, 0 if definitely not seen
    return r.execute_command("BF.EXISTS", BLOOM_KEY, url) == 1

def mark_url_seen(url: str):
    r = get_redis()
    # BF.ADD creates the filter automatically on first call
    r.execute_command("BF.ADD", BLOOM_KEY, url)