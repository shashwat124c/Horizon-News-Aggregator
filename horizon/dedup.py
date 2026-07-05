import numpy as np
from .models import Article

from .models import Article
from .redis_client import is_url_seen, mark_url_seen

def dedup_by_url(articles: list[Article], ignore_seen: bool = False) -> list[Article]:
    """
    Drop articles whose URL we've seen in any previous run.
    New URLs are marked as seen so future runs skip them too.
    """
    seen_in_batch = set()
    fresh = []
    for article in articles:
        if article.url in seen_in_batch:
            continue
        seen_in_batch.add(article.url)
        
        if not ignore_seen:
            if is_url_seen(article.url):
                continue
            mark_url_seen(article.url)
            
        fresh.append(article)
    return fresh

def dedup_by_similarity(
    articles: list[Article],
    threshold: float = 0.92,
) -> list[Article]:
    """
    Drop articles whose title embedding is near-identical to one
    already kept. Articles must already have .embedding set
    (i.e. this runs after scoring, not before).

    When two articles collide, the higher-scored one survives.
    """
    # process highest-scored first, so duplicates lose to the better-ranked copy
    sorted_articles = sorted(articles, key=lambda a: a.score, reverse=True)

    kept: list[Article] = []
    kept_embeddings: list[np.ndarray] = []

    for article in sorted_articles:
        is_duplicate = False
        for emb in kept_embeddings:
            similarity = float(np.dot(article.embedding, emb))
            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(article)
            kept_embeddings.append(article.embedding)

    return kept