import unittest
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from horizon.models import Article
from horizon.scorer import build_profile_from_string, score_articles

class TestProfileScoring(unittest.TestCase):
    def test_profile_scoring(self) -> None:
        # Profile interest
        interest = "Python backend development"
        profile_vec = build_profile_from_string(interest)
        
        # Test articles
        titles = [
            "FastAPI tutorial",
            "Rust ownership explained",
            "Latest football match",
            "Django authentication",
            "Best pizza recipes"
        ]
        
        articles = [
            Article(title=title, url=f"http://example.com/{i}", source="test")
            for i, title in enumerate(titles)
        ]
        
        # Score articles against the profile
        ranked = score_articles(articles, profile_vec, top_n=len(articles))
        
        # Print results for manual inspection/visualization
        print("\n" + "=" * 50)
        print(f"Scored Articles for Profile: '{interest}'")
        print("=" * 50)
        for i, art in enumerate(ranked, 1):
            print(f"{i}. [{art.score:.4f}] {art.title}")
        print("=" * 50 + "\n")
        
        # Assertions
        # 1. FastAPI tutorial and Django authentication are both Python backend development topics.
        # They should be scored significantly higher than all other topics.
        top_titles = {ranked[0].title, ranked[1].title}
        self.assertIn("FastAPI tutorial", top_titles)
        self.assertIn("Django authentication", top_titles)
        
        # Check that their scores are > 0.2
        self.assertGreater(ranked[0].score, 0.2)
        self.assertGreater(ranked[1].score, 0.2)
        
        # 2. All other topics are not direct Python backend development topics
        # and should score significantly lower (< 0.1).
        for art in ranked[2:]:
            self.assertLess(art.score, 0.1)

if __name__ == "__main__":
    unittest.main()
