import asyncio
import sys
import os
import aiohttp

# Add the parent directory to sys.path so we can import horizon modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from horizon.fetcher import fetch_hackernews

async def main():
    # Use a ClientSession and Semaphore as required by the fetcher functions
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(10)
        
        print("Fetching HackerNews articles...")
        articles = await fetch_hackernews(session, sem)
        
        print(f"Fetched {len(articles)} articles.")
        
        if articles:
            print("\nFirst article details:")
            print(f"Title:  {articles[0].title}")
            print(f"URL:    {articles[0].url}")
            print(f"Source: {articles[0].source}")
        else:
            print("No articles were fetched. Check your network or API status.")

if __name__ == "__main__":
    asyncio.run(main())