import json
import os
from pathlib import Path

from ingestion.unsplash_client import UnsplashClient


DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_images(query: str, pages: int = 1, per_page: int = 10):
    client = UnsplashClient()
    all_results = []

    for page in range(1, pages + 1):
        print(f"Fetching page {page} for query='{query}'")

        response = client.search_photos(
            query=query,
            page=page,
            per_page=per_page
        )

        all_results.extend(response["results"])

    return all_results


def save_metadata(data, query: str):
    output_file = DATA_DIR / f"{query}_metadata.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved metadata to {output_file}")


if __name__ == "__main__":
    QUERY = "nature"
    PAGES = 1
    PER_PAGE = 10

    results = fetch_images(QUERY, PAGES, PER_PAGE)
    save_metadata(results, QUERY)
