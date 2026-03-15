import json
import re
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/nature_metadata.json")
OUTPUT_PATH = Path("data/processed/captions.json")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def extract_caption(item: dict) -> str:
    # Prefer description, fallback to alt_description
    caption = item.get("description") or item.get("alt_description")

    if not caption:
        caption = "image with no description"

    return clean_text(caption)


def main():
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []

    for item in data:
        processed.append({
            "id": item["id"],
            "caption": extract_caption(item)
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)

    print(f"Saved {len(processed)} captions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
