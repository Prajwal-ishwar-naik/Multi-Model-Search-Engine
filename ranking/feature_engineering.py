import os
import json
import numpy as np
import pandas as pd

TEXT_EMB_PATH = "data/embeddings/text_embeddings.npy"
IMAGE_EMB_PATH = "data/embeddings/image_embeddings.npy"
METADATA_PATH = "data/raw/nature_metadata.json"

OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "ranking_features.csv")


def cosine_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


def main():
    print("📥 Loading embeddings...")

    text_emb = np.load(TEXT_EMB_PATH)
    image_emb = np.load(IMAGE_EMB_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise ValueError("Metadata must be a list")

    rows = []

    for i, item in enumerate(metadata):
        desc = item.get("description") or ""

        sim = cosine_sim(text_emb[i], image_emb[i])

        rows.append({
            "clip_similarity": sim,
            "text_length": len(desc),
            "image_brightness": 0.5,   # placeholder
            "label": 1 if sim > 0.25 else 0
        })

    df = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    df.to_csv(OUT_PATH, index=False)

    print(f"✅ Ranking features saved to {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
