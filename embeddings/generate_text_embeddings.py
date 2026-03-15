import json
import os
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

METADATA_PATH = "data/raw/nature_metadata.json"
OUTPUT_PATH = "data/embeddings/text_embeddings.npy"
MODEL_NAME = "openai/clip-vit-base-patch32"


def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Missing {METADATA_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Metadata JSON must be a list")

    texts = []

    for item in data:
        text = item.get("description") or item.get("alt_description")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    if len(texts) == 0:
        raise RuntimeError("No valid text found in metadata")

    print(f"Cleaned text samples: {len(texts)}")

    batch_size = 32
    embeddings = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = processor(
                text=batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            embeddings.append(feats.cpu().numpy())

    text_embeddings = np.vstack(embeddings)
    np.save(OUTPUT_PATH, text_embeddings)

    print(f"✅ Saved text embeddings → {OUTPUT_PATH}")
    print(f"Shape: {text_embeddings.shape}")


if __name__ == "__main__":
    main()
