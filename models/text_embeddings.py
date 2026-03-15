import torch
from pathlib import Path
import json
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


METADATA_PATH = Path("data/raw/nature_metadata.json")
OUTPUT_PATH = Path("data/embeddings/text_embeddings.json")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    with open(METADATA_PATH, "r") as f:
        data = json.load(f)

    print(f"Total items in metadata: {len(data)}")

    embeddings = []

    for item in tqdm(data):
        caption = (
            item.get("alt_description")
            or item.get("description")
            or ""
        )

        if not caption:
            continue

        inputs = processor(
            text=caption,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        embeddings.append({
            "id": item["id"],
            "caption": caption,
            "embedding": text_features.cpu().numpy()[0].tolist()
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(embeddings, f)

    print(f"Saved text embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
