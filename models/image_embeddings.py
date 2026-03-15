import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
from transformers import CLIPProcessor, CLIPModel


IMAGE_DIR = Path("data/images")
OUTPUT_PATH = Path("data/embeddings/image_embeddings.json")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    embeddings = []

    image_files = list(IMAGE_DIR.glob("*.jpg"))
    print(f"Total images found: {len(image_files)}")

    for img_path in tqdm(image_files):
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embeddings.append({
            "id": img_path.stem,
            "embedding": image_features.cpu().numpy()[0].tolist()
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(embeddings, f)

    print(f"Saved image embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
