# embeddings/generate_embeddings.py

import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

IMAGES_DIR = "data/raw/images"
OUTPUT_PATH = "data/embeddings/image_embeddings.npy"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    embeddings = []

    image_files = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        raise RuntimeError("No images found in data/raw/images")

    for img_name in tqdm(image_files, desc="Encoding images"):
        img_path = os.path.join(IMAGES_DIR, img_name)
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        features = features.cpu().numpy()
        embeddings.append(features[0])

    embeddings = np.array(embeddings)

    os.makedirs("data/embeddings", exist_ok=True)
    np.save(OUTPUT_PATH, embeddings)

    print("Saved image embeddings:", embeddings.shape)


if __name__ == "__main__":
    main()
