import faiss
import numpy as np
import json
import torch
from transformers import CLIPProcessor, CLIPModel

INDEX_PATH = "data/embeddings/image.index"
IMAGE_EMB_PATH = "data/embeddings/image_embeddings.npy"
METADATA_PATH = "data/raw/nature_metadata.json"

device = "cuda" if torch.cuda.is_available() else "cpu"


class MultimodalSearchEngine:
    def __init__(self):
        # Load FAISS index
        self.index = faiss.read_index(INDEX_PATH)

        # Load embeddings
        self.image_embeddings = np.load(IMAGE_EMB_PATH)

        # Load metadata
        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

        # Load CLIP
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device)

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def search(self, query: str, top_k: int = 5):
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)

        text_emb = text_emb.cpu().numpy()
        faiss.normalize_L2(text_emb)

        scores, indices = self.index.search(text_emb, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            img = self.metadata[idx]
            results.append({
                "id": img["id"],
                "description": img["alt_description"],
                "image_url": img["urls"]["small"],
                "score": float(score)
            })

        return results


if __name__ == "__main__":
    engine = MultimodalSearchEngine()
    results = engine.search("sunset mountains", top_k=5)

    for r in results:
        print(r)
