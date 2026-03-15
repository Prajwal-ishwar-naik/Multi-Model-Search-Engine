import os
import json
import faiss
import torch
import joblib
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

# ============================================================
# Project root & paths
# ============================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INDEX_PATH = os.path.join(DATA_DIR, "index", "faiss.index")
METADATA_PATH = os.path.join(DATA_DIR, "raw", "nature_metadata.json")
RANKER_PATH = os.path.join(DATA_DIR, "models", "ranker.pkl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10


# ============================================================
# Multimodal Search Engine
# ============================================================
class MultimodalSearchEngine:
    def __init__(self):
        print("🚀 Loading models...")

        # ---------- FAISS ----------
        self.index = faiss.read_index(INDEX_PATH)

        # ---------- Metadata ----------
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS size ({self.index.ntotal}) != metadata size ({len(self.metadata)})"
            )

        # ---------- Ranker ----------
        self.ranker = joblib.load(RANKER_PATH)

        # ---------- CLIP ----------
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(DEVICE)

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        print("✅ Models loaded")
        print(f"📦 Indexed items: {self.index.ntotal}")

    # ============================================================
    # Encode text
    # ============================================================
    def encode_text(self, query: str):
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)

        emb = emb.cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    # ============================================================
    # Search + Re-rank
    # ============================================================
    def search(self, query: str, top_k: int = TOP_K):
        print(f"\n🔍 Query: {query}")

        query_emb = self.encode_text(query)
        distances, indices = self.index.search(query_emb, top_k)

        results = []

        for rank, idx in enumerate(indices[0]):
            idx = int(idx)
            item = self.metadata[idx]

            # ---------- Description ----------
            description = (
                item.get("description")
                or item.get("alt_description")
                or ""
            )

            # ---------- IMAGE (🔥 FIX) ----------
            image = (
                item.get("image_path")
                or item.get("image_url")
                or item.get("urls", {}).get("regular")
            )

            # ---------- Features ----------
            clip_similarity = float(-distances[0][rank])
            text_length = len(description)
            brightness = item.get("image_brightness", 0.5)

            features = pd.DataFrame([{
                "clip_similarity": clip_similarity,
                "text_length": text_length,
                "image_brightness": brightness
            }])

            rank_score = float(self.ranker.predict_proba(features)[0][1])

            results.append({
                "id": idx,
                "image": image,   # ✅ HTTPS URL
                "description": description,
                "clip_similarity": round(clip_similarity, 4),
                "rank_score": round(rank_score, 4)
            })

        results.sort(key=lambda x: x["rank_score"], reverse=True)
        return results


# ============================================================
# CLI test
# ============================================================
if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "sunset beach"

    engine = MultimodalSearchEngine()
    results = engine.search(query)

    print("\n🏆 Final Ranked Results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. ID: {r['id']}")
        print(f"   🧠 Rank score: {r['rank_score']}")
        print(f"   🔗 CLIP sim: {r['clip_similarity']}")
        print(f"   🖼️ Image: {r['image']}")
        print(f"   📝 {r['description']}\n")
