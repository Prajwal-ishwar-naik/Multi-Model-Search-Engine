import os
import faiss
import numpy as np
import json

EMBEDDINGS_PATH = "data/embeddings/image_embeddings.npy"
INDEX_PATH = "data/index/faiss.index"

os.makedirs("data/index", exist_ok=True)

def main():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError("Run embedding generation first")

    print("📥 Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"✅ FAISS index saved to {INDEX_PATH}")

if __name__ == "__main__":
    main()
