import faiss
import numpy as np
import os

EMB_PATH = "data/embeddings/image_embeddings.npy"
INDEX_PATH = "data/embeddings/image.index"


def main():
    if not os.path.exists(EMB_PATH):
        raise RuntimeError("image_embeddings.npy not found")

    embeddings = np.load(EMB_PATH).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs("data/embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    print(f"✅ FAISS index created with {index.ntotal} vectors")


if __name__ == "__main__":
    main()
