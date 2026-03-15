import json
import numpy as np
from vector_store.search_with_ranker import MultimodalSearchEngine
from evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k
)

# =========================
# Config
# =========================
QUERY_FILE = "data/eval/eval_queries.json"
TOP_K = 10


def load_eval_queries():
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_relevance_vector(results, relevant_ids):
    return [
        1 if ("id" in r and r["id"] in relevant_ids) else 0
        for r in results
    ]


def main():
    print("🚀 Running offline evaluation...")

    engine = MultimodalSearchEngine()
    eval_data = load_eval_queries()

    all_relevances = []
    ndcgs = []
    precisions = []

    for sample in eval_data:
        query = sample["query"]
        relevant_ids = set(sample["relevant_ids"])

        print(f"\n🔍 Evaluating query: '{query}'")
        print(f"Relevant IDs: {sorted(list(relevant_ids))}")

        results = engine.search(query)
        print(f"Retrieved IDs: {[r.get('id') for r in results]}")

        relevances = build_relevance_vector(results, relevant_ids)

        all_relevances.append(relevances)
        ndcgs.append(ndcg_at_k(relevances, TOP_K))
        precisions.append(precision_at_k(relevances, TOP_K))

    print("\n📊 Evaluation Results")
    print(f"MRR@{TOP_K}:  {mean_reciprocal_rank(all_relevances):.4f}")
    print(f"nDCG@{TOP_K}: {np.mean(ndcgs):.4f}")
    print(f"P@{TOP_K}:    {np.mean(precisions):.4f}")


if __name__ == "__main__":
    main()
