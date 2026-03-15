import numpy as np


def precision_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    return np.sum(relevances) / k


def reciprocal_rank(relevances):
    for i, rel in enumerate(relevances):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(all_relevances):
    return np.mean([reciprocal_rank(r) for r in all_relevances])


def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    return np.sum(
        (2 ** relevances - 1) / np.log2(np.arange(2, len(relevances) + 2))
    )


def ndcg_at_k(relevances, k):
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0
