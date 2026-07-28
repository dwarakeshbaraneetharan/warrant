"""Scoring retrieval against human relevance judgments.

Standard IR metrics, implemented here rather than imported, because the whole
argument of the project is that the numbers are trustworthy and a reader should be
able to check how they were computed without following a dependency.

Every metric is computed only over claims that have at least one judged abstract.
Including unjudged claims would score every system zero on them and pull all the
averages toward each other, which is the opposite of what an ablation needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .corpus import Dataset
from .retrieval import Hit


def dcg(relevances: list[int]) -> float:
    """Discounted cumulative gain with the standard log2 discount."""
    return sum(
        rel / math.log2(position + 1) for position, rel in enumerate(relevances, start=1)
    )


def ndcg_at_k(retrieved: list[str], judgments: dict[str, int], k: int) -> float:
    """nDCG@k: gain from what was returned, over gain from the best possible order.

    The ideal ranking is the judged documents sorted by relevance, truncated at k.
    Normalising against it is what makes the number comparable across claims with
    different numbers of relevant abstracts.
    """
    gains = [judgments.get(doc_id, 0) for doc_id in retrieved[:k]]
    ideal = sorted(judgments.values(), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(gains) / best if best > 0 else 0.0


def recall_at_k(retrieved: list[str], judgments: dict[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, score in judgments.items() if score > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def reciprocal_rank(retrieved: list[str], judgments: dict[str, int]) -> float:
    """1/rank of the first relevant hit, or zero.

    Worth reporting alongside nDCG because it answers a different question: not
    "how good is the ranking" but "does the user have to scroll". For a claim
    checker showing three sources, that is the metric a reader feels.
    """
    for position, doc_id in enumerate(retrieved, start=1):
        if judgments.get(doc_id, 0) > 0:
            return 1.0 / position
    return 0.0


@dataclass
class RetrievalScores:
    label: str
    claims: int
    ndcg_at_10: float
    recall_at_10: float
    recall_at_50: float
    mrr: float
    seconds: float

    def row(self) -> str:
        return (
            f"{self.label:<22}{self.ndcg_at_10:>9.3f}{self.recall_at_10:>11.3f}"
            f"{self.recall_at_50:>11.3f}{self.mrr:>8.3f}{self.seconds:>9.1f}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'configuration':<22}{'nDCG@10':>9}{'Recall@10':>11}"
            f"{'Recall@50':>11}{'MRR':>8}{'sec':>9}"
        )


def score_retrieval(
    label: str,
    dataset: Dataset,
    search: callable[[str, int], list[Hit]],
    seconds: float,
    limit: int | None = None,
) -> RetrievalScores:
    """Run `search` over every judged claim and average the metrics."""
    claims = dataset.claims_with_qrels()
    if limit is not None:
        claims = claims[:limit]

    ndcg_total = recall10_total = recall50_total = mrr_total = 0.0
    for claim in claims:
        judgments = dataset.qrels.get(claim.claim_id, {})
        hits = search(claim.text, 50)
        retrieved = [hit.doc_id for hit in hits]
        ndcg_total += ndcg_at_k(retrieved, judgments, 10)
        recall10_total += recall_at_k(retrieved, judgments, 10)
        recall50_total += recall_at_k(retrieved, judgments, 50)
        mrr_total += reciprocal_rank(retrieved, judgments)

    count = max(len(claims), 1)
    return RetrievalScores(
        label=label,
        claims=len(claims),
        ndcg_at_10=ndcg_total / count,
        recall_at_10=recall10_total / count,
        recall_at_50=recall50_total / count,
        mrr=mrr_total / count,
        seconds=seconds,
    )
