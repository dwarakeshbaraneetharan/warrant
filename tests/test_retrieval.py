"""Tests for retrieval and the metrics that judge it.

The metric tests matter most. Every claim in the README rests on nDCG and recall
being computed correctly, so they are checked against cases whose answers can be
worked out by hand rather than only against the pipeline's own output.
"""

from __future__ import annotations

import math

import pytest

from warrant.evaluation import dcg, ndcg_at_k, recall_at_k, reciprocal_rank
from warrant.retrieval import (
    Bm25,
    Hit,
    reciprocal_rank_fusion,
    tokenize,
)

DOCS = {
    "1": "Vitamin D deficiency increases the risk of respiratory infection in adults.",
    "2": "Chocolate consumption and cardiovascular outcomes in a Swedish cohort study.",
    "3": "Respiratory infection rates fell after a vitamin D supplementation trial.",
    "4": "The structural biology of ribosome assembly in yeast.",
}


@pytest.fixture
def bm25() -> Bm25:
    ids = list(DOCS)
    return Bm25().index(ids, [DOCS[i] for i in ids])


def test_tokenizer_is_lowercase_and_alphanumeric():
    assert tokenize("Vitamin-D, 25(OH)D levels!") == ["vitamin", "d", "25", "oh", "d", "levels"]
    assert tokenize("") == []


def test_bm25_ranks_the_relevant_document_first(bm25):
    hits = bm25.search("vitamin D respiratory infection", top_k=4)
    assert hits, "expected at least one hit"
    assert hits[0].doc_id in {"1", "3"}
    # The yeast abstract shares no terms, so it must not be returned at all.
    assert "4" not in {hit.doc_id for hit in hits}


def test_bm25_returns_nothing_for_an_unseen_term(bm25):
    assert bm25.search("proteomics telomerase", top_k=4) == []


def test_bm25_respects_top_k(bm25):
    assert len(bm25.search("vitamin infection cohort study", top_k=2)) <= 2


def test_bm25_scores_are_positive_and_ordered(bm25):
    hits = bm25.search("vitamin D infection", top_k=4)
    scores = [hit.score for hit in hits]
    assert all(score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_bm25_length_normalisation_prefers_the_focused_document():
    """b > 0 means a short, on-topic abstract should beat a long, padded one."""
    padding = " ".join(["unrelated filler text"] * 60)
    index = Bm25().index(
        ["short", "long"],
        [
            "vitamin D and respiratory infection",
            f"vitamin D and respiratory infection {padding}",
        ],
    )
    hits = index.search("vitamin D respiratory infection", top_k=2)
    assert hits[0].doc_id == "short"


def test_rrf_rewards_agreement_between_rankings():
    """A document both retrievers like should beat one only a single retriever likes."""
    a = [Hit("x", 9.0), Hit("y", 8.0), Hit("z", 1.0)]
    b = [Hit("y", 0.9), Hit("x", 0.8), Hit("w", 0.1)]

    fused = reciprocal_rank_fusion([a, b], top_k=4)
    top_two = {hit.doc_id for hit in fused[:2]}

    assert top_two == {"x", "y"}
    assert fused[0].stage == "hybrid"


def test_rrf_ignores_score_scale():
    """The reason RRF is used instead of a weighted sum of scores.

    One retriever's scores are inflated by three orders of magnitude. A score-based
    fusion would let it dominate; a rank-based one cannot see the difference.
    """
    huge = [Hit("a", 5000.0), Hit("b", 4000.0)]
    tiny = [Hit("b", 0.002), Hit("a", 0.001)]

    scaled = reciprocal_rank_fusion([huge, tiny], top_k=2)
    unscaled = reciprocal_rank_fusion(
        [[Hit("a", 2.0), Hit("b", 1.0)], [Hit("b", 2.0), Hit("a", 1.0)]], top_k=2
    )
    assert [h.doc_id for h in scaled] == [h.doc_id for h in unscaled]


def test_rrf_of_a_single_ranking_preserves_its_order():
    single = [Hit("a", 3.0), Hit("b", 2.0), Hit("c", 1.0)]
    fused = reciprocal_rank_fusion([single], top_k=3)
    assert [hit.doc_id for hit in fused] == ["a", "b", "c"]


# --- metrics ---------------------------------------------------------------


def test_dcg_applies_the_log_discount():
    # A relevant document at rank 2 is worth 1/log2(3), not 1.
    assert dcg([0, 1]) == pytest.approx(1 / math.log2(3))
    assert dcg([1, 0]) == pytest.approx(1.0)


def test_ndcg_is_one_for_the_ideal_ranking():
    judgments = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], judgments, 10) == pytest.approx(1.0)


def test_ndcg_is_zero_when_nothing_relevant_is_returned():
    assert ndcg_at_k(["x", "y"], {"a": 1}, 10) == 0.0


def test_ndcg_rewards_putting_the_relevant_document_higher():
    judgments = {"a": 1}
    assert ndcg_at_k(["a", "x"], judgments, 10) > ndcg_at_k(["x", "a"], judgments, 10)


def test_ndcg_truncates_at_k():
    """A relevant hit past position k must not count."""
    judgments = {"a": 1}
    assert ndcg_at_k(["x", "y", "z", "a"], judgments, 3) == 0.0
    assert ndcg_at_k(["x", "y", "z", "a"], judgments, 4) > 0.0


def test_ndcg_with_no_judgments_is_zero_not_an_error():
    assert ndcg_at_k(["a"], {}, 10) == 0.0


def test_recall_counts_relevant_documents_found():
    judgments = {"a": 1, "b": 1, "c": 1}
    assert recall_at_k(["a", "b", "x"], judgments, 10) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b", "c"], judgments, 10) == pytest.approx(1.0)
    assert recall_at_k([], judgments, 10) == 0.0


def test_recall_is_insensitive_to_order_within_k():
    """Unlike nDCG, recall should not care where in the top-k a hit landed."""
    judgments = {"a": 1}
    assert recall_at_k(["a", "x"], judgments, 2) == recall_at_k(["x", "a"], judgments, 2)


def test_reciprocal_rank_finds_the_first_relevant_position():
    judgments = {"b": 1}
    assert reciprocal_rank(["a", "b", "c"], judgments) == pytest.approx(0.5)
    assert reciprocal_rank(["b"], judgments) == pytest.approx(1.0)
    assert reciprocal_rank(["a", "c"], judgments) == 0.0
