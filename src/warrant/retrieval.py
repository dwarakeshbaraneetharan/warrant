"""Retrieval: BM25, dense embeddings, fusion, and reranking.

Each stage is separable and separately measurable, which is the point. A pipeline
that bolts all four together and reports one number cannot tell you whether the
reranker earned its 200 ms, and most portfolio RAG systems cannot answer that
question about themselves. `warrant evaluate` turns each stage on and off and
reports nDCG against human judgments, so every choice here is a measurement rather
than an opinion.

BM25 is written out rather than imported. It is thirty lines, it is the baseline
everything else is judged against, and its two constants are exactly the kind of
thing an interviewer asks about.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

#: Lowercase alphanumeric runs. Scientific text is full of hyphenated compounds and
#: units, so splitting on non-alphanumerics keeps "il-6" as "il" and "6" — crude,
#: but consistent between indexing and querying, which is what matters for BM25.
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    doc_id: str
    score: float
    #: Which stage produced this ranking, for debugging and for the API response.
    stage: str = ""


class Bm25:
    """Okapi BM25 over an in-memory inverted index.

    `k1` bounds how much a repeated term can keep helping: without it, an abstract
    that says "melanoma" nine times would outrank one that says it twice and is
    actually about the claim. `b` controls length normalisation, which matters here
    because SciFact abstracts vary from one sentence to twenty.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths: np.ndarray = np.zeros(0)
        self.average_length: float = 0.0
        self.idf: dict[str, float] = {}

    def index(self, doc_ids: list[str], texts: list[str]) -> Bm25:
        self.doc_ids = list(doc_ids)
        self.postings = {}
        lengths = []

        for position, text in enumerate(texts):
            tokens = tokenize(text)
            lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings.setdefault(term, []).append((position, count))

        self.lengths = np.array(lengths, dtype=np.float64)
        self.average_length = float(self.lengths.mean()) if len(lengths) else 0.0

        total = len(doc_ids)
        # Robertson-Sparck-Jones idf with the +0.5 smoothing, floored at zero so a
        # term appearing in most documents cannot contribute a negative score.
        self.idf = {
            term: max(math.log((total - len(posting) + 0.5) / (len(posting) + 0.5) + 1.0), 0.0)
            for term, posting in self.postings.items()
        }
        return self

    def search(self, query: str, top_k: int = 50) -> list[Hit]:
        scores = np.zeros(len(self.doc_ids), dtype=np.float64)
        denominator_length = self.k1 * (
            1 - self.b + self.b * (self.lengths / max(self.average_length, 1e-9))
        )

        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            weight = self.idf[term]
            for position, count in posting:
                scores[position] += weight * (
                    count * (self.k1 + 1) / (count + denominator_length[position])
                )

        return self._top(scores, top_k, "bm25")

    def _top(self, scores: np.ndarray, top_k: int, stage: str) -> list[Hit]:
        if not len(scores):
            return []
        top_k = min(top_k, len(scores))
        # argpartition first: sorting 5,000 scores to read the top 50 is wasteful.
        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        candidates = candidates[np.argsort(-scores[candidates])]
        return [
            Hit(doc_id=self.doc_ids[i], score=float(scores[i]), stage=stage)
            for i in candidates
            if scores[i] > 0
        ]


class DenseIndex:
    """Embedding retrieval over normalised vectors.

    Vectors are L2-normalised at index time so similarity is a single matrix
    multiply rather than a cosine with per-query norms. At 5,000 abstracts an exact
    search is well under a millisecond, so there is no approximate index here; the
    complexity would buy nothing and would be a thing to explain that does not
    matter.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.doc_ids: list[str] = []
        self.vectors: np.ndarray = np.zeros((0, 0))
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, doc_ids: list[str], texts: list[str], batch_size: int = 64) -> DenseIndex:
        self.doc_ids = list(doc_ids)
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.vectors = np.asarray(vectors, dtype=np.float32)
        return self

    def search(self, query: str, top_k: int = 50) -> list[Hit]:
        if not len(self.vectors):
            return []
        vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.vectors @ np.asarray(vector, dtype=np.float32)
        top_k = min(top_k, len(scores))
        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        candidates = candidates[np.argsort(-scores[candidates])]
        return [
            Hit(doc_id=self.doc_ids[i], score=float(scores[i]), stage="dense")
            for i in candidates
        ]


def reciprocal_rank_fusion(
    rankings: list[list[Hit]], k: float = 60.0, top_k: int = 50
) -> list[Hit]:
    """Combine rankings by rank rather than by score.

    BM25 scores are unbounded sums and cosine similarities live in [-1, 1], so any
    weighted average of the two is really a weighted average of two arbitrary
    scales, and the weight silently changes meaning as the corpus does. RRF only
    uses positions, so it needs no calibration and cannot be broken by one
    retriever's scores drifting.

    `k` damps the advantage of rank 1 over rank 2; 60 is the value from the original
    paper and the sweep in `warrant evaluate` did not beat it here.
    """
    totals: dict[str, float] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            totals[hit.doc_id] = totals.get(hit.doc_id, 0.0) + 1.0 / (k + position)

    ordered = sorted(totals.items(), key=lambda pair: -pair[1])[:top_k]
    return [Hit(doc_id=doc_id, score=score, stage="hybrid") for doc_id, score in ordered]


class CrossEncoderReranker:
    """Rescore candidates by reading claim and abstract together.

    BM25 and embeddings both score a query and a document independently, which is
    what makes them fast enough to run over the whole corpus and also what limits
    them. A cross-encoder attends across both at once and is far more accurate, but
    costs a forward pass per candidate, so it only ever sees the top few dozen.
    That asymmetry is the entire reason for the retrieve-then-rerank shape.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, hits: list[Hit], documents: dict[str, str], top_k: int = 10
    ) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, documents.get(hit.doc_id, "")) for hit in hits]
        # Already in [0, 1]: sentence-transformers applies a sigmoid for
        # single-label cross-encoders, so these are usable as confidences directly.
        # Worth checking rather than assuming — squashing them a second time
        # collapses everything to ~0.5 and throws away all the discrimination.
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            (
                Hit(doc_id=hit.doc_id, score=float(score), stage="rerank")
                for hit, score in zip(hits, scores, strict=True)
            ),
            key=lambda hit: -hit.score,
        )
        return ranked[:top_k]


#: General-purpose sentence embedder. Small and CPU-friendly, which is a
#: requirement rather than a preference: the deployed demo runs on a free tier.
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: A cross-encoder trained on PubMed search logs rather than web search. See the
#: README: on this corpus the domain of the reranker matters far more than its size.
DEFAULT_RERANK_MODEL = "ncbi/MedCPT-Cross-Encoder"


@dataclass
class RetrieverConfig:
    use_bm25: bool = True
    use_dense: bool = True
    #: On, but only because the *domain-matched* reranker earns it: +0.095 nDCG@10
    #: on 300 judged claims, against +0.005 for a general web-search reranker of
    #: comparable size. Run `python bench/ablation.py` to reproduce that.
    use_reranker: bool = True
    #: Candidates each retriever contributes before fusion. Wider costs the reranker
    #: proportionally more, since it runs one forward pass per candidate.
    candidates: int = 50
    top_k: int = 10
    rrf_k: float = 60.0
    dense_model: str = DEFAULT_DENSE_MODEL
    rerank_model: str = DEFAULT_RERANK_MODEL


class Retriever:
    """The pipeline: retrieve wide with cheap scorers, then optionally rerank."""

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig()
        self.bm25 = Bm25()
        self.dense = DenseIndex(self.config.dense_model)
        self.reranker = CrossEncoderReranker(self.config.rerank_model)
        self.texts: dict[str, str] = {}

    def index(self, doc_ids: list[str], texts: list[str]) -> Retriever:
        self.texts = dict(zip(doc_ids, texts, strict=True))
        if self.config.use_bm25:
            self.bm25.index(doc_ids, texts)
        if self.config.use_dense:
            self.dense.index(doc_ids, texts)
        return self

    def use(self, config: RetrieverConfig) -> Retriever:
        """Swap configuration, rebuilding only what the change requires.

        Lets the ablation switch stages without re-embedding 5,000 abstracts each
        time, which would dominate the runtime and make the timing column useless.
        """
        if config.dense_model != self.config.dense_model:
            raise ValueError("changing the dense model requires a fresh index")
        if config.rerank_model != self.config.rerank_model:
            self.reranker = CrossEncoderReranker(config.rerank_model)
        self.config = config
        return self

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.config.top_k
        rankings = []
        if self.config.use_bm25:
            rankings.append(self.bm25.search(query, self.config.candidates))
        if self.config.use_dense:
            rankings.append(self.dense.search(query, self.config.candidates))

        if not rankings:
            return []
        if len(rankings) == 1:
            candidates = rankings[0]
        else:
            candidates = reciprocal_rank_fusion(
                rankings, k=self.config.rrf_k, top_k=self.config.candidates
            )

        if self.config.use_reranker:
            return self.reranker.rerank(query, candidates, self.texts, top_k)
        return candidates[:top_k]
