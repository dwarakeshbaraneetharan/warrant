#!/usr/bin/env python3
"""Turn each retrieval stage on and off, and score it against human judgments.

The reason this script exists is that "hybrid retrieval with a cross-encoder
reranker" is a sentence anybody can write in a README. Whether the hybrid beats
BM25 alone, and whether the reranker earns its latency, are measurable questions,
and this answers them on 300 claims with human relevance labels.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from warrant.corpus import load_dataset
from warrant.evaluation import RetrievalScores, score_retrieval
from warrant.retrieval import Retriever, RetrieverConfig

GENERAL_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BIOMEDICAL_RERANKER = "ncbi/MedCPT-Cross-Encoder"

# Ordered so each row isolates one change from the row above it. The last two are
# the same pipeline with two rerankers of comparable size, differing only in the
# domain they were trained on, which is the comparison the README turns on.
CONFIGURATIONS = [
    ("bm25 only", RetrieverConfig(use_bm25=True, use_dense=False, use_reranker=False)),
    ("dense only", RetrieverConfig(use_bm25=False, use_dense=True, use_reranker=False)),
    ("hybrid (rrf)", RetrieverConfig(use_bm25=True, use_dense=True, use_reranker=False)),
    (
        "hybrid + web rerank",
        RetrieverConfig(use_reranker=True, rerank_model=GENERAL_RERANKER),
    ),
    (
        "hybrid + bio rerank",
        RetrieverConfig(use_reranker=True, rerank_model=BIOMEDICAL_RERANKER),
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/scifact")
    ap.add_argument("--limit", type=int, default=None, help="claims to score")
    ap.add_argument("--only", default=None, help="substring of a configuration label")
    args = ap.parse_args()

    if not Path(args.data).exists():
        print(f"no data at {args.data}", file=sys.stderr)
        return 1

    dataset = load_dataset(args.data, "test")
    doc_ids = dataset.doc_ids
    texts = [dataset.documents[d].body for d in doc_ids]

    judged = dataset.claims_with_qrels()
    with_evidence = [c for c in judged if c.evidence]
    print(
        f"{len(dataset.documents):,} abstracts | {len(judged)} judged claims "
        f"({len(with_evidence)} also carry verdict labels)\n",
        file=sys.stderr,
    )

    # One retriever, indexed once. Rebuilding per configuration would spend all the
    # time re-embedding and would make the timing column meaningless.
    retriever = Retriever(RetrieverConfig())
    started = time.perf_counter()
    retriever.index(doc_ids, texts)
    print(f"indexed in {time.perf_counter() - started:.1f}s\n", file=sys.stderr)

    print(RetrievalScores.header())
    print("-" * 70)
    results = []
    for label, config in CONFIGURATIONS:
        if args.only and args.only not in label:
            continue
        retriever.use(config)
        started = time.perf_counter()
        scores = score_retrieval(label, dataset, retriever.search, 0.0, limit=args.limit)
        scores.seconds = time.perf_counter() - started
        results.append(scores)
        print(scores.row(), flush=True)

    print("-" * 70)
    print(f"scored on {results[0].claims if results else 0} claims with human judgments")

    named = {s.label: s for s in results}
    bm25, hybrid = named.get("bm25 only"), named.get("hybrid (rrf)")
    web, bio = named.get("hybrid + web rerank"), named.get("hybrid + bio rerank")

    if bm25 and hybrid:
        print(f"\nhybrid over bm25 alone:  {hybrid.ndcg_at_10 - bm25.ndcg_at_10:+.3f} nDCG@10")
    if hybrid and web:
        print(
            f"web-search reranker:     {web.ndcg_at_10 - hybrid.ndcg_at_10:+.3f} nDCG@10 "
            f"at {web.seconds / max(hybrid.seconds, 1e-9):.1f}x the time"
        )
    if hybrid and bio:
        print(
            f"biomedical reranker:     {bio.ndcg_at_10 - hybrid.ndcg_at_10:+.3f} nDCG@10 "
            f"at {bio.seconds / max(hybrid.seconds, 1e-9):.1f}x the time"
        )
    if web and bio:
        print(
            f"\nsame pipeline, same rough model size, different training domain: "
            f"{bio.ndcg_at_10 - web.ndcg_at_10:+.3f} nDCG@10"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
