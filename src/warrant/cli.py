"""Command line entry point: fetch the corpus, serve the app, score retrieval."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import httpx

from .corpus import load_dataset, summarise
from .evaluation import RetrievalScores, score_retrieval
from .retrieval import Retriever, RetrieverConfig

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download SciFact, which ships the corpus, claims and human judgments."""
    target = Path(args.out)
    if (target / "scifact" / "corpus.jsonl").exists() and not args.force:
        print(f"already present at {target / 'scifact'} (use --force to refetch)")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    print(f"fetching {SCIFACT_URL} ...", file=sys.stderr)
    response = httpx.get(SCIFACT_URL, timeout=180.0, follow_redirects=True)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(target)

    dataset = load_dataset(target / "scifact", "test")
    print(summarise(dataset))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(args.data), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score retrieval against human relevance judgments, and optionally gate on it."""
    data = Path(args.data)
    if not (data / "corpus.jsonl").exists():
        print(f"no corpus at {data}. Run `warrant fetch` first.", file=sys.stderr)
        return 1

    dataset = load_dataset(data, args.split)
    doc_ids = dataset.doc_ids
    texts = [dataset.documents[d].body for d in doc_ids]

    config = RetrieverConfig(use_reranker=args.reranker, candidates=args.candidates)
    retriever = Retriever(config)
    started = time.perf_counter()
    retriever.index(doc_ids, texts)
    index_seconds = time.perf_counter() - started

    label = "hybrid + rerank" if args.reranker else "hybrid"
    started = time.perf_counter()
    scores = score_retrieval(label, dataset, retriever.search, 0.0, limit=args.limit)
    scores.seconds = time.perf_counter() - started

    print(f"indexed {len(doc_ids):,} abstracts in {index_seconds:.1f}s\n")
    print(RetrievalScores.header())
    print("-" * 70)
    print(scores.row())
    print("-" * 70)
    print(f"{scores.claims} claims with human relevance judgments")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "label": scores.label,
                    "claims": scores.claims,
                    "ndcg_at_10": scores.ndcg_at_10,
                    "recall_at_10": scores.recall_at_10,
                    "recall_at_50": scores.recall_at_50,
                    "mrr": scores.mrr,
                },
                indent=2,
            )
            + "\n"
        )

    # A regression gate rather than a leaderboard. The threshold sits below the
    # measured value with room for the variance of a smaller claim sample, so it
    # catches a pipeline that broke, not one that moved a little.
    if args.min_ndcg is not None and scores.ndcg_at_10 < args.min_ndcg:
        print(
            f"\nFAIL: nDCG@10 {scores.ndcg_at_10:.3f} is below the gate of {args.min_ndcg:.3f}",
            file=sys.stderr,
        )
        return 1
    if args.min_recall is not None and scores.recall_at_50 < args.min_recall:
        print(
            f"\nFAIL: Recall@50 {scores.recall_at_50:.3f} is below the gate "
            f"of {args.min_recall:.3f}",
            file=sys.stderr,
        )
        return 1
    if args.min_ndcg is not None or args.min_recall is not None:
        print("\ngates passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="warrant",
        description="Check scientific claims against the literature, with measured retrieval.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download the SciFact corpus and judgments")
    fetch.add_argument("--out", default="data")
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    serve = sub.add_parser("serve", help="run the web app")
    serve.add_argument("--data", default="data/scifact")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    ev = sub.add_parser("evaluate", help="score retrieval against human judgments")
    ev.add_argument("--data", default="data/scifact")
    ev.add_argument("--split", default="test")
    ev.add_argument("--limit", type=int, default=None, help="claims to score")
    ev.add_argument("--candidates", type=int, default=50)
    ev.add_argument(
        "--reranker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include the cross-encoder stage (slow; downloads a larger model)",
    )
    ev.add_argument("--min-ndcg", type=float, default=None, help="fail below this nDCG@10")
    ev.add_argument("--min-recall", type=float, default=None, help="fail below this Recall@50")
    ev.add_argument("--json", default=None, help="write scores to this path")
    ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
