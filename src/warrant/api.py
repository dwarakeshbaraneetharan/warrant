"""HTTP service and the web UI.

Indexing happens once at startup and lives in memory. At 5,183 abstracts that is
roughly 8 MB of float32 vectors plus an inverted index, which fits comfortably in a
free-tier container and removes an entire class of moving parts: no vector database
to run, no schema to migrate, no network hop per query. It is the right call at this
size and the wrong one two orders of magnitude up, which is the sort of thing worth
saying out loud rather than discovering in production.

The interesting endpoint is `/api/verify`, which always returns its retrieved
sources and any grounding warning even when generation fails. A claim checker that
silently degrades to an unsourced opinion is worse than one that says it could not
answer.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .corpus import Dataset, Verdict, load_dataset
from .retrieval import Retriever, RetrieverConfig
from .verdict import Answer, LlmClient, verify


def get_web_root() -> Path:
    env_root = os.environ.get("WARRANT_WEB")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    if Path("/app/web").exists():
        return Path("/app/web")
    return Path(__file__).resolve().parents[2] / "web"


WEB_ROOT = get_web_root()


class VerifyRequest(BaseModel):
    claim: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    #: Lets a reader reproduce the ablation from the UI instead of taking it on
    #: trust, which is most of the point of the project.
    use_reranker: bool = True


class SourceOut(BaseModel):
    doc_id: str
    title: str
    score: float
    stage: str
    sentences: list[str]
    #: Sentence indices the answer cited, so the UI can highlight them.
    cited: list[int] = []


class VerifyResponse(BaseModel):
    claim: str
    verdict: str
    reasoning: str
    grounded: bool
    warning: str | None
    sources: list[SourceOut]
    retrieval_ms: float
    generation_ms: float
    model: str


class SearchResponse(BaseModel):
    claim: str
    took_ms: float
    sources: list[SourceOut]


class State:
    """Everything loaded once at startup."""

    dataset: Dataset | None = None
    retriever: Retriever | None = None
    client: LlmClient | None = None
    ready: bool = False
    error: str | None = None
    indexed_seconds: float = 0.0


state = State()


def build_state(data_dir: str) -> None:
    started = time.perf_counter()
    dataset = load_dataset(data_dir, "test")
    doc_ids = dataset.doc_ids
    texts = [dataset.documents[d].body for d in doc_ids]

    # Narrower candidate set than the benchmark uses. The reranker costs one forward
    # pass per candidate, and 50 puts an interactive request near two seconds; 25
    # halves that. The ablation shows Recall@50 is already 0.940 before reranking,
    # so the abstracts that matter are almost always inside the first 25.
    retriever = Retriever(RetrieverConfig(candidates=25))
    retriever.index(doc_ids, texts, cache_dir=data_dir)

    state.dataset = dataset
    state.retriever = retriever
    state.client = LlmClient()
    state.indexed_seconds = time.perf_counter() - started
    state.ready = True


def _sources(hits, dataset: Dataset, answer: Answer | None = None) -> list[SourceOut]:
    cited_by_doc: dict[str, list[int]] = {}
    if answer is not None:
        for citation in answer.citations:
            cited_by_doc.setdefault(citation.doc_id, []).append(citation.sentence_index)

    out = []
    for hit in hits:
        document = dataset.documents.get(hit.doc_id)
        if document is None:
            continue
        out.append(
            SourceOut(
                doc_id=hit.doc_id,
                title=document.title,
                score=round(hit.score, 4),
                stage=hit.stage,
                sentences=list(document.sentences),
                cited=sorted(cited_by_doc.get(hit.doc_id, [])),
            )
        )
    return out


_background_tasks = set()


def create_app(data_dir: str | None = None) -> FastAPI:
    data_dir = data_dir or os.environ.get("WARRANT_DATA", "data/scifact")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import asyncio

        def _worker():
            try:
                build_state(data_dir)
            except Exception as exc:
                state.error = str(exc)

        task = asyncio.create_task(asyncio.to_thread(_worker))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        yield

    app = FastAPI(
        title="warrant",
        version="0.1.0",
        description="Checks a scientific claim against the literature, with citations.",
        lifespan=lifespan,
    )

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok" if state.ready else "loading",
            "error": state.error,
            "abstracts": len(state.dataset.documents) if state.dataset else 0,
            "indexed_seconds": round(state.indexed_seconds, 2),
            "generation_available": bool(state.client and state.client.available),
        }

    def require_ready() -> tuple[Dataset, Retriever]:
        if state.error:
            raise HTTPException(status_code=503, detail=f"index failed: {state.error}")
        if not state.ready or state.dataset is None or state.retriever is None:
            raise HTTPException(status_code=503, detail="still indexing, try again shortly")
        return state.dataset, state.retriever

    @app.post("/api/search", response_model=SearchResponse)
    async def search(body: VerifyRequest) -> SearchResponse:
        """Retrieval only. Useful for seeing what the reranker actually changes."""
        dataset, retriever = require_ready()
        retriever.config.use_reranker = body.use_reranker
        started = time.perf_counter()
        hits = retriever.search(body.claim, body.top_k)
        return SearchResponse(
            claim=body.claim,
            took_ms=round((time.perf_counter() - started) * 1000, 1),
            sources=_sources(hits, dataset),
        )

    @app.post("/api/verify", response_model=VerifyResponse)
    async def verify_claim(body: VerifyRequest) -> VerifyResponse:
        dataset, retriever = require_ready()
        retriever.config.use_reranker = body.use_reranker

        started = time.perf_counter()
        hits = retriever.search(body.claim, body.top_k)
        retrieval_ms = (time.perf_counter() - started) * 1000

        client = state.client or LlmClient()
        if not client.available:
            # Retrieval still works without a key, so return it rather than 503.
            return VerifyResponse(
                claim=body.claim,
                verdict=Verdict.NOT_ENOUGH_INFO.value,
                reasoning="",
                grounded=False,
                warning="GROQ_API_KEY is not configured, so only retrieval ran",
                sources=_sources(hits, dataset),
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=0.0,
                model="",
            )

        answer = verify(body.claim, hits, dataset.documents, client)
        return VerifyResponse(
            claim=body.claim,
            verdict=answer.verdict.value,
            reasoning=answer.reasoning,
            grounded=answer.grounded,
            warning=answer.warning,
            sources=_sources(hits, dataset, answer),
            retrieval_ms=round(retrieval_ms, 1),
            generation_ms=round(answer.latency_ms, 1),
            model=answer.model,
        )

    @app.get("/api/examples")
    async def examples() -> dict[str, list[dict[str, str]]]:
        """Claims with known verdicts, so a visitor can see all three outcomes.

        Picked from the labelled set rather than invented, so the demo cannot be
        accused of being tuned to its own examples.
        """
        if state.dataset is None:
            return {"examples": []}

        wanted = [Verdict.SUPPORTED, Verdict.REFUTED, Verdict.NOT_ENOUGH_INFO]
        chosen: list[dict[str, str]] = []
        for verdict in wanted:
            for claim in state.dataset.claims.values():
                if claim.verdict is verdict and 40 < len(claim.text) < 130:
                    chosen.append({"claim": claim.text, "expected": verdict.value})
                    break
        return {"examples": chosen}

    if WEB_ROOT.exists():
        app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(WEB_ROOT / "index.html")

    return app


app = create_app()
