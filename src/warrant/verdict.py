"""Turning retrieved abstracts into a verdict with citations.

Two design commitments here, both of which exist because the failure mode of a
claim checker is not being wrong — it is being confidently wrong with a citation
attached.

**Every sentence of the answer must carry a citation, or it is rejected.** The
model is asked to write in the form `[doc_id:sentence_index]`, those markers are
parsed, and any answer with an uncited assertion is downgraded rather than shown.
This is checked in code, not requested in the prompt, because a prompt is a
suggestion and a parser is a guarantee.

**NOT_ENOUGH_INFO is a first-class answer.** Roughly a third of SciFact's claims
have no supporting evidence in the corpus at all, and a system that always produces
a verdict will confidently answer all of them. Declining is the behaviour worth
measuring, so the prompt and the evaluation both treat it as a real option.

The generation call goes to Groq because its free tier is fast enough to keep a
demo interactive. Nothing here depends on that: `LlmClient` is one method wide.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import httpx

from .corpus import Document, Verdict
from .retrieval import Hit

#: `[12345:2]` means sentence 2 of abstract 12345.
CITATION = re.compile(r"\[(\d+):(\d+)\]")

SYSTEM_PROMPT = """\
You verify scientific claims against abstracts. You are precise and you never \
guess.

Rules:
1. Answer only from the numbered abstracts provided. Never use outside knowledge.
2. Every sentence you write must end with at least one citation in the form \
[abstract_id:sentence_number], naming the exact sentence that supports it.
3. Choose one verdict:
   SUPPORTED       - an abstract directly supports the claim
   REFUTED         - an abstract directly contradicts the claim
   NOT_ENOUGH_INFO - the abstracts do not settle it
4. Prefer NOT_ENOUGH_INFO over a guess. A claim that is merely related to an \
abstract is not supported by it.

Reply with JSON only:
{"verdict": "...", "reasoning": "one or two sentences, every one cited"}"""


@dataclass
class Citation:
    doc_id: str
    sentence_index: int


@dataclass
class Answer:
    verdict: Verdict
    reasoning: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[Hit] = field(default_factory=list)
    #: Set when the answer was rejected or degraded, and why.
    warning: str | None = None
    model: str = ""
    latency_ms: float = 0.0

    @property
    def cited_doc_ids(self) -> set[str]:
        return {c.doc_id for c in self.citations}

    @property
    def grounded(self) -> bool:
        """Whether every claim in the reasoning carries a citation."""
        return self.warning is None and bool(self.citations)


class LlmClient:
    """Minimal Groq chat client.

    Deliberately not the vendor SDK: one HTTP call is easier to reason about, has
    no version drift, and makes the retry and timeout behaviour visible instead of
    hidden in a library default.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 45.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> str:
        if not self.available:
            raise RuntimeError("GROQ_API_KEY is not set")
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # Zero temperature so the evaluation is repeatable; a claim checker
                # has no use for creativity.
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def build_prompt(claim: str, hits: list[Hit], documents: dict[str, Document]) -> str:
    """Render the retrieved abstracts with stable sentence numbers.

    Sentences are numbered explicitly rather than left to the model to count, so a
    citation can be checked against the source mechanically.
    """
    blocks = []
    for hit in hits:
        document = documents.get(hit.doc_id)
        if document is None:
            continue
        lines = [f"Abstract {hit.doc_id}: {document.title}"]
        for index, sentence in enumerate(document.sentences):
            lines.append(f"  [{hit.doc_id}:{index}] {sentence}")
        blocks.append("\n".join(lines))

    return f"Claim: {claim}\n\n" + "\n\n".join(blocks)


def parse_citations(text: str) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    out = []
    for doc_id, index in CITATION.findall(text):
        key = (doc_id, int(index))
        if key not in seen:
            seen.add(key)
            out.append(Citation(doc_id=doc_id, sentence_index=int(index)))
    return out


def _sentences_of(reasoning: str) -> list[str]:
    """Split reasoning into sentences for the citation check."""
    parts = re.split(r"(?<=[.!?])\s+", reasoning.strip())
    return [p for p in parts if p.strip()]


def validate(answer: Answer, documents: dict[str, Document], hits: list[Hit]) -> Answer:
    """Reject answers that cite nothing, cite fiction, or assert without citing.

    Enforced after generation rather than asked for in the prompt. A model that
    invents `[99999:3]` has produced exactly the kind of authoritative-looking
    output this project exists to catch, so the check is mechanical.
    """
    retrieved = {hit.doc_id for hit in hits}

    if answer.verdict is Verdict.NOT_ENOUGH_INFO:
        # Declining needs no evidence, so there is nothing to ground.
        return answer

    if not answer.citations:
        answer.warning = "answer cited no source, downgraded to NOT_ENOUGH_INFO"
        answer.verdict = Verdict.NOT_ENOUGH_INFO
        return answer

    for citation in answer.citations:
        if citation.doc_id not in retrieved:
            answer.warning = f"cited abstract {citation.doc_id}, which was never retrieved"
            answer.verdict = Verdict.NOT_ENOUGH_INFO
            return answer
        document = documents.get(citation.doc_id)
        if document is None or citation.sentence_index >= len(document.sentences):
            answer.warning = (
                f"cited sentence {citation.doc_id}:{citation.sentence_index}, "
                "which does not exist"
            )
            answer.verdict = Verdict.NOT_ENOUGH_INFO
            return answer

    uncited = [s for s in _sentences_of(answer.reasoning) if not CITATION.search(s)]
    if uncited:
        answer.warning = f"{len(uncited)} sentence(s) asserted without a citation"

    return answer


def verify(
    claim: str,
    hits: list[Hit],
    documents: dict[str, Document],
    client: LlmClient,
) -> Answer:
    """Retrieve-then-verify: produce a cited verdict for one claim."""
    import time

    if not hits:
        return Answer(
            verdict=Verdict.NOT_ENOUGH_INFO,
            reasoning="No abstract in the corpus was relevant to this claim.",
            warning="retrieval returned nothing",
        )

    started = time.perf_counter()
    try:
        raw = client.complete(SYSTEM_PROMPT, build_prompt(claim, hits, documents))
    except Exception as exc:
        return Answer(
            verdict=Verdict.NOT_ENOUGH_INFO,
            reasoning="",
            sources=hits,
            warning=f"generation failed: {exc}",
            model=client.model,
        )
    latency = (time.perf_counter() - started) * 1000

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return Answer(
            verdict=Verdict.NOT_ENOUGH_INFO,
            reasoning=raw[:400],
            sources=hits,
            warning="model did not return JSON",
            model=client.model,
            latency_ms=latency,
        )

    reasoning = str(payload.get("reasoning", "")).strip()
    try:
        verdict = Verdict(str(payload.get("verdict", "")).strip().upper())
    except ValueError:
        verdict = Verdict.NOT_ENOUGH_INFO
        reasoning = reasoning or "Model returned an unrecognised verdict."

    answer = Answer(
        verdict=verdict,
        reasoning=reasoning,
        citations=parse_citations(reasoning),
        sources=hits,
        model=client.model,
        latency_ms=latency,
    )
    return validate(answer, documents, hits)
