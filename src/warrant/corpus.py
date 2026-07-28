"""Loading SciFact: the abstracts, the claims, and the human labels.

The reason this project uses a benchmark rather than a folder of PDFs is that a
benchmark comes with human judgments, and without them there is no honest way to
say whether retrieval worked. SciFact supplies three separate kinds of label, and
this module keeps all three because the evaluation uses all three:

    qrels       which abstracts a human judged relevant to a claim
    verdict     whether the literature SUPPORTS or REFUTES the claim
    rationale   which sentences of the abstract justify that verdict

The third is the interesting one. It makes it possible to ask not just "did the
answer cite a source" but "did it cite the sentence a human would have cited",
which is the difference between a citation and a citation that means something.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    #: SciFact's claims with no annotated evidence. Answering these correctly means
    #: declining to answer, which is the behaviour that matters most in practice.
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


#: SciFact writes labels from the abstract's point of view; these are ours.
_LABEL_MAP = {"SUPPORT": Verdict.SUPPORTED, "CONTRADICT": Verdict.REFUTED}


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    #: The abstract split into sentences, as SciFact's rationale indices address it.
    sentences: tuple[str, ...]

    @property
    def body(self) -> str:
        return f"{self.title}\n\n{self.text}"


@dataclass
class Evidence:
    """One abstract's verdict on a claim, and the sentences that justify it."""

    doc_id: str
    verdict: Verdict
    sentence_indices: tuple[int, ...]


@dataclass
class Claim:
    claim_id: str
    text: str
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        """The claim's overall verdict.

        A claim with no annotated evidence is NOT_ENOUGH_INFO. Where abstracts
        disagree, SUPPORTED wins, matching SciFact's own convention.
        """
        if not self.evidence:
            return Verdict.NOT_ENOUGH_INFO
        verdicts = {e.verdict for e in self.evidence}
        if Verdict.SUPPORTED in verdicts:
            return Verdict.SUPPORTED
        return Verdict.REFUTED

    @property
    def evidence_doc_ids(self) -> set[str]:
        return {e.doc_id for e in self.evidence}

    def rationale_for(self, doc_id: str) -> set[int]:
        """Sentence indices a human marked as justifying the verdict."""
        out: set[int] = set()
        for item in self.evidence:
            if item.doc_id == doc_id:
                out.update(item.sentence_indices)
        return out


@dataclass
class Dataset:
    documents: dict[str, Document]
    claims: dict[str, Claim]
    #: claim_id -> {doc_id -> relevance}, the human relevance judgments.
    qrels: dict[str, dict[str, int]]

    @property
    def doc_ids(self) -> list[str]:
        return list(self.documents)

    def claims_with_qrels(self) -> list[Claim]:
        """Claims that have at least one judged abstract.

        Retrieval metrics are only meaningful for these: a claim with no judgments
        would score zero for every system and drag every average toward zero
        without distinguishing between them.
        """
        return [self.claims[cid] for cid in self.qrels if cid in self.claims]


def split_sentences(text: str) -> tuple[str, ...]:
    """Split an abstract into sentences.

    SciFact's corpus ships the abstract already split, so this is only a fallback
    for free text arriving from the API. Deliberately simple: a real sentence
    splitter is a dependency and a source of drift, and the rationale indices only
    have to line up with SciFact's own splitting for the corpus itself.
    """
    out: list[str] = []
    current: list[str] = []
    for token in text.replace("\n", " ").split(" "):
        if not token:
            continue
        current.append(token)
        if token.endswith((".", "?", "!")) and len(token) > 2:
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))
    return tuple(out)


def load_dataset(root: str | Path, split: str = "test") -> Dataset:
    """Read a BEIR-format SciFact directory."""
    root = Path(root)
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{split}.tsv"

    for path in (corpus_path, queries_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Fetch the data first:\n"
                "  warrant fetch    (or see the README)"
            )

    documents: dict[str, Document] = {}
    with open(corpus_path, encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            text = raw.get("text", "")
            documents[str(raw["_id"])] = Document(
                doc_id=str(raw["_id"]),
                title=raw.get("title", ""),
                text=text,
                sentences=split_sentences(text),
            )

    claims: dict[str, Claim] = {}
    with open(queries_path, encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            claim = Claim(claim_id=str(raw["_id"]), text=raw.get("text", ""))
            # metadata maps an abstract id to the annotations against it. An empty
            # object means the claim has no evidence either way.
            for doc_id, annotations in (raw.get("metadata") or {}).items():
                for annotation in annotations:
                    label = _LABEL_MAP.get(annotation.get("label", ""))
                    if label is None:
                        continue
                    claim.evidence.append(
                        Evidence(
                            doc_id=str(doc_id),
                            verdict=label,
                            sentence_indices=tuple(annotation.get("sentences", [])),
                        )
                    )
            claims[claim.claim_id] = claim

    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path, encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header and header[0].strip() != "query-id":
            # Some mirrors ship without a header row; do not lose the first line.
            handle.seek(0)
            reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            claim_id, doc_id, score = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                relevance = int(score)
            except ValueError:
                continue
            if relevance > 0:
                qrels.setdefault(claim_id, {})[doc_id] = relevance

    return Dataset(documents=documents, claims=claims, qrels=qrels)


def summarise(dataset: Dataset) -> str:
    judged = dataset.claims_with_qrels()
    verdicts: dict[Verdict, int] = {}
    for claim in dataset.claims.values():
        verdicts[claim.verdict] = verdicts.get(claim.verdict, 0) + 1
    parts = ", ".join(f"{k.value.lower()} {v}" for k, v in sorted(verdicts.items()))
    return (
        f"{len(dataset.documents):,} abstracts, {len(dataset.claims):,} claims, "
        f"{len(judged)} with human relevance judgments\nclaim verdicts: {parts}"
    )
