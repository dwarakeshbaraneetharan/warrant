"""Tests for citation grounding, which is the safety property of the whole system.

None of these call an LLM. The point is that grounding is enforced in code after
generation, so it can be tested by handing `validate` an answer a model might
plausibly produce — including the dangerous ones, where a fabricated citation makes
a wrong answer look authoritative.
"""

from __future__ import annotations

import pytest

from warrant.corpus import Document, Verdict, split_sentences
from warrant.retrieval import Hit
from warrant.verdict import Answer, LlmClient, build_prompt, parse_citations, validate

DOCUMENTS = {
    "100": Document(
        doc_id="100",
        title="Vitamin D and respiratory infection",
        text="Vitamin D was measured in 500 adults. Deficiency doubled infection risk.",
        sentences=(
            "Vitamin D was measured in 500 adults.",
            "Deficiency doubled infection risk.",
        ),
    ),
    "200": Document(
        doc_id="200",
        title="Unrelated work",
        text="Yeast ribosome assembly proceeds in stages.",
        sentences=("Yeast ribosome assembly proceeds in stages.",),
    ),
}
HITS = [Hit("100", 0.9, "rerank"), Hit("200", 0.1, "rerank")]


def answer(verdict: Verdict, reasoning: str) -> Answer:
    return Answer(verdict=verdict, reasoning=reasoning, citations=parse_citations(reasoning))


def test_citation_parsing():
    cites = parse_citations("Deficiency raises risk [100:1]. Confirmed again [100:1] [200:0].")
    assert [(c.doc_id, c.sentence_index) for c in cites] == [("100", 1), ("200", 0)]


def test_citation_parsing_ignores_other_brackets():
    assert parse_citations("No citation here [see above] or [abc:1].") == []


def test_a_well_cited_answer_survives():
    result = validate(
        answer(Verdict.SUPPORTED, "Deficiency doubled infection risk [100:1]."),
        DOCUMENTS,
        HITS,
    )
    assert result.verdict is Verdict.SUPPORTED
    assert result.warning is None
    assert result.grounded


def test_an_uncited_verdict_is_downgraded():
    """The most common failure: a confident answer with no source at all."""
    result = validate(answer(Verdict.SUPPORTED, "This is clearly true."), DOCUMENTS, HITS)
    assert result.verdict is Verdict.NOT_ENOUGH_INFO
    assert result.warning is not None
    assert not result.grounded


def test_a_fabricated_abstract_is_rejected():
    """The dangerous failure: a citation that looks real and points nowhere."""
    result = validate(answer(Verdict.SUPPORTED, "Proven by trial [99999:0]."), DOCUMENTS, HITS)
    assert result.verdict is Verdict.NOT_ENOUGH_INFO
    assert "never retrieved" in (result.warning or "")


def test_a_citation_to_an_abstract_that_exists_but_was_not_retrieved_is_rejected():
    """Grounding is relative to what this answer actually saw."""
    only_one = [Hit("100", 0.9, "rerank")]
    result = validate(
        answer(Verdict.REFUTED, "Contradicted by other work [200:0]."), DOCUMENTS, only_one
    )
    assert result.verdict is Verdict.NOT_ENOUGH_INFO


def test_a_sentence_index_past_the_end_is_rejected():
    result = validate(
        answer(Verdict.SUPPORTED, "Stated in the abstract [100:9]."), DOCUMENTS, HITS
    )
    assert result.verdict is Verdict.NOT_ENOUGH_INFO
    assert "does not exist" in (result.warning or "")


def test_a_partially_cited_answer_is_flagged_but_kept():
    """A citation gap is a warning, not grounds for discarding real evidence."""
    result = validate(
        answer(
            Verdict.SUPPORTED,
            "Deficiency doubled infection risk [100:1]. This is broadly accepted.",
        ),
        DOCUMENTS,
        HITS,
    )
    assert result.verdict is Verdict.SUPPORTED
    assert "without a citation" in (result.warning or "")
    assert not result.grounded


def test_declining_needs_no_citation():
    result = validate(
        answer(Verdict.NOT_ENOUGH_INFO, "The abstracts do not address this."),
        DOCUMENTS,
        HITS,
    )
    assert result.verdict is Verdict.NOT_ENOUGH_INFO
    assert result.warning is None


def test_prompt_numbers_every_sentence():
    """The model can only cite a sentence if the prompt gave it a stable index."""
    prompt = build_prompt("Vitamin D matters.", HITS, DOCUMENTS)
    assert "[100:0]" in prompt
    assert "[100:1]" in prompt
    assert "[200:0]" in prompt
    assert "Claim: Vitamin D matters." in prompt


def test_prompt_skips_hits_with_no_document():
    prompt = build_prompt("x", [Hit("missing", 1.0)], DOCUMENTS)
    assert "missing" not in prompt


def test_sentence_splitting_matches_indices():
    sentences = split_sentences("First one. Second one! Third one?")
    assert len(sentences) == 3
    assert sentences[1].startswith("Second")


def test_sentence_splitting_handles_no_terminator():
    assert split_sentences("no final period") == ("no final period",)


def test_client_reports_when_it_has_no_key():
    assert not LlmClient(api_key="").available
    assert LlmClient(api_key="abc").available


def test_client_refuses_to_call_without_a_key():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        LlmClient(api_key="").complete("system", "user")
