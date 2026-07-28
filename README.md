# Warrant

**Checks a scientific claim against the literature, and measures whether it
retrieved the right papers.**

[![CI](https://github.com/dwarakeshbaraneetharan/warrant/actions/workflows/ci.yml/badge.svg)](https://github.com/dwarakeshbaraneetharan/warrant/actions/workflows/ci.yml)

Paste a claim like *"Vitamin D deficiency increases the risk of respiratory
infection"*. Warrant searches 5,183 research abstracts, decides whether the
literature **supports**, **refutes**, or **does not settle** it, and cites the exact
sentence behind every statement it makes. If a sentence has no citation, the answer
is rejected rather than shown.

In Toulmin's model of argument, the *warrant* is what connects evidence to a claim.
That is the thing this produces.

> **Live demo:** _deploying — see [Running it](#running-it) to run locally in two
> commands._

---

## Why this exists

Retrieval-augmented generation is easy to build and hard to trust. The standard
demo answers questions over your documents and looks convincing immediately, which
is the problem: when it is wrong, it is wrong with a citation attached, and nothing
in the demo tells you how often that happens.

Almost every portfolio RAG project scores its *generation* — is the answer
faithful, does it hallucinate. Very few score its *retrieval*, because doing that
honestly requires knowing which documents were actually relevant, and that means
human labels. Without them you cannot tell a retrieval failure from a generation
failure, which are different bugs with different fixes.

So Warrant is built on [SciFact](https://github.com/allenai/scifact), which supplies
three independent kinds of human annotation:

| label | what it gives |
|---|---|
| relevance judgments | which abstracts a human judged relevant to a claim |
| verdicts | whether the literature supports or refutes the claim |
| rationales | *which sentences* justify that verdict |

That third one is unusual and it is why the corpus was chosen. It makes it possible
to ask not "did the answer cite something" but "did it cite the sentence a human
would have cited".

## The measured result

Every stage of the pipeline, scored on all 300 claims that carry human relevance
judgments. Reproduce with `python bench/ablation.py`.

| configuration | nDCG@10 | Recall@10 | Recall@50 | MRR | sec |
|---|---|---|---|---|---|
| BM25 only | 0.662 | 0.784 | 0.870 | 0.634 | 1.8 |
| dense only | 0.645 | 0.783 | 0.892 | 0.611 | 3.9 |
| hybrid (RRF) | 0.691 | 0.822 | 0.940 | 0.659 | 4.4 |
| hybrid + **web-search** reranker | 0.697 | 0.834 | 0.940 | 0.668 | 51.8 |
| hybrid + **biomedical** reranker | **0.786** | **0.897** | 0.940 | **0.757** | 213.1 |

Three things fall out of this, and the third is the reason the project is worth
reading.

**Hybrid retrieval beats either half.** BM25 and embeddings fail differently — one
misses paraphrase, the other misses rare technical terms — so fusing them adds
+0.029 nDCG@10 over the better of the two.

**Reranking cannot improve recall, only order.** Recall@50 is 0.940 for all three
reranked rows, identical to the hybrid it started from, because a reranker only
reorders the candidates it was handed. If the right abstract is not in the candidate
set, no reranker will find it.

**The reranker's training domain matters roughly 18x more than having a reranker at
all.** The advice everywhere is "add a cross-encoder reranker". Doing that with the
default choice — `ms-marco-MiniLM-L-6-v2`, trained on web search queries — buys
**+0.005 nDCG@10 for 11.8x the latency**, which is nothing at all. Swapping it for
`MedCPT-Cross-Encoder`, a comparable-size model trained on PubMed search logs, buys
**+0.095**. Same pipeline, same rough parameter count; only the training data
differs.

That is the whole finding: the popular recipe is not wrong so much as
underspecified, and following it without measuring would have produced a system 48
times slower and no better.

A methodological note, because it nearly caught me out. On a 60-claim sample the
web-search reranker looked actively *harmful*, at −0.052 nDCG@10. On all 300 it is
+0.005. The first number was noise, and reporting it would have been a confident
claim about nothing.

## How it works

```
claim
  ├── BM25            lexical, catches exact technical terms
  ├── dense embedding semantic, catches paraphrase
  ├── reciprocal rank fusion
  ├── cross-encoder rerank   (domain-matched; see above)
  └── LLM verdict     with citations enforced in code
```

The retrieve-then-rerank shape exists because of an asymmetry. BM25 and embeddings
score a query and a document *independently*, which is what makes them cheap enough
to run over the whole corpus. A cross-encoder reads both together and is far more
accurate, but costs a forward pass per candidate, so it only ever sees the top few
dozen.

Fusion is by rank, not score. BM25 produces unbounded sums and cosine similarity
lives in [-1, 1], so any weighted average of the two is really a weighted average of
two arbitrary scales, and the weight silently changes meaning as the corpus grows.
Reciprocal rank fusion only looks at positions, so it needs no calibration.

### Citations are enforced, not requested

The prompt asks for `[abstract_id:sentence_index]` markers. That is a suggestion. So
after generation, the markers are parsed and checked, and an answer is downgraded to
*not enough evidence* if it:

- cites nothing at all,
- cites an abstract that was never retrieved,
- cites a sentence index that does not exist, or
- asserts a sentence with no citation (kept, but flagged).

A model that invents `[99999:3]` has produced exactly the authoritative-looking
nonsense this project exists to catch, so the check is mechanical rather than
trusting. There are tests for each of those four cases.

**Declining is a first-class answer.** About a third of SciFact's claims have no
supporting evidence in the corpus, so a system that always reaches a verdict is
confidently wrong on all of them. Both the prompt and the evaluation treat
*not enough evidence* as a real outcome rather than a failure.

## CI fails when retrieval gets worse

The interesting half of the pipeline can be tested without an LLM at all, which
means it can gate every push for free and deterministically:

```yaml
- run: |
    python -m warrant.cli evaluate \
      --no-reranker --min-ndcg 0.65 --min-recall 0.90
```

Change the chunking, the embedding model, or the fusion, and if quality drops the
build goes red. The thresholds sit just under the measured values so ordinary
variation does not fail the build while a genuine break does. The gate skips the
reranker deliberately: it needs a 90 MB embedding model rather than a 440 MB
cross-encoder, which keeps the job to a couple of minutes.

## Running it

```bash
pip install -e '.[dev]'
warrant fetch          # 3 MB: corpus, claims, human judgments
warrant serve          # http://127.0.0.1:8000
```

Retrieval works with no API key. For verdicts, set a free
[Groq](https://console.groq.com) key:

```bash
export GROQ_API_KEY=...
```

Score it yourself:

```bash
warrant evaluate                 # hybrid, ~15s
python bench/ablation.py         # every stage, ~5 min
```

## Layout

```
src/warrant/
  corpus.py      SciFact: abstracts, claims, and all three kinds of human label
  retrieval.py   BM25 (written out), dense index, RRF, cross-encoder rerank
  evaluation.py  nDCG, recall, MRR — implemented, not imported
  verdict.py     prompting, citation parsing, grounding enforcement
  api.py         FastAPI service and the JSON contract
  cli.py         warrant fetch / serve / evaluate
web/             the UI: one hand-written page, no build step
bench/ablation.py  the table above
tests/           33 tests
```

BM25 and the IR metrics are written out rather than imported. They are thirty lines
each, they are the baseline everything is judged against, and the argument of the
project is that the numbers can be checked without following a dependency.

## Known gaps

Honest list, roughly in the order I would fix them.

- **The reranker is slow on a free CPU.** 25 candidates through a BERT-base
  cross-encoder is about 1.4s locally and worse on shared hardware. Distilling it,
  or reranking only the top 10, is the obvious trade.
- **Verdict accuracy is not yet reported.** The retrieval numbers here are the
  measured ones. Scoring verdicts against SciFact's labels, and citations against
  its rationale sentences, needs an LLM in the loop and so is not in CI; the harness
  is the next piece of work.
- **One corpus, one domain.** Every number is SciFact. A biomedical reranker winning
  on biomedical abstracts is exactly what you would expect, and the general lesson —
  match the reranker to the domain — is the claim, not "MedCPT is best".
- **No chunking.** Abstracts are short enough to embed whole, which conveniently
  sidesteps the hardest decision in most RAG systems. A full-text corpus would not
  let me.
- **Retrieval is exact, not approximate.** A brute-force scan over 5,183 vectors is
  under a millisecond, so there is no ANN index. At a million documents there would
  have to be.
- **`GROQ_API_KEY` gates generation only.** Without it the app still retrieves and
  says so, rather than failing.

## License

MIT.
