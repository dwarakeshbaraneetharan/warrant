# Decisions

Why this is built the way it is, including the choices that turned out wrong.

---

## Why a benchmark instead of my own documents

A RAG demo over a folder of PDFs cannot be evaluated. You can ask whether an answer
*looks* faithful, but not whether retrieval found the right document, because nobody
ever wrote down which document was right. Every conclusion becomes a vibe.

SciFact ships human relevance judgments, so retrieval can be scored with standard IR
metrics and the ablation in the README means something. The cost is that the corpus
is not mine and is only 5,183 abstracts. That trade is worth it: a smaller corpus
with labels supports claims that a larger one without them cannot.

I considered generating an eval set with an LLM — write a question for each chunk,
then check the chunk is retrieved. It is a reasonable technique and RAGAS does it.
But it makes retrieval circular: the question is written *from* the chunk, so it
inherits its vocabulary, and the retriever is graded on finding text the question was
copied from. Scores come out high and mean little.

## Why hybrid retrieval

BM25 and embeddings fail differently, and the failures are close to independent.

BM25 matches tokens, so it is unbeatable on rare precise terms — a claim mentioning
`25-hydroxyvitamin D` will find the abstract containing that exact string — and it is
blind to paraphrase. Embeddings are the reverse: they match *reduces the risk of* to
*is protective against*, and they dilute rare tokens into a 384-dimensional average
where a single distinctive term barely moves the vector.

Measured, alone: BM25 0.662 nDCG@10, dense 0.645. Fused: 0.691. The fusion beats
both, which is only worth having because the errors are not the same errors.

## Why reciprocal rank fusion instead of weighted scores

The obvious approach is `alpha * bm25 + (1 - alpha) * cosine`. It has a hidden
problem: BM25 scores are unbounded sums whose magnitude depends on corpus statistics,
and cosine similarity is bounded in [-1, 1]. Any weighted sum of the two is really a
weighted sum of two arbitrary scales, so `alpha` does not mean what it looks like it
means, and its correct value drifts as the corpus changes.

RRF uses only positions, so there is no scale to calibrate and no way for one
retriever's score distribution to quietly take over. `k = 60` is the value from the
original paper; changing it did not beat that here.

There is a real cost: RRF discards score magnitude, so it cannot tell a rank-1 hit
the retriever was certain about from a rank-1 hit it barely preferred. For this
corpus that has not mattered.

## Why MedCPT rather than the default reranker

**This is the decision the project turns on, and I got it wrong first.**

I reached for `cross-encoder/ms-marco-MiniLM-L-6-v2` because it is what every RAG
tutorial uses. Measured on 300 claims, it buys **+0.005 nDCG@10 for 11.8x the
latency**. It does essentially nothing while making the system an order of magnitude
slower.

The reason is domain. MS MARCO is web search: short keyword-ish queries against web
passages. This task is a scientific assertion against a biomedical abstract. The
model is confidently out of distribution.

`ncbi/MedCPT-Cross-Encoder` is a comparable-size model trained on PubMed search logs.
Same pipeline, same rough parameter count, only the training data differs:
**+0.095 nDCG@10**.

Two things follow. The generic advice "add a cross-encoder reranker" is not wrong but
is badly underspecified — nearly all the available gain lives in the domain match,
not the architecture. And a reranker is worth 48x the latency here only because it is
the *right* reranker; with the default one the same latency buys nothing.

## Why the candidate set is 25 in the app and 50 in the benchmark

The reranker costs one forward pass per candidate, so this is a straight
latency-for-quality dial. The benchmark uses 50 to give it the best shot. The app
uses 25 because 50 puts an interactive request near two seconds.

The ablation makes the trade safe: Recall@50 is 0.940 *before* reranking, so the
abstracts that matter are nearly always inside the first 25 anyway. Reranking cannot
add recall, only reorder — which is also why all three reranked rows share the same
Recall@50.

## Why citations are checked in code

The prompt asks for `[abstract_id:sentence_index]` on every sentence. Prompts are
requests, not guarantees, and the specific failure that matters here is a *plausible
fabricated citation*: an answer that is wrong and looks sourced is worse than one
that is obviously unsupported.

So markers are parsed and verified against what was actually retrieved. Citing an
abstract that was not retrieved, or a sentence index past the end of the abstract,
downgrades the answer. Asserting without citing flags it.

Sentences are numbered explicitly in the prompt rather than left to the model to
count, because a citation is only checkable if both sides agree on what sentence 3
is.

## Why NOT_ENOUGH_INFO is a real answer

416 of SciFact's 1,109 claims have no supporting evidence in the corpus. A system
that always produces a verdict is confidently wrong on 38% of the workload, and that
is invisible unless declining is something you measure.

It is also the honest behaviour for the actual use case. "The literature here does not
settle this" is frequently the true answer to a question about health research, and a
tool that cannot say it is worse than no tool.

## Why the index is in memory

5,183 abstracts is about 8 MB of float32 vectors plus an inverted index. A vector
database would add a service to run, a schema to migrate, and a network hop per
query, in exchange for nothing at this size.

This is the wrong decision two orders of magnitude up, where the index stops fitting
and rebuilding on deploy stops being acceptable. Worth stating explicitly rather than
discovering later: the choice is right *because* the corpus is small, not because
in-memory is better.

## Why Groq, and why one HTTP call

Groq's free tier is fast enough to keep a demo interactive, which matters more than
model quality for the verdict step — the hard part is retrieval, and the generator is
mostly reading. Temperature is zero so evaluation is repeatable.

The call is one `httpx.post` rather than the vendor SDK. It is a dozen lines, has no
version drift, and puts the timeout and error handling where they can be seen instead
of inheriting a library default.

## Why the frontend has no framework

The page is one form and one result list. React plus a build step would be more code
than the thing it renders, and it would add a deployment stage to a container that
currently copies three static files. If it grew a second view, that calculus would
change.

## Things I would do differently with more time

- Report verdict accuracy and rationale-level citation precision. The labels are
  right there; the harness needs an LLM in the loop, so it does not belong in CI.
- Distil the reranker, or rerank only the top 10, to get the quality without the 1.4s.
- Test the domain-match finding on a second corpus. One dataset supports "match the
  reranker to the domain" as a hypothesis, not as a law.
