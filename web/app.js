// No framework on purpose: the page is one form and one list, and a build step
// would be more machinery than the thing it renders.

const form = document.getElementById("form");
const claimInput = document.getElementById("claim");
const rerankerInput = document.getElementById("reranker");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const verdictEl = document.getElementById("verdict");
const reasoningEl = document.getElementById("reasoning");
const warningEl = document.getElementById("warning");
const timingEl = document.getElementById("timing");
const sourcesEl = document.getElementById("sources");
const errorEl = document.getElementById("error");
const examplesEl = document.getElementById("examples");
const exampleButtons = document.getElementById("exampleButtons");

const VERDICT_LABEL = {
  SUPPORTED: "Supported by the literature",
  REFUTED: "Refuted by the literature",
  NOT_ENOUGH_INFO: "Not enough evidence",
};

function show(el, visible) {
  el.hidden = !visible;
}

function fail(message) {
  errorEl.textContent = message;
  show(errorEl, true);
}

/** Poll until the index finishes building, so a cold container is explained. */
async function waitForReady() {
  for (let attempt = 0; attempt < 120; attempt++) {
    try {
      const response = await fetch("/api/health");
      const health = await response.json();
      if (health.error) {
        statusEl.textContent = `Index failed: ${health.error}`;
        return null;
      }
      if (health.status === "ok") {
        const generation = health.generation_available
          ? "retrieval and verdicts"
          : "retrieval only — no generation key configured";
        statusEl.textContent = `${health.abstracts.toLocaleString()} abstracts indexed in ${health.indexed_seconds}s · ${generation}`;
        return health;
      }
      statusEl.textContent = "Building the index (first load only)…";
    } catch {
      statusEl.textContent = "Waiting for the server…";
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
  statusEl.textContent = "Server did not become ready.";
  return null;
}

async function loadExamples() {
  try {
    const response = await fetch("/api/examples");
    const { examples } = await response.json();
    if (!examples?.length) return;

    for (const example of examples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example";
      // Naming the expected verdict up front means a visitor can see the system
      // decline as readily as agree, which is the behaviour worth showing.
      button.textContent = `${example.expected.replace(/_/g, " ").toLowerCase()} · ${example.claim}`;
      button.addEventListener("click", () => {
        claimInput.value = example.claim;
        form.requestSubmit();
      });
      exampleButtons.appendChild(button);
    }
    show(examplesEl, true);
  } catch {
    // Examples are a convenience; their absence should not look like a failure.
  }
}

function renderSources(sources) {
  sourcesEl.replaceChildren();

  for (const source of sources) {
    const cited = new Set(source.cited ?? []);
    const item = document.createElement("li");
    if (cited.size) item.classList.add("cited");

    const head = document.createElement("div");
    head.className = "source-head";

    const title = document.createElement("span");
    title.className = "source-title";
    title.textContent = source.title || `Abstract ${source.doc_id}`;

    const meta = document.createElement("span");
    meta.className = "source-meta";
    meta.textContent = `${source.stage} ${source.score.toFixed(3)}`;

    head.append(title, meta);
    item.append(head);

    const body = document.createElement("div");
    source.sentences.forEach((sentence, index) => {
      const span = document.createElement("span");
      span.className = cited.has(index) ? "sentence cited" : "sentence";
      span.textContent = sentence + " ";
      body.append(span);
    });
    item.append(body);
    sourcesEl.append(item);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const claim = claimInput.value.trim();
  if (claim.length < 3) return;

  show(errorEl, false);
  submitButton.disabled = true;
  submitButton.textContent = "Checking…";

  try {
    const response = await fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claim,
        top_k: 5,
        use_reranker: rerankerInput.checked,
      }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      fail(detail.detail || `Request failed (${response.status})`);
      return;
    }

    const data = await response.json();

    verdictEl.textContent = VERDICT_LABEL[data.verdict] ?? data.verdict;
    verdictEl.dataset.v = data.verdict;
    reasoningEl.textContent = data.reasoning || "";

    if (data.warning) {
      warningEl.textContent = data.warning;
      show(warningEl, true);
    } else {
      show(warningEl, false);
    }

    const parts = [`retrieval ${data.retrieval_ms} ms`];
    if (data.generation_ms) parts.push(`generation ${data.generation_ms} ms`);
    if (data.model) parts.push(data.model);
    if (data.reasoning) parts.push(data.grounded ? "fully cited" : "citation gap");
    timingEl.textContent = parts.join(" · ");

    renderSources(data.sources ?? []);
    show(resultEl, true);
  } catch (error) {
    fail(`Could not reach the server: ${error.message}`);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Check claim";
  }
});

waitForReady().then((health) => {
  if (health) loadExamples();
});
