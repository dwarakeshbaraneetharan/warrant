# Hugging Face Spaces serves on 7860 and runs as a non-root user, which is why the
# cache locations below are set explicitly: the default ~/.cache is not writable
# there, and sentence-transformers fails at import rather than at first use.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    WARRANT_DATA=/app/data/scifact

WORKDIR /app

# Dependencies first, so a change to application code does not reinstall torch.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && mkdir -p /app/.cache && chmod -R 777 /app/.cache

COPY web ./web

# Corpus and model weights are baked in at build time. Downloading them on first
# request would make a cold start look like a hang, and the corpus is only 3 MB.
RUN python -m warrant.cli fetch --out /app/data \
 && python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('ncbi/MedCPT-Cross-Encoder')" \
 && chmod -R 777 /app/.cache /app/data

EXPOSE 7860

# One worker on purpose: the index lives in process memory, so a second worker
# would double a 400 MB footprint to serve the same read-only data.
CMD ["sh", "-c", "uvicorn warrant.api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
