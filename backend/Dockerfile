# GeorgeBot backend — FastAPI RAG serving.
# Railway builds this with Root Directory = backend, so the build context is
# backend/ and the app files (api.py, chatbot.py, graph_queries.py) land at /app.
#
# NOTE: the serving artifacts (chroma_db/, graph_data/, vector_taxonomy.json)
# are gitignored and NOT baked into this image — they arrive at runtime via a
# Railway Volume. chatbot.py currently expects them under /app (BASE_DIR =
# Path(__file__).parent), so mount the volume accordingly (or land the pending
# env-var path override first). See CLAUDE.md "Data" / "Known issues".

# ---- builder: compile deps into an isolated venv --------------------------
FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1

# build-essential covers any dependency that compiles from source (e.g. the
# C++ extension in chromadb's hnswlib) when no prebuilt wheel is available.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- runtime: slim image with just the venv + app code --------------------
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# libgomp1 is the OpenMP runtime that chromadb's native deps (hnswlib /
# onnxruntime) link against at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .

# Railway injects $PORT; api.py reads it and binds 0.0.0.0. EXPOSE just
# documents the local-dev default.
EXPOSE 5001
CMD ["python", "api.py"]
