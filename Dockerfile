# Docs Assistant - production image (CPU only)
#   docker build -t docs-assistant .
#   docker run --env-file .env -p 8000:8000 -v docs-data:/app/data docs-assistant
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# 1) CPU-only PyTorch first. The default PyPI build bundles GPU (CUDA) libraries worth
#    several GB that this app never uses; installing it first means requirements.txt
#    finds torch already satisfied and doesn't pull the GPU build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# 2) Dependencies before code: this layer is cached and only rebuilt when
#    requirements.txt changes, so code edits rebuild in seconds.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Download the embedding model at BUILD time, so the container starts fast and
#    works offline. HF_HUB_OFFLINE then blocks any network call to the model hub.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HUB_OFFLINE=1

# 4) App code last (changes most often).
COPY app ./app

# 5) Run as a normal user, not root: if the app were ever compromised, the attacker
#    can't modify the system. /app/data is where the volume is mounted.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/uploads /app/data/chroma \
    && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000

# Docker marks the container "unhealthy" if /health stops answering.
# (python is used because the slim image has no curl.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# 0.0.0.0 = accept connections from outside the container (127.0.0.1 would only allow
# connections from inside it). The API key arrives at runtime as an environment variable.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
