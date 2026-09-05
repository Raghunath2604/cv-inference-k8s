# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
WORKDIR /build

COPY requirements.txt .
# CPU-only torch first — see requirements.txt comment for why this is
# separate from the rest of the install. Deliberately kept as two RUN
# layers rather than merged (hadolint DL3059 flags this as
# consolidatable) so the torch layer stays cached across rebuilds where
# only requirements.txt changes.
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY artifacts/model.pt ./artifacts/model.pt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.pt \
    PYTHONPATH=/app

USER 1000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else sys.exit(1)"]

# Single uvicorn worker per container — horizontal scaling is handled
# by Kubernetes running more pod replicas (see k8s/hpa.yaml), not by
# running multiple workers inside one container.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
