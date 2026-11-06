# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder
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

# Build the model artifact inside the image so a clean checkout is enough
# to produce a runnable image; generated data remains confined to this
# disposable builder stage.
ENV PYTHONPATH=/install/lib/python3.11/site-packages
COPY src/ ./src/
ARG TRAIN_EPOCHS=20
RUN python -u src/generate_data.py && python -u -m src.train --epochs ${TRAIN_EPOCHS}

FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY --from=builder /build/artifacts/model.pt ./artifacts/model.pt
RUN pip install --no-cache-dir --upgrade \
    wheel==0.46.2 \
    jaraco.context==6.1.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.pt \
    APP_VERSION=1.0.0 \
    MODEL_VERSION=synthetic-shapes-v1 \
    PYTHONPATH=/app

USER 1000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else sys.exit(1)"]

# Single uvicorn worker per container — horizontal scaling is handled
# by Kubernetes running more pod replicas (see k8s/hpa.yaml), not by
# running multiple workers inside one container.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
