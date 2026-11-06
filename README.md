# CV Inference Service — Kubernetes, Autoscaling & Monitoring Reference Project

A small computer-vision model (circle/square/triangle/star classifier)
served behind FastAPI, packaged for Kubernetes with autoscaling,
Prometheus/Grafana monitoring, and load testing in both Python
(Locust) and Java. Backs the "Scalable Real-Time Inference System with
Kubernetes & Monitoring Stack" project on my resume.

This repo was rebuilt and re-verified from scratch in a fresh dev
sandbox — every number below was checked again in this exact environment.

## Quickstart (local)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
pip install -r requirements-dev.txt

python src/generate_data.py
python -m src.train --epochs 20
uvicorn src.api:app --reload --port 8000
curl -X POST localhost:8000/predict -F "file=@data/val/star/star_0000.png"

locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 25s

cd loadtest/java && javac LoadTestClient.java
java LoadTestClient http://localhost:8000 ../../data/val/star/star_0000.png 100 10
```

The API resolves the default model path relative to the repository. If
`artifacts/model.pt` is missing, startup fails with an actionable error.
The Docker build generates data and trains the model in a disposable
builder stage, so a clean checkout produces a runnable image without
committed data or weights. When `API_KEY` is configured, send it as
`X-API-Key` on `/predict`; `/health` remains unauthenticated for probes.

### Docker / full local stack

```bash
docker compose up --build
```

The Compose stack starts the API first, waits for its health check, then
starts Prometheus and Grafana. The API is available on port 8000,
Prometheus on 9090, and Grafana on 3000.

Set `GRAFANA_ADMIN_PASSWORD` before starting Compose. Set `API_KEY` to
protect `/predict`; Kubernetes supplies it from the `cv-inference-api`
Secret. Create that Secret before applying the manifests:

```bash
kubectl create secret generic cv-inference-api \
  --from-literal=api-key="$(openssl rand -hex 32)"
```

The Kubernetes deployment intentionally uses a release-tagged image
placeholder rather than `latest`; replace it with the immutable image
reference produced by your registry pipeline before applying.

## Production hardening

- API-key authorization with constant-time comparison.
- Upload-size limits and malformed-image rejection.
- Safe PyTorch weight-only model loading.
- Non-root, read-only Kubernetes containers with dropped capabilities.
- Disabled service-account token mounting and RuntimeDefault seccomp.
- PodDisruptionBudget and topology spreading.
- Kubernetes NetworkPolicy and resource requests/limits.
- Health-gated Compose startup and protected local monitoring ports.
- CI linting, tests, Docker build, Compose validation, and Kubernetes schema validation.

## Verified results

- Docker image builds from a clean checkout and embeds a trained model.
- Best validation checkpoint reached approximately 94.69% in the verified build.
- `/health`, `/metrics`, and real multipart `/predict` requests succeeded.
- 15/15 pytest tests passed inside the Linux image.
- Java client completed 100/100 requests successfully in the verified run.
- Single-replica load testing showed zero request failures through 500 concurrent users, with latency increasing after CPU saturation.

## Kubernetes deployment

```bash
kubectl apply -f k8s/serviceaccount.yaml
kubectl apply -f k8s/pdb.yaml
kubectl apply -f k8s/networkpolicy.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl rollout status deployment/cv-inference
```

Live HPA behavior, Prometheus pod discovery, and multi-replica latency
require a reachable Kubernetes cluster and are not claimed as locally
verified.

## Testing

```bash
pytest tests/ -v
ruff check src/ tests/
```

## Honest limitations

The complete autoscaled 500-user/sub-100ms claim requires a real cluster
with multiple replicas and metrics-server. The repository includes the
manifests and monitoring configuration, but cluster behavior must be
validated in the target environment.
