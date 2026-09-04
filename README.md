# CV Inference Service — Kubernetes, Autoscaling & Monitoring Reference Project

A small computer-vision model (circle/square/triangle/star classifier)
served behind FastAPI, packaged for Kubernetes with autoscaling,
Prometheus/Grafana monitoring, and load testing in both Python
(Locust) and Java. Backs the "Scalable Real-Time Inference System with
Kubernetes & Monitoring Stack" project on my resume.

This repo was rebuilt and re-verified from scratch in a fresh dev
sandbox (the original one this was first built in was reset partway
through an earlier work session) — every number below was checked
again in this exact environment, not assumed to still hold from
before.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
pip install -r requirements-dev.txt

python src/generate_data.py
python -m src.train --epochs 20   # ~3 min on CPU; BatchNorm gets you to ~93% val accuracy

uvicorn src.api:app --reload --port 8000
curl -X POST localhost:8000/predict -F "file=@data/val/star/star_0000.png"

locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 25s

cd loadtest/java && javac LoadTestClient.java
java LoadTestClient http://localhost:8000 ../../data/val/star/star_0000.png 100 10
```

### Docker / full local stack

```bash
docker compose up --build
```
See "What's actually verified" below for the real story on whether
this build completes in a sandboxed dev environment — the short
version: the Dockerfile is correct and lint-clean, but this sandbox
can't reach any container registry to pull the base image.

## Bugs found while building this (all reproduced fresh in this rebuild)

**1. Edge-clipping bug in synthetic image generation.** Shapes near the
canvas border could get clipped into ambiguous partial silhouettes —
first training run only hit ~65% accuracy. Fixed by computing padding
from each shape's true bounding radius (a rotated square's corners
extend to `size × √2`, not just `size`).

**2. Undertraining, not a hidden bug.** After the clipping fix,
accuracy still plateaued in the 60-80s. Verified this wasn't another
bug by training longer and watching accuracy keep climbing rather than
stall. Added BatchNorm — real fix for slow convergence — bringing
final validation accuracy to **93.4%** (reproduced fresh in this
rebuild, consistent with the original build's ~93%).

**3. `/predict` was silently head-of-line-blocking under concurrent
load.** `async def` with a synchronous, CPU-bound `model(tensor)` call
inside blocks the whole event loop. Found via a real Locust run (max
latency 3.3s, p99.9 2.4s at just 20 concurrent users in the original
build). Fixed by making the endpoint a plain `def` so FastAPI runs it
in a worker thread pool. This fix was applied from the start in this
rebuild and reproduced cleanly: 100/100 successful Java-client requests
with no catastrophic tail latency.

**4. The Java load-test client's HTTP requests were rejected outright**
(`400 Invalid HTTP request received.`) due to `java.net.http.HttpClient`
defaulting to an HTTP/2 cleartext-upgrade attempt that uvicorn's
HTTP/1.1-only parser doesn't handle. Fixed with an explicit
`.version(HttpClient.Version.HTTP_1_1)`. Applied from the start in this
rebuild; compiled and ran clean immediately.

**5. New this rebuild — a load test result that looked like a
regression, but wasn't.** A Java-client run right after this sandbox's
VM booted showed p95 latency spiking to 2.4 seconds. Rather than report
that as a code problem, checked `uptime` (the VM had booted 2 minutes
earlier) and process state (still in disk-I/O wait) before concluding
anything. Re-ran immediately after: p95 dropped to 246ms, max 329ms.
The first run was genuine environment noise from a still-settling VM,
not a regression — both runs are recorded honestly in
`loadtest/results/java_client_results.txt` rather than only keeping
the good one.

**6. Stale Docker daemon state caused a false failure.** After an
earlier `dockerd` attempt in this same session, retrying the daemon
start failed with "timeout waiting for containerd to start." Traced it
to leftover socket/pid files in `/run/docker/` from the dead prior
instance (confirmed via `ps aux` that no process was actually running)
rather than assuming Docker itself was broken. Cleared the stale state
and the daemon started cleanly in 2 seconds.

**7. New this session — tested the actual "500+ concurrent users" claim
directly, and it doesn't hold for a single instance.** All earlier load
testing in this repo used 20 concurrent users. Asked directly whether
the resume's exact claim — sub-100ms latency at 500+ concurrent users —
holds, so ran it for real: 500 concurrent Locust users against one
unscaled instance on this sandbox's single CPU core. Result: **0%
request failures, but median latency ~1.9s, p99 ~3.3s** — nowhere near
sub-100ms. Checked this wasn't a code regression by testing the
progression at 1/10/50/100/500 users: latency degrades smoothly and
throughput plateaus at ~210 req/s starting around 50 users, the
textbook signature of single-core CPU saturation, not a bug (see the
full table below).

| Concurrent users | Median latency | p99 | Throughput |
|---|---|---|---|
| 1 | 8ms | 31ms | 5 req/s |
| 10 | 7ms | 45ms | 57 req/s |
| 50 | 43ms | 260ms | 212 req/s |
| 100 | 270ms | 590ms | 212 req/s |
| 500 | ~1900ms | ~3300ms | 190 req/s |

**What this means honestly:** the resume's sub-100ms-at-500-users claim
is a claim about the *full system* — Kubernetes HPA actually spreading
500 concurrent users across up to 10 pods on real multi-core
infrastructure (see `k8s/hpa.yaml`). What's been measured here is the
ceiling of *one* replica on *one* core, which is precisely the
scenario HPA exists to prevent, not a flaw in the service itself — the
service degrades gracefully (zero failures, just queueing) rather than
crashing under overload, which is itself the correct behavior. But the
composite claim — 500 users, sub-100ms, via working autoscaling — has
not been verified end-to-end, because that requires a real multi-node
cluster this sandbox cannot reach (see the Docker/Kubernetes section
below). This is now measured and documented rather than assumed either
way.

## What's actually verified vs. not

**Verified by actually running it, in this exact environment:**
- Full training pipeline — real 93.4% validation accuracy on this
  rebuild, matching the original build.
- FastAPI service as a real subprocess — `/health`, `/predict` (real
  images, real HTTP), `/metrics`, error handling.
- **12/12 pytest tests pass** on a simulated fresh checkout.
- **A single instance's real load-scaling ceiling, measured directly**:
  0% failures from 1 to 500 concurrent users; latency stays under
  ~50ms through 10 users, then degrades smoothly as one CPU core
  saturates, plateauing at ~210 req/s throughput from 50 users onward.
  At 500 users: median ~1.9s, p99 ~3.3s — see bug #7 above for the full
  table and what this does and doesn't tell you about the resume's
  500-user claim.
- Java client: installed the JDK (this sandbox only ships the JRE by
  default), compiled clean (both known bugs pre-fixed), ran two full
  load tests with 100/100 success each.
- **Kubernetes manifests validated against the actual Kubernetes
  Python client's schema classes** (`V1Deployment`,
  `V2HorizontalPodAutoscaler`, `V1Service`).
- **`monitoring/prometheus.yml` validated with the real `promtool`**
  (downloaded the actual v3.13.1 release binary).
- **All 8 PromQL expressions in the Grafana dashboard validated** with
  `promtool`'s real query parser; metric/label names cross-checked
  against what `src/api.py` actually exposes.
- **The Dockerfile was linted with the real `hadolint` binary** — clean
  except for one intentionally-documented layer-caching tradeoff.
- **`docker-compose.yml` validated with the real `docker compose`
  binary** (downloaded from Docker's GitHub releases since neither the
  apt plugin nor the old pip package worked) — `docker compose config`
  resolves it with zero warnings after fixing one real issue it found
  (an obsolete top-level `version` field).
- **Docker itself was actually installed and its daemon actually
  started** (`apt-get install docker.io`, `dockerd`). `docker build`
  was attempted for real against this exact Dockerfile.

**Not verified — and here's the precise reason why, not just "couldn't
test it":**
- **The resume's exact composite claim — 500+ concurrent users AND
  sub-100ms latency, together, via HPA — has not been verified
  end-to-end, and based on direct measurement, a single replica
  genuinely cannot deliver both at once** (see bug #7 above: ~1.9s
  median at 500 users on one core). The architecture is sound — that's
  exactly what horizontal autoscaling across multiple pods/cores is
  for — but proving it requires a real multi-node cluster actually
  scaling this service out, which this sandbox cannot reach (next
  point). Treat the resume bullet as describing the intended,
  architected behavior of the full system, not something proven here.
- `docker build` does not complete. The daemon runs correctly and the
  build process starts correctly, but it fails at the very first step
  — pulling the `python:3.12-slim` base image — because **every major
  container registry this sandbox tried (Docker Hub, GHCR, GCR, ECR
  public) returns 403 Forbidden** from this network's egress policy.
  This was tested directly, not assumed.
- `kubectl apply` / a real cluster. Also tested directly: `kubectl`'s
  own distribution domains (`dl.k8s.io`, `storage.googleapis.com`) are
  also blocked. Even if the binary were obtained another way, a local
  cluster (kind/minikube) would need to pull node images from the
  already-confirmed-blocked registries, so there's no viable path to a
  real cluster in this specific sandbox.
- HPA actually scaling pods under load, or Prometheus's pod discovery
  actually finding pods — both require a running cluster.

To close these gaps: run `docker compose up --build` and
`kubectl apply -f k8s/` in an environment with normal internet access
(your own machine, or GitHub Actions, both of which reach these
registries fine) — the Dockerfile and manifests themselves have been
checked as thoroughly as this sandbox allows.

## Design decisions worth knowing

- **Why synthetic shape images?** Zero external downloads needed to
  run the pipeline.
- **Why train from scratch instead of transfer learning?** Pretrained
  weights download from `download.pytorch.org` at runtime — avoided by
  design, same reasoning as the sibling text-classification project.
- **Why a plain-JDK HTTP client instead of Spring Boot for "Java"?**
  Maven Central isn't reachable from this sandbox, so a Spring Boot
  service could be written but not compiled/verified here. The
  plain-JDK client is fully compiled, run, and debugged instead — a
  deliberate "verified over impressive" tradeoff.
- **Why does the Deployment default to 2 replicas, not 1?** So a
  single pod restart or node drain doesn't cause a full outage.
- **Known gap:** CPU-utilization-based HPA is a proxy for the real
  goal (sub-100ms latency); a request-latency-based custom metric via
  the Prometheus Adapter would be more direct, but verifying that setup
  needs a real cluster, which — as documented above — this sandbox
  genuinely cannot reach.

## Testing

```bash
pytest tests/ -v
ruff check src/ tests/
```

## Additional local commands

Run the dependency-free Python replacement for the Java load-test client (added in this rebuild):

```bash
python -m venv .venv && . .venv/bin/activate   # or ".\.venv\Scripts\activate" on Windows
pip install -r requirements-dev.txt

# Run the Python load test (200 requests, concurrency 50)
.venv\Scripts\python.exe loadtest\python_load_test.py http://127.0.0.1:8000 data\val\star\star_0000.png 200 50

# Or run Locust headless (mix of health + predict):
.venv\Scripts\python.exe -m locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 --headless -u 20 -r 5 -t 25s
```

Notes on pushing to a remote repository

- I did not push changes to any remote. If you provide a GitHub repo URL and credentials, I can push the current branch. Alternatively, to push from your machine:

```bash
git add loadtest/python_load_test.py README.md
git commit -m "Add Python load-test client and README instructions"
git remote add origin <your-repo-url>
git push -u origin main
```

If you'd like, I can prepare a short `CONTRIBUTING.md` or an abbreviated release checklist before you push.
