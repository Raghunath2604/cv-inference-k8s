"""
Unit tests for the FastAPI CV service.

Uses a fake model (no real trained weights) so tests run in
milliseconds and don't depend on artifacts/model.pt existing.
"""
import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src import api
from src.model import CLASSES


class FakeModel:
    """Always predicts 'circle' with high confidence — enough to
    exercise request parsing, response shape, and error handling
    without a real forward pass."""

    def __call__(self, tensor):
        logits = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
        return logits.repeat(tensor.shape[0], 1)

    def eval(self):
        return self


def _make_test_png_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), (200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())
    with TestClient(api.app) as c:
        yield c
    api._state["model"] = None


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["classes"]) == set(CLASSES)


def test_predict_returns_expected_shape(client):
    png_bytes = _make_test_png_bytes()
    resp = client.post("/predict", files={"file": ("test.png", png_bytes, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "circle"  # FakeModel always favors class 0
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["scores"].keys()) == set(CLASSES)
    assert body["latency_ms"] >= 0


def test_predict_rejects_non_image(client):
    resp = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert resp.status_code == 422


def test_predict_rejects_missing_file(client):
    resp = client.post("/predict")
    assert resp.status_code == 422


def test_predict_returns_503_when_model_not_loaded(client):
    api._state["model"] = None
    png_bytes = _make_test_png_bytes()
    resp = client.post("/predict", files={"file": ("test.png", png_bytes, "image/png")})
    assert resp.status_code == 503


def test_metrics_endpoint_exposes_prometheus_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "inference_latency_seconds" in resp.text
    assert "predictions_total" in resp.text


def test_predict_records_metrics(client):
    png_bytes = _make_test_png_bytes()
    client.post("/predict", files={"file": ("test.png", png_bytes, "image/png")})
    metrics_text = client.get("/metrics").text
    assert 'predictions_total{predicted_class="circle"}' in metrics_text
