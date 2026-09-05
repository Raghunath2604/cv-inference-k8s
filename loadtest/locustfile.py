"""
Locust load test for the /predict endpoint.

Run against a live server:
    locust -f loadtest/locustfile.py --host http://localhost:8000

Headless:
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 10 -t 60s --csv=loadtest/results/run
"""
import io
import random

from locust import HttpUser, task, between
from PIL import Image, ImageDraw

_COLORS = [(220, 60, 60), (60, 130, 220), (60, 180, 90), (230, 170, 40)]


def _make_image_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    color = random.choice(_COLORS)
    draw.ellipse([16, 16, 48, 48], fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_IMAGE_POOL = [_make_image_bytes() for _ in range(20)]


class InferenceUser(HttpUser):
    wait_time = between(0.05, 0.3)

    @task
    def predict(self):
        image_bytes = random.choice(_IMAGE_POOL)
        self.client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
            name="/predict",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
