"""
Generates a small synthetic image-classification dataset: 64x64 RGB
images of one of four shapes (circle, square, triangle, star) drawn
with random position, size, rotation, color and background noise.

Why synthetic? Keeps the repo runnable with zero external downloads
(no ImageNet weights, no Kaggle dataset). Swap `load_dataset()` in
src/model.py for a real image source in a real deployment.

Stands in for a real-world task like defect/product classification —
same shape (small image, 4-way classification, needs sub-100ms
inference), different domain.

Run:
    python src/generate_data.py
Produces:
    data/train/<class>/*.png   (400 images per class)
    data/val/<class>/*.png     (80 images per class)
"""
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_SIZE = 64
SUPERSAMPLE = 4  # draw at 4x then downscale — gives anti-aliased edges
                 # instead of PIL's jagged default polygon/ellipse edges.
CLASSES = ["circle", "square", "triangle", "star"]

COLORS = [
    (220, 60, 60), (60, 130, 220), (60, 180, 90),
    (230, 170, 40), (160, 90, 200), (40, 180, 180),
]


def _random_bg():
    base = random.randint(230, 255)
    return (base, base, base)


def _add_noise(img: Image.Image, amount: int = 10) -> Image.Image:
    pixels = img.load()
    for _ in range(amount * 20):
        x, y = random.randint(0, IMG_SIZE - 1), random.randint(0, IMG_SIZE - 1)
        r, g, b = pixels[x, y]
        jitter = random.randint(-15, 15)
        pixels[x, y] = (max(0, min(255, r + jitter)),
                         max(0, min(255, g + jitter)),
                         max(0, min(255, b + jitter)))
    return img


def _draw_star(draw, cx, cy, r_outer, r_inner, rotation, color):
    points = []
    for i in range(10):
        angle = math.pi / 5 * i + rotation
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * math.sin(angle), cy - r * math.cos(angle)))
    draw.polygon(points, fill=color)


def _draw_shape(shape: str) -> Image.Image:
    hi = IMG_SIZE * SUPERSAMPLE
    img = Image.new("RGB", (hi, hi), _random_bg())
    draw = ImageDraw.Draw(img)
    color = random.choice(COLORS)

    # Choose size first, then constrain the center so the shape's
    # bounding radius always fits fully on the canvas — a shape near
    # the border must never get clipped into an ambiguous partial
    # silhouette. The square's corners extend to size*sqrt(2) from
    # center (not just `size`), so pad uses a 1.5x factor to safely
    # cover all four shape types with margin to spare.
    size = random.randint(hi // 6, hi // 4)
    pad = int(size * 1.5) + 4
    cx = random.randint(pad, hi - pad)
    cy = random.randint(pad, hi - pad)
    rotation = random.uniform(0, math.pi)

    if shape == "circle":
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
    elif shape == "square":
        pts = []
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            x = cx + size * (dx * math.cos(rotation) - dy * math.sin(rotation))
            y = cy + size * (dx * math.sin(rotation) + dy * math.cos(rotation))
            pts.append((x, y))
        draw.polygon(pts, fill=color)
    elif shape == "triangle":
        pts = []
        for i in range(3):
            angle = rotation + i * (2 * math.pi / 3)
            pts.append((cx + size * math.sin(angle), cy - size * math.cos(angle)))
        draw.polygon(pts, fill=color)
    elif shape == "star":
        _draw_star(draw, cx, cy, size, size * 0.38, rotation, color)

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)  # anti-alias
    img = _add_noise(img, amount=random.randint(3, 8))
    return img


def generate_split(split: str, n_per_class: int):
    for cls in CLASSES:
        out_dir = DATA_DIR / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = _draw_shape(cls)
            img.save(out_dir / f"{cls}_{i:04d}.png")


def main():
    generate_split("train", n_per_class=400)
    generate_split("val", n_per_class=80)
    total_train = 400 * len(CLASSES)
    total_val = 80 * len(CLASSES)
    print(f"Wrote {total_train} training images -> data/train/<class>/")
    print(f"Wrote {total_val} validation images -> data/val/<class>/")


if __name__ == "__main__":
    main()
