import torch
from PIL import Image

from src.model import CLASSES, IDX2CLASS, IMG_SIZE, ShapeCNN, image_to_tensor


def test_image_to_tensor_shape_and_range():
    img = Image.new("RGB", (32, 32), (100, 150, 200))
    t = image_to_tensor(img)
    assert t.shape == (3, IMG_SIZE, IMG_SIZE)
    assert t.dtype == torch.float32
    assert 0.0 <= t.min().item() and t.max().item() <= 1.0


def test_image_to_tensor_preserves_color_relationships():
    img = Image.new("RGB", (16, 16), (255, 0, 0))
    t = image_to_tensor(img)
    assert t[0].mean() > t[1].mean()
    assert t[0].mean() > t[2].mean()


def test_shapecnn_forward_pass_output_shape():
    model = ShapeCNN()
    model.eval()
    batch = torch.rand(4, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (4, len(CLASSES))


def test_shapecnn_single_image_forward_pass():
    model = ShapeCNN()
    model.eval()
    single = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        logits = model(single)
    probs = torch.softmax(logits[0], dim=0)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_class_label_mappings_are_consistent():
    assert len(CLASSES) == 4
    assert set(IDX2CLASS.values()) == set(CLASSES)
    for i, cls in IDX2CLASS.items():
        assert CLASSES[i] == cls
