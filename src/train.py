"""
Trains the ShapeCNN from scratch and saves weights to artifacts/model.pt.

Usage:
    python -m src.train --epochs 20
"""
import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model import ShapeCNN, ShapeDataset

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cpu")
    train_ds = ShapeDataset("train")
    val_ds = ShapeDataset("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ShapeCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        train_loss = epoch_loss / len(train_ds)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    total_time = time.time() - start
    final_acc = evaluate(model, val_loader, device)
    print(f"\nTraining complete in {total_time:.1f}s. Final val_acc={final_acc:.4f}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pt")
    print(f"Saved weights -> {ARTIFACTS_DIR / 'model.pt'}")

    if final_acc < 0.90:
        print("WARNING: final val_acc below 90% — check data generation / training config.")


if __name__ == "__main__":
    main()
