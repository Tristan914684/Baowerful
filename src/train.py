"""
Trains the classifier head on top of the frozen CLIP backbone.

Usage:
    python -m src.train --train_dir data/train --val_dir data/val --epochs 10

Data layout expected (see dataset.py):
    data/train/real, data/train/fake
    data/val/real,   data/val/fake
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import ClipAigcDetector
from src.dataset import AigcImageDataset
from src.augmentations import RandomRobustnessAugment


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels, _ in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--clip_model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aug_prob", type=float, default=0.5,
                         help="Probability of applying a robustness augmentation per training image.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", default="results/head_best.pt")
    parser.add_argument("--max_train_samples", type=int, default=None,
                         help="Cap training set size for a quick smoke test.")
    parser.add_argument("--max_val_samples", type=int, default=None,
                         help="Cap validation set size for a quick smoke test.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = ClipAigcDetector(clip_model_name=args.clip_model, pretrained=args.pretrained).to(device)

    train_ds = AigcImageDataset(
        args.train_dir,
        augment=RandomRobustnessAugment(p=args.aug_prob),
        preprocess=model.preprocess,
        max_samples=args.max_train_samples,
    )
    val_ds = AigcImageDataset(
        args.val_dir, augment=None, preprocess=model.preprocess, max_samples=args.max_val_samples
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "head_state_dict": model.head.state_dict(),
                "clip_model": args.clip_model,
                "pretrained": args.pretrained,
                "val_acc": val_acc,
            }, args.out)
            print(f"  -> saved new best checkpoint ({val_acc:.4f}) to {args.out}")

    with open("results/train_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
