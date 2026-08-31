"""
Produces the robustness evaluation summary required in the deliverables:
compares clean accuracy against accuracy under each transform/severity from
the hackathon's robustness table (spec 5.2), and dumps sample false
positives/negatives per condition for the error analysis writeup.

Usage:
    python -m src.eval_robustness --data_dir data/test --checkpoint results/head_best.pt

data_dir must have the same real/ + fake/ layout as training data, and
should be a held-out split the model has never trained on.
"""
import argparse
import csv
import json

import torch
from torch.utils.data import DataLoader

from src.model import ClipAigcDetector
from src.dataset import AigcImageDataset
from src.augmentations import TRANSFORM_REGISTRY


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _SeverityAugment:
    """
    Picklable stand-in for the old `lambda img, f=..., s=...: f(img, s)`.
    On Windows, DataLoader workers are spawned (not forked), so every
    argument -- including the dataset's `augment` callable -- must be
    pickled to hand off to the worker process. Lambdas can't be pickled;
    a plain class with __call__ can.
    """
    def __init__(self, fn, severity):
        self.fn = fn
        self.severity = severity

    def __call__(self, img):
        return self.fn(img, self.severity)


def evaluate(model, loader, device):
    correct, total = 0, 0
    false_pos, false_neg = [], []
    with torch.no_grad():
        for images, handcrafted, labels, paths in loader:
            images = images.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device=device, dtype=torch.float32)
            probs = torch.sigmoid(model(images, handcrafted))
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)
            for path, pred, label, prob in zip(paths, preds.tolist(), labels.tolist(), probs.tolist()):
                if pred == 1 and label == 0:
                    false_pos.append({"image_path": path, "pred_prob": prob})
                elif pred == 0 and label == 1:
                    false_neg.append({"image_path": path, "pred_prob": prob})
    return correct / total, false_pos, false_neg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Held-out folder with real/ and fake/ subfolders.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_csv", default="docs/robustness_summary.csv")
    parser.add_argument("--out_errors", default="docs/error_examples.json")
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Cap each condition's evaluation set size (balanced across real/fake) for a quick smoke test.")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    # lean_head is saved into the checkpoint by train.py, so the correct
    # head shape (wide vs. lean) is always reconstructed automatically --
    # no need to remember/pass a matching flag by hand here. Falls back to
    # False (the original default) for checkpoints saved before this field
    # existed.
    model = ClipAigcDetector(clip_model_name=ckpt["clip_model"], pretrained=ckpt["pretrained"],
                              lean_head=ckpt.get("lean_head", False)).to(device)
    model.trainable.load_state_dict(ckpt["trainable_state_dict"])
    model.eval()

    rows = []
    all_errors = {}

    # Clean baseline
    ds = AigcImageDataset(args.data_dir, augment=None, preprocess=model.preprocess, max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    acc, fp, fneg = evaluate(model, loader, device)
    rows.append({"transform": "clean", "severity": "-", "accuracy": round(acc, 4)})
    all_errors["clean"] = {"false_positives": fp[:10], "false_negatives": fneg[:10]}
    print(f"clean: acc={acc:.4f}")

    # Each transform x severity from the robustness table
    for name, (transform_fn, severities) in TRANSFORM_REGISTRY.items():
        for severity in severities:
            augment = _SeverityAugment(transform_fn, severity)
            ds = AigcImageDataset(args.data_dir, augment=augment, preprocess=model.preprocess,
                                   max_samples=args.max_samples)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            acc, fp, fneg = evaluate(model, loader, device)
            rows.append({"transform": name, "severity": severity, "accuracy": round(acc, 4)})
            all_errors[f"{name}_{severity}"] = {"false_positives": fp[:5], "false_negatives": fneg[:5]}
            print(f"{name} (severity={severity}): acc={acc:.4f}")

            # Write incrementally after every condition, so an interrupted
            # run (Ctrl+C, crash, closed terminal) still leaves a partial
            # CSV/JSON on disk instead of losing everything.
            with open(args.out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["transform", "severity", "accuracy"])
                writer.writeheader()
                writer.writerows(rows)
            with open(args.out_errors, "w") as f:
                json.dump(all_errors, f, indent=2)

    print(f"Wrote robustness summary to {args.out_csv}")
    print(f"Wrote error examples to {args.out_errors}")


if __name__ == "__main__":
    main()