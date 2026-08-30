"""
Produces the robustness evaluation summary required in the deliverables:
compares clean accuracy against accuracy under each transform/severity from
the hackathon's robustness table (spec 5.2), and dumps sample false
positives/negatives per condition for the error analysis writeup.

Usage:
    python -m src.eval_robustness --data_dir data/val --checkpoint results/head_best.pt

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


def evaluate(model, loader, device):
    correct, total = 0, 0
    false_pos, false_neg = [], []
    with torch.no_grad():
        for images, labels, paths in loader:
            images, labels = images.to(device), labels.to(device=device, dtype=torch.float32)
            probs = torch.sigmoid(model(images))
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
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ClipAigcDetector(clip_model_name=ckpt["clip_model"], pretrained=ckpt["pretrained"]).to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval()

    rows = []
    all_errors = {}

    # Clean baseline
    ds = AigcImageDataset(args.data_dir, augment=None, preprocess=model.preprocess)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    acc, fp, fneg = evaluate(model, loader, device)
    rows.append({"transform": "clean", "severity": "-", "accuracy": round(acc, 4)})
    all_errors["clean"] = {"false_positives": fp[:10], "false_negatives": fneg[:10]}
    print(f"clean: acc={acc:.4f}")

    # Each transform x severity from the robustness table
    for name, (transform_fn, severities) in TRANSFORM_REGISTRY.items():
        for severity in severities:
            augment = lambda img, f=transform_fn, s=severity: f(img, s)
            ds = AigcImageDataset(args.data_dir, augment=augment, preprocess=model.preprocess)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            acc, fp, fneg = evaluate(model, loader, device)
            rows.append({"transform": name, "severity": severity, "accuracy": round(acc, 4)})
            all_errors[f"{name}_{severity}"] = {"false_positives": fp[:5], "false_negatives": fneg[:5]}
            print(f"{name} (severity={severity}): acc={acc:.4f}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["transform", "severity", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote robustness summary to {args.out_csv}")

    with open(args.out_errors, "w") as f:
        json.dump(all_errors, f, indent=2)
    print(f"Wrote error examples to {args.out_errors}")


if __name__ == "__main__":
    main()
