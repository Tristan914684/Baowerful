"""
REQUIRED deliverable script: scans an image directory and writes a JSON file
with a per-image AIGC-likelihood score.

Usage:
    python -m src.infer --image_dir path/to/images --checkpoint results/head_best.pt --output predictions.json

Output JSON format (a list, one entry per image):
    [
      {"image_path": "path/to/images/foo.jpg", "pred": 0.93},
      ...
    ]
`pred` is the model's confidence that the image is AI-generated, in [0, 1].
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.model import ClipAigcDetector
from src.dataset import IMG_EXTENSIONS
from src.handcrafted_features import compute_handcrafted_features


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_images(image_dir):
    root = Path(image_dir)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--batch_size", type=int, default=32)
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

    image_paths = find_images(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.image_dir}")

    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), args.batch_size)):
            batch_paths = image_paths[i:i + args.batch_size]
            batch_imgs, batch_handcrafted, valid_paths = [], [], []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_handcrafted.append(compute_handcrafted_features(img))
                    batch_imgs.append(model.preprocess(img))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Skipping {p}: {e}")

            if not batch_imgs:
                continue

            batch_tensor = torch.stack(batch_imgs).to(device)
            handcrafted_tensor = torch.from_numpy(np.stack(batch_handcrafted)).to(device)
            probs = torch.sigmoid(model(batch_tensor, handcrafted_tensor)).cpu().tolist()

            for path, prob in zip(valid_paths, probs):
                results.append({"image_path": str(path), "pred": round(prob, 6)})

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()