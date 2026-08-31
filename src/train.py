"""
Trains the classifier head (+ handcrafted-feature fusion) on top of the
frozen CLIP backbone.

Usage:
    python -m src.train --train_dir data/train --val_dir data/test --epochs 10

    # Lean-head ablation (narrow head, no widen step) -- always run this
    # as a fresh, separately-named checkpoint since its head shape isn't
    # compatible with the default wide-head checkpoint:
    python -m src.train --train_dir data/train --val_dir data/test --epochs 10 \
        --lean_head --fresh --out results/head_lean.pt

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


def _partial_resume_head(model, ckpt) -> float:
    """
    Transplant a checkpoint's weights into `model.trainable` when the only
    architecture change is FEATURE_DIM growing (e.g. a new handcrafted
    feature was appended in handcrafted_features.py). Everything except
    the head's first Linear layer and the handcrafted BatchNorm is
    unaffected by FEATURE_DIM and gets copied over unchanged; only the
    brand-new feature's input weight column(s) are left at their random
    init. Raises if the checkpoint doesn't match this "FEATURE_DIM grew"
    assumption (e.g. widen_dim/floor_dim also changed, or the head shape
    itself changed -- e.g. switching --lean_head on or off), so the
    caller can fall back to a full fresh init.

    Returns the checkpoint's saved val_acc, so the caller can still treat
    that as the "best score to beat" even though the load was partial.
    """
    old_sd = ckpt["trainable_state_dict"]
    new_sd = model.trainable.state_dict()

    embed_dim = model.backbone.visual.output_dim
    new_feature_dim = model.trainable["handcrafted_norm"].num_features
    old_feature_dim = old_sd["handcrafted_norm.weight"].shape[0]

    old_w = old_sd["head.0.weight"]  # (hidden, embed_dim + old_feature_dim)
    new_w = new_sd["head.0.weight"]  # (hidden, embed_dim + new_feature_dim)

    # Sanity-check this really is a "FEATURE_DIM grew, nothing else changed"
    # situation before touching anything -- same hidden width, same CLIP
    # embed_dim, old feature count smaller than new. This will correctly
    # raise (and fall back to fresh init) if the head *shape* changed too,
    # e.g. resuming a lean-head run from a wide-head checkpoint or vice
    # versa -- those aren't a "FEATURE_DIM grew" situation and shouldn't be
    # partially transplanted.
    if (old_w.shape[0] != new_w.shape[0]
            or old_w.shape[1] != embed_dim + old_feature_dim
            or new_w.shape[1] != embed_dim + new_feature_dim
            or old_feature_dim >= new_feature_dim):
        raise RuntimeError(
            "Checkpoint shape mismatch doesn't match the expected "
            "'FEATURE_DIM grew, everything else identical' pattern -- "
            "can't safely do a partial transplant."
        )

    # BatchNorm1d(FEATURE_DIM) over handcrafted features: keep the new
    # tensors' random init for new channels, overwrite the first
    # old_feature_dim channels with the learned values.
    for key in ["handcrafted_norm.weight", "handcrafted_norm.bias",
                "handcrafted_norm.running_mean", "handcrafted_norm.running_var"]:
        new_sd[key][:old_feature_dim] = old_sd[key]
    new_sd["handcrafted_norm.num_batches_tracked"] = old_sd["handcrafted_norm.num_batches_tracked"]

    # Head's first Linear layer: CLIP columns unchanged, old handcrafted
    # feature columns copied into their same positions, new feature
    # column(s) left at random init.
    new_w[:, :embed_dim] = old_w[:, :embed_dim]
    new_w[:, embed_dim:embed_dim + old_feature_dim] = old_w[:, embed_dim:]
    new_sd["head.0.bias"] = old_sd["head.0.bias"]  # output width unchanged, safe to copy

    # Every other head layer never depended on FEATURE_DIM -- copy wholesale.
    for key in old_sd:
        if key.startswith("head.") and key not in ("head.0.weight", "head.0.bias"):
            new_sd[key] = old_sd[key]

    model.trainable.load_state_dict(new_sd)
    return ckpt.get("val_acc", 0.0)


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, handcrafted, labels, _ in tqdm(loader, leave=False):
            images = images.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device=device, dtype=torch.float32)
            logits = model(images, handcrafted)
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
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing checkpoint at --out and start from a randomly "
                              "initialized head. Default behavior (no --fresh) resumes from --out "
                              "if it already exists, and only overwrites it if a new epoch beats "
                              "the val_acc saved in that checkpoint.")
    parser.add_argument("--lean_head", action="store_true",
                         help="Use the narrow head (input_dim -> hidden*2 -> hidden -> 1, no "
                              "widen-then-halve step) instead of the default wide head. This "
                              "changes head parameter shapes, so it is NOT resumable from a "
                              "checkpoint trained without --lean_head (or vice versa) -- always "
                              "pair this with --fresh and a distinct --out path, e.g. "
                              "results/head_lean.pt, so you don't clobber or fail to load an "
                              "incompatible checkpoint.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = ClipAigcDetector(clip_model_name=args.clip_model, pretrained=args.pretrained,
                              lean_head=args.lean_head).to(device)

    best_val_acc = 0.0
    if not args.fresh and Path(args.out).exists():
        ckpt = torch.load(args.out, map_location=device)
        if ckpt.get("clip_model") != args.clip_model or ckpt.get("pretrained") != args.pretrained:
            print(f"  WARNING: existing checkpoint was trained with clip_model={ckpt.get('clip_model')!r}, "
                  f"pretrained={ckpt.get('pretrained')!r}, but this run uses "
                  f"clip_model={args.clip_model!r}, pretrained={args.pretrained!r}. "
                  f"Skipping resume and starting fresh instead.")
        else:
            try:
                model.trainable.load_state_dict(ckpt["trainable_state_dict"])
                best_val_acc = ckpt.get("val_acc", 0.0)
                print(f"Resumed weights from {args.out} (existing val_acc={best_val_acc:.4f}). "
                      f"Will only overwrite it if a new epoch beats this.")
            except RuntimeError as e:
                # Exact-shape load failed -- most commonly because FEATURE_DIM grew
                # (a new handcrafted feature was added in handcrafted_features.py),
                # or because --lean_head doesn't match how this checkpoint's head
                # was built. Try a surgical partial transplant instead of discarding
                # everything: only the head's very first Linear layer and the
                # handcrafted BatchNorm actually depend on FEATURE_DIM, so every
                # other learned weight (all later head layers, and the old feature
                # columns in that first layer) can still be reused as-is, PROVIDED
                # the head shape itself (wide vs. lean) didn't also change -- if it
                # did, _partial_resume_head raises and we fall back to random init.
                try:
                    best_val_acc = _partial_resume_head(model, ckpt)
                    print(f"Checkpoint architecture changed (likely FEATURE_DIM grew), but "
                          f"transplanted all reusable weights from {args.out} "
                          f"(existing val_acc={best_val_acc:.4f}); only the new feature's "
                          f"input weights are randomly initialized. "
                          f"Will only overwrite it if a new epoch beats this.")
                except Exception as partial_e:
                    print(f"  WARNING: could not load or partially transplant checkpoint at "
                          f"{args.out} -- starting from a fully randomly initialized head "
                          f"instead. (Expected if --lean_head differs from how this checkpoint "
                          f"was trained -- use a distinct --out path for lean-head runs.)\n"
                          f"  Original error: {e}\n"
                          f"  Partial-transplant error: {partial_e}")
                    best_val_acc = 0.0
    elif args.fresh:
        print("--fresh set: starting from a randomly initialized head (ignoring any existing checkpoint).")
    else:
        print(f"No existing checkpoint found at {args.out}; starting from a randomly initialized head.")

    train_ds = AigcImageDataset(
        args.train_dir,
        augment=RandomRobustnessAugment(p=args.aug_prob),
        preprocess=model.preprocess,
        max_samples=args.max_train_samples,
    )
    val_ds = AigcImageDataset(
        args.val_dir, augment=None, preprocess=model.preprocess, max_samples=args.max_val_samples
    )

    # drop_last=True: avoids a rare crash if the last batch of an epoch has
    # exactly 1 sample -- BatchNorm1d (used to normalize handcrafted
    # features) requires more than 1 value per channel in training mode.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.trainable.parameters(), lr=args.lr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    history = []

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
                "trainable_state_dict": model.trainable.state_dict(),
                "clip_model": args.clip_model,
                "pretrained": args.pretrained,
                "val_acc": val_acc,
            }, args.out)
            print(f"  -> saved new best checkpoint ({val_acc:.4f}) to {args.out}")

    history_path = Path("results/train_history.json")
    if history_path.exists() and not args.fresh:
        with open(history_path) as f:
            prior_history = json.load(f)
        # Re-number epochs so they continue counting across resumed runs
        # instead of restarting at 1 and colliding with prior epoch numbers.
        offset = prior_history[-1]["epoch"] if prior_history else 0
        for h in history:
            h["epoch"] += offset
        history = prior_history + history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()