"""
AIGC detector: a frozen CLIP image encoder + handcrafted forensic features,
fused and fed into a trainable classifier head.

Why this design: CLIP's visual encoder was pretrained on hundreds of
millions of images and already extracts rich, general-purpose semantic
features. Freezing it and only training a head on top is fast, needs
little data, and generalizes well. On top of that, we concatenate a few
handcrafted forensic features (lighting-direction consistency, gradient
smoothness -- see handcrafted_features.py) that CLIP wasn't trained to
capture, since CLIP was trained for semantic content, not physical
lighting/texture statistics. ViT-B-32 CLIP is ~87M image-encoder params,
so even a sizeable head stays comfortably under any reasonable total
parameter budget.
"""
import torch
import torch.nn as nn
import open_clip

from src.handcrafted_features import FEATURE_DIM


def _widen_then_halve_dims(start_dim: int, widen_dim: int = 5056, floor_dim: int = 256) -> list:
    """
    Build a sequence of hidden dims that starts by widening from `start_dim`
    up to `widen_dim`, then halves each step until it reaches `floor_dim`.
    The final halving step is clamped so it lands exactly on `floor_dim`
    rather than undershooting it. Every width in this sequence stays at or
    above `floor_dim` (except the final output layer, added separately in
    build_head), so there are no narrow bottleneck layers where signal can
    collapse.

    e.g. start_dim=516, widen_dim=5056, floor_dim=256 ->
        [516, 5056, 2528, 1264, 632, 316, 256]
    """
    dims = [start_dim, widen_dim]
    current = widen_dim
    while current > floor_dim:
        current = max(floor_dim, current // 2)
        dims.append(current)
    return dims


def build_head(input_dim: int, widen_dim: int = 5056, floor_dim: int = 256, dropout: float = 0.3) -> nn.Sequential:
    """
    Classifier head: widens from `input_dim` up to `widen_dim`, then halves
    each layer down to `floor_dim`, with each hidden Linear followed by
    GELU + Dropout. A final plain Linear(floor_dim, 1) projects to a single
    raw logit (no activation after it -- BCEWithLogitsLoss expects raw
    logits).

    For CLIP ViT-B/32 (embed_dim=512) + FEATURE_DIM=4 handcrafted features
    (input_dim=516) with the defaults this yields:
        516 -> 5056 -> 2528 -> 1264 -> 632 -> 316 -> 256 -> 1
    (7 Linear layers, ~19.7M head params). Combined with the frozen
    ~87M-param CLIP backbone, the model totals ~106.7M params.
    """
    dims = _widen_then_halve_dims(input_dim, widen_dim, floor_dim)
    layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(dims[-1], 1))  # final output layer: no activation
    return nn.Sequential(*layers)


class ClipAigcDetector(nn.Module):
    def __init__(self, clip_model_name="ViT-B-32", pretrained="openai",
                 widen_dim=5056, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        # Freeze the backbone entirely -- we only ever train `self.trainable`.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        embed_dim = self.backbone.visual.output_dim
        combined_dim = embed_dim + FEATURE_DIM

        # Everything that actually trains lives in one ModuleDict, so
        # train.py/infer.py/eval_robustness.py can save/load/optimize it as
        # a single unit instead of tracking two separate state dicts.
        self.trainable = nn.ModuleDict({
            # Handcrafted features live on a totally different numeric scale
            # than CLIP's embedding, so normalize them before concatenating
            # -- otherwise a few raw numbers with large/small magnitude can
            # dominate or get ignored next to 512 CLIP dims.
            "handcrafted_norm": nn.BatchNorm1d(FEATURE_DIM),
            "head": build_head(combined_dim, widen_dim=widen_dim, floor_dim=hidden_dim, dropout=dropout),
        })

    def train(self, mode=True):
        # Keep the frozen backbone in eval mode even when the model as a
        # whole is set to train() -- only self.trainable should behave
        # differently between train/eval (dropout, batchnorm running stats).
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images, handcrafted):
        with torch.no_grad():
            features = self.backbone.encode_image(images).float()
        handcrafted = self.trainable["handcrafted_norm"](handcrafted)
        combined = torch.cat([features, handcrafted], dim=1)
        return self.trainable["head"](combined).squeeze(-1)