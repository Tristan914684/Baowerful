"""
AIGC detector: a frozen CLIP image encoder + a small trainable classifier head.

Why this design: CLIP's visual encoder was pretrained on hundreds of millions
of images and already extracts rich, general-purpose features. Freezing it
and only training a small head on top is fast (few trainable params, quick
epochs), needs little data, and generalizes well -- a good fit for a
72-hour hackathon. It also comfortably satisfies the <2B parameter limit:
ViT-B-32 CLIP is ~87M image-encoder params.
"""
import torch
import torch.nn as nn
import open_clip


class ClipAigcDetector(nn.Module):
    def __init__(self, clip_model_name="ViT-B-32", pretrained="openai", hidden_dim=600_000, dropout=0.3):
        super().__init__()
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        # Freeze the backbone entirely -- we only ever train `self.head`.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        embed_dim = self.backbone.visual.output_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode=True):
        # Keep the frozen backbone in eval mode even when the model as a
        # whole is set to train() -- only the head should behave differently
        # between train/eval (dropout).
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone.encode_image(images).float()
        return self.head(features).squeeze(-1)
