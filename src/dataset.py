"""
Dataset loader for AIGC detection.

Expects a directory laid out as:
    root/
      real/   ...jpg/png/etc  (label 0 = authentic)
      fake/   ...jpg/png/etc  (label 1 = AI-generated)

This matches the CIFAKE dataset's folder structure out of the box. For
SID_Set / WildFake, reorganize them into this same real/ + fake/ layout
before pointing training at them (see README).
"""
import random
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class AigcImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        augment: Optional[Callable] = None,
        preprocess: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        """
        root_dir: folder containing `real/` and `fake/` subfolders.
        augment: optional transform applied to the raw PIL image first
                 (e.g. RandomRobustnessAugment for training).
        preprocess: CLIP's preprocessing transform (resize/crop/normalize to
                    a tensor). Always applied last, after `augment`.
        max_samples: if set, randomly cap the dataset to this many images
                     total (balanced across real/fake). Handy for a fast
                     smoke test before committing to a full training run.
        """
        self.samples = []
        root = Path(root_dir)
        for label, subdir in ((0, "real"), (1, "fake")):
            folder = root / subdir
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Expected '{subdir}/' inside {root_dir} but it doesn't exist. "
                    f"See README for the expected data layout."
                )
            for path in folder.rglob("*"):
                if path.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((str(path), label))

        if not self.samples:
            raise RuntimeError(f"No images found under {root_dir}/real or {root_dir}/fake")

        if max_samples is not None and max_samples < len(self.samples):
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        self.augment = augment
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.augment is not None:
            img = self.augment(img)
        if self.preprocess is not None:
            img = self.preprocess(img)
        return img, float(label), path
