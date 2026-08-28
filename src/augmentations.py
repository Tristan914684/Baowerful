"""
Robustness transforms mirroring the hackathon's evaluation table (spec 5.2).

Used two ways:
  1. Randomly, during training (RandomRobustnessAugment) -- so the model
     learns to be invariant to these degradations instead of only ever
     seeing clean images.
  2. Deterministically, one transform/severity at a time, during evaluation
     (see eval_robustness.py) -- to produce the clean-vs-transformed
     robustness table required in the deliverables.
"""
import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode through JPEG at the given quality (social-media re-upload)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Out-of-focus blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_degrade(img: Image.Image, scale: float) -> Image.Image:
    """Downscale then upscale back (thumbnail generation)."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Additive Gaussian sensor noise. sigma is in [0, 1] pixel-intensity units."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def color_jitter(img: Image.Image, strength: float) -> Image.Image:
    """Randomly nudge brightness / contrast / saturation by up to +/- strength."""
    img = img.convert("RGB")
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-strength, strength)
        img = enhancer_cls(img).enhance(factor)
    return img


def center_crop_frac(img: Image.Image, frac: float) -> Image.Image:
    """Center-crop to `frac` of width/height, then resize back (profile-pic cropping)."""
    w, h = img.size
    new_w, new_h = int(w * frac), int(h * frac)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


# name -> (function, [severity values swept during evaluation])
TRANSFORM_REGISTRY = {
    "jpeg_compression": (jpeg_compress, [90, 70, 50, 30]),
    "gaussian_blur": (gaussian_blur, [0.5, 1.0, 2.0]),
    "resize": (resize_degrade, [0.5, 0.25]),
    "gaussian_noise": (gaussian_noise, [0.02, 0.05, 0.10]),
    "color_jitter": (color_jitter, [0.2]),
    "center_crop": (center_crop_frac, [0.8]),
}


class RandomRobustnessAugment:
    """
    Training-time augmentation: with probability `p`, apply ONE randomly
    chosen transform (at a randomly chosen severity) from TRANSFORM_REGISTRY.
    Otherwise pass the image through unchanged.
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self.names = list(TRANSFORM_REGISTRY.keys())

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        name = random.choice(self.names)
        fn, severities = TRANSFORM_REGISTRY[name]
        severity = random.choice(severities)
        try:
            return fn(img, severity)
        except Exception:
            # Never let a broken augmentation crash training.
            return img
