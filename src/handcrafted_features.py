"""
Handcrafted forensic features that complement the CLIP embedding.

CLIP was trained for semantic content, not for physical lighting
consistency or pixel-level texture statistics, so these features are meant
to capture signal CLIP doesn't naturally encode:

1. Lighting-direction consistency: in a real photo, light bounces off
   multiple surfaces, so edge/gradient directions across the image are
   spread over many angles. AI generators often imply a single dominant
   light source, so gradient directions can cluster more tightly. We
   summarize this with the *entropy* of a magnitude-weighted
   gradient-direction histogram: low entropy = concentrated in one
   direction (more "AI-like" per this hypothesis), high entropy = scattered
   (more "real-like").

2. Gradient / texture smoothness: AI images tend to be smoother at the
   pixel level than real camera sensor output, which has fine, somewhat
   irregular high-frequency detail. Summarized two ways:
     - Laplacian variance (a standard sharpness/blur metric)
     - the fraction of 2D FFT energy sitting in the high-frequency band

3. Radial light-source consistency: real photos are usually lit by
   several light sources/bounces plus occlusion from scene geometry, so
   brightness doesn't fall off cleanly from a single point and edge
   gradients aren't strongly organized around one center. AI generators
   often imply a single dominant light source with a suspiciously smooth,
   almost vignette-like falloff. Summarized two ways, both computed
   relative to an estimated "light source" location (the centroid of the
   brightest ~5% of a smoothed luminance map):
     - grad_direction_entropy already covers rotational clustering of
       edge directions; this adds the *radial* component specifically --
       how well luminance is explained by a simple linear falloff with
       distance from that centroid (radial_falloff_r2), and how strongly
       edge gradients point toward/away from that centroid rather than in
       scene-driven directions (radial_gradient_alignment). High values on
       either = a more "centric", single-source-like lighting pattern.

4. Channel saturation: real camera sensors (Bayer filter + demosaicing +
   sensor noise + gamma/white-balance processing) essentially never
   produce pixels that sit at an exact primary-color extreme -- e.g. a
   channel reading exactly 0 or 255 with the other two channels also
   pinned at an extreme. AI generators (and flat vector/UI-style content)
   more readily produce mathematically "pure" colors. This is computed on
   the RGB image directly, since converting to grayscale first would
   destroy the per-channel information needed to detect it.

All features are computed on a fixed-size version of the image so they're
comparable across images of different original resolutions. Uses only
numpy/PIL -- no extra dependencies (no scipy/cv2 needed).
"""
import numpy as np
from PIL import Image

FEATURE_DIM = 7
FEATURE_NAMES = [
    "grad_direction_entropy", "laplacian_var", "fft_highfreq_ratio", "grad_mag_std",
    "radial_falloff_r2", "radial_gradient_alignment", "channel_saturation_ratio",
]

_RESIZE = 224
_N_DIR_BINS = 16


def _to_gray_array(img: Image.Image, size: int = _RESIZE) -> np.ndarray:
    gray = img.convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(gray, dtype=np.float32) / 255.0


def _to_rgb_array(img: Image.Image, size: int = _RESIZE) -> np.ndarray:
    rgb = img.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(rgb, dtype=np.float32) / 255.0  # (H, W, 3)


def _conv3x3(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual 3x3 convolution via numpy slicing (no scipy/cv2 dependency)."""
    padded = np.pad(arr, 1, mode="edge")
    out = np.zeros_like(arr)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + arr.shape[0], j:j + arr.shape[1]]
    return out


_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _sobel_gradients(arr: np.ndarray):
    gx = _conv3x3(arr, _SOBEL_X)
    gy = _conv3x3(arr, _SOBEL_Y)
    return gx, gy


def _grad_direction_entropy(gx: np.ndarray, gy: np.ndarray) -> float:
    """Entropy (normalized to [0, 1]) of a magnitude-weighted gradient-direction histogram."""
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx)  # [-pi, pi]
    # Only count edges with meaningful gradient magnitude -- flat regions
    # (sky, backgrounds, AI-smoothed areas) have near-zero, noisy angles
    # that would just add uninformative entropy.
    threshold = magnitude.mean() * 0.5
    mask = magnitude > threshold
    if not np.any(mask):
        return 0.0
    hist, _ = np.histogram(angle[mask], bins=_N_DIR_BINS, range=(-np.pi, np.pi), weights=magnitude[mask])
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    entropy = -(p * np.log(p)).sum()
    return float(entropy / np.log(_N_DIR_BINS))


def _laplacian_var(arr: np.ndarray) -> float:
    lap = _conv3x3(arr, _LAPLACIAN)
    return float(lap.var())


_cy, _cx = _RESIZE // 2, _RESIZE // 2
_yy, _xx = np.mgrid[0:_RESIZE, 0:_RESIZE]
_dist = np.sqrt((_yy - _cy) ** 2 + (_xx - _cx) ** 2)
_HIGH_FREQ_MASK = _dist > (min(_cy, _cx) * 0.5)  # precomputed once: image size is fixed at _RESIZE


def _fft_highfreq_ratio(arr: np.ndarray) -> float:
    mag = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    total_energy = mag.sum() + 1e-8
    high_energy = mag[_HIGH_FREQ_MASK].sum()
    return float(high_energy / total_energy)


_BOX3 = np.ones((3, 3), dtype=np.float32) / 9.0
_YY, _XX = np.mgrid[0:_RESIZE, 0:_RESIZE]  # pixel coordinate grids, reused per image below
_BRIGHT_PERCENTILE = 95
_N_SMOOTH_PASSES = 4


def _light_source_centroid(arr: np.ndarray):
    """
    Rough estimate of a single dominant "light source" location: the
    intensity-weighted centroid of the brightest ~5% of pixels in a
    heavily smoothed luminance map. Smoothing first avoids anchoring on a
    single noisy hot pixel (e.g. a small bright highlight or sensor
    speck) instead of the broad bright region a real light source or
    generator's implied lighting would produce.
    """
    smoothed = arr
    for _ in range(_N_SMOOTH_PASSES):
        smoothed = _conv3x3(smoothed, _BOX3)
    threshold = np.percentile(smoothed, _BRIGHT_PERCENTILE)
    mask = smoothed >= threshold
    if not np.any(mask):
        return _RESIZE / 2.0, _RESIZE / 2.0
    weights = smoothed[mask]
    cy = float(np.average(_YY[mask], weights=weights))
    cx = float(np.average(_XX[mask], weights=weights))
    return cy, cx


def _radial_falloff_r2(arr: np.ndarray, cy: float, cx: float) -> float:
    """
    How well a simple linear fit of brightness vs. distance from the
    estimated light-source centroid explains the image's luminance
    (R^2, clamped to [0, 1]). A real scene's brightness pattern is shaped
    by multiple surfaces/occluders and rarely reduces to one clean radial
    trend, so this stays low. A generator implying one dominant light with
    a smooth vignette-like falloff pushes this higher.
    """
    dist = np.sqrt((_YY - cy) ** 2 + (_XX - cx) ** 2).ravel()
    lum = arr.ravel()
    d_mean, l_mean = dist.mean(), lum.mean()
    dd = dist - d_mean
    ll = lum - l_mean
    denom = (dd ** 2).sum()
    if denom < 1e-8:
        return 0.0
    slope = (dd * ll).sum() / denom
    intercept = l_mean - slope * d_mean
    pred = slope * dist + intercept
    ss_res = ((lum - pred) ** 2).sum()
    ss_tot = ((lum - l_mean) ** 2).sum() + 1e-8
    r2 = 1.0 - ss_res / ss_tot
    return float(np.clip(r2, 0.0, 1.0))


def _radial_gradient_alignment(gx: np.ndarray, gy: np.ndarray, cy: float, cx: float) -> float:
    """
    Magnitude-weighted average alignment between each significant edge's
    gradient direction and the radial direction from the estimated light
    centroid to that pixel (0 = gradients unrelated to the centroid's
    direction, i.e. scene-driven; 1 = gradients point straight
    toward/away from the centroid, as a clean point-source falloff would
    produce). Uses absolute cosine similarity since a radial falloff
    produces gradients pointing either toward or away from the source
    depending on sign, and both indicate the same "centric" pattern.
    """
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    threshold = magnitude.mean() * 0.5
    mask = magnitude > threshold
    if not np.any(mask):
        return 0.0

    ry = _YY - cy
    rx = _XX - cx
    rmag = np.sqrt(ry ** 2 + rx ** 2) + 1e-8
    ry_unit, rx_unit = ry / rmag, rx / rmag

    gmag = magnitude + 1e-8
    gy_unit, gx_unit = gy / gmag, gx / gmag

    cos_sim = np.abs(gy_unit * ry_unit + gx_unit * rx_unit)
    return float(np.average(cos_sim[mask], weights=magnitude[mask]))


_SATURATION_EPS = 2.0 / 255.0  # "near" 0/255 tolerance, in [0,1]-normalized units


def _channel_saturation_ratio(rgb_arr: np.ndarray, eps: float = _SATURATION_EPS) -> float:
    """
    Fraction of pixels where ALL THREE channels sit at (or within `eps`
    of) an extreme -- 0 or 255. Real camera photos almost never produce
    this: Bayer demosaicing, sensor noise, and in-camera processing all
    leave small nonzero spread across channels even in bright/saturated
    regions. Flat, mathematically "pure" colors (pure red/green/blue,
    pure black/white) show up far more in synthetic/rendered/AI content.
    Fully vectorized (whole-array numpy ops) -- no per-pixel Python loop.
    """
    near_extreme = (rgb_arr <= eps) | (rgb_arr >= (1.0 - eps))  # (H, W, 3) bool
    fully_saturated = near_extreme.all(axis=-1)                  # (H, W) bool
    return float(fully_saturated.mean())


def compute_handcrafted_features(img: Image.Image) -> np.ndarray:
    """
    Returns a length-FEATURE_DIM float32 numpy array:
        [grad_direction_entropy, laplacian_var, fft_highfreq_ratio, grad_mag_std,
         radial_falloff_r2, radial_gradient_alignment, channel_saturation_ratio]
    """
    arr = _to_gray_array(img)
    gx, gy = _sobel_gradients(arr)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    direction_entropy = _grad_direction_entropy(gx, gy)
    lap_var = _laplacian_var(arr)
    fft_ratio = _fft_highfreq_ratio(arr)
    grad_mag_std = float(magnitude.std())

    cy, cx = _light_source_centroid(arr)
    radial_r2 = _radial_falloff_r2(arr, cy, cx)
    radial_alignment = _radial_gradient_alignment(gx, gy, cy, cx)

    rgb_arr = _to_rgb_array(img)
    saturation_ratio = _channel_saturation_ratio(rgb_arr)

    return np.array(
        [direction_entropy, lap_var, fft_ratio, grad_mag_std,
         radial_r2, radial_alignment, saturation_ratio],
        dtype=np.float32,
    )