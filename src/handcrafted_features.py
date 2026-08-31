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

5. Local blur/sharpness inconsistency: features 2-3 above summarize
   blur/texture as a SINGLE number for the whole image, which dilutes a
   localized edit -- e.g. an inpainted or spliced region that was blended
   with local smoothing, or composited from a differently-compressed
   source, typically covers only part of the frame. A real, untouched
   photo is usually fairly consistent in sharpness across regions (same
   camera, lens, and moment everywhere), whereas a tampered region tends
   to stand out as a patch with noticeably different high-frequency
   detail than its surroundings. We tile the image into a small grid,
   compute Laplacian variance (sharpness) independently per tile, and
   summarize how much those tile-level values disagree with each other
   (coefficient of variation): low = uniformly sharp/blurred everywhere
   (real-like), high = one or more tiles stand out from the rest
   (tamper-like).

6. JPEG blockiness: real photos re-encoded through JPEG (or already
   compressed at the source) develop a characteristic discontinuity at
   8x8 DCT block boundaries -- pixel differences across block edges tend
   to be larger, relative to differences *within* a block, than you'd get
   from a smoothly-varying natural image region. AI generators don't
   produce this block-grid artifact natively, so the ratio of
   cross-block-boundary difference to within-block difference is a
   structural (not sharpness-based) signal. This matters specifically
   because our robustness eval includes JPEG re-compression at several
   qualities plus blur/resize/noise -- sharpness-based features (2, 5)
   degrade under those transforms, but blockiness is measuring grid
   *structure*, so it degrades more gracefully.

7. Resampling periodicity: generator upsampling stages (transposed
   conv / interpolation stacks) often leave faint periodic structure in
   the image's high-frequency residual, which shows up as a sharp peak
   in the FFT magnitude of the Laplacian response. A real photo's
   high-frequency residual is closer to unstructured sensor noise, so
   its FFT magnitude doesn't have as strong a peak. Like blockiness, this
   is a structural/periodic signal rather than a raw-sharpness one, so it
   is comparatively more robust to blur/resize/JPEG than laplacian_var or
   fft_highfreq_ratio alone.

All features are computed on a fixed-size version of the image so they're
comparable across images of different original resolutions. Uses only
numpy/PIL -- no extra dependencies (no scipy/cv2 needed).
"""
import numpy as np
from PIL import Image

FEATURE_DIM = 10
FEATURE_NAMES = [
    "grad_direction_entropy", "laplacian_var", "fft_highfreq_ratio", "grad_mag_std",
    "radial_falloff_r2", "radial_gradient_alignment", "channel_saturation_ratio",
    "local_blur_inconsistency", "jpeg_blockiness", "resample_periodicity",
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
_HIGH_FREQ_MASK = _dist > (min(_cy, _cx) * 0.5)


def _fft_highfreq_ratio(arr: np.ndarray) -> float:
    mag = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    total_energy = mag.sum() + 1e-8
    high_energy = mag[_HIGH_FREQ_MASK].sum()
    return float(high_energy / total_energy)


_BOX3 = np.ones((3, 3), dtype=np.float32) / 9.0
_YY, _XX = np.mgrid[0:_RESIZE, 0:_RESIZE]
_BRIGHT_PERCENTILE = 95
_N_SMOOTH_PASSES = 4


def _light_source_centroid(arr: np.ndarray):
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


_SATURATION_EPS = 2.0 / 255.0


def _channel_saturation_ratio(rgb_arr: np.ndarray, eps: float = _SATURATION_EPS) -> float:
    near_extreme = (rgb_arr <= eps) | (rgb_arr >= (1.0 - eps))
    fully_saturated = near_extreme.all(axis=-1)
    return float(fully_saturated.mean())


_BLUR_GRID = 4  # 4x4 = 16 tiles for local sharpness comparison
_BLUR_TILE = _RESIZE // _BLUR_GRID


def _local_blur_inconsistency(lap: np.ndarray, grid: int = _BLUR_GRID) -> float:
    """
    Coefficient of variation (std / mean) of Laplacian variance computed
    independently per tile in a `grid` x `grid` split of the image.
    `lap` is passed in (already computed by the caller) to avoid a
    redundant convolution pass. Low value = sharpness is fairly uniform
    across the image (typical of an untouched photo); high value = one or
    more tiles have noticeably different sharpness than the rest
    (consistent with a locally blurred/blended edit, or a patch
    composited from a differently-compressed source).
    """
    tile = _RESIZE // grid
    tile_vars = np.empty(grid * grid, dtype=np.float64)
    idx = 0
    for i in range(grid):
        for j in range(grid):
            patch = lap[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]
            tile_vars[idx] = patch.var()
            idx += 1
    mean_v = tile_vars.mean()
    if mean_v < 1e-8:
        return 0.0
    return float(tile_vars.std() / mean_v)


_BLOCK_SIZE = 8  # standard JPEG DCT block size


def _jpeg_blockiness(arr: np.ndarray, block: int = _BLOCK_SIZE) -> float:
    """
    Ratio of mean absolute pixel difference *across* 8x8 block boundaries
    to mean absolute pixel difference *within* blocks (i.e. between
    adjacent pixels generally). JPEG's block-based DCT quantization
    introduces a small but characteristic discontinuity right at block
    edges that isn't present in natural (non-block-quantized) gradients.
    This is a structural/grid signal rather than a raw-sharpness one, so
    unlike laplacian_var / fft_highfreq_ratio it degrades more gracefully
    under blur, resize, and additional JPEG re-compression -- all of
    which are part of the robustness eval this feature is meant to
    survive.

    Returns ~1.0 when block-boundary differences are indistinguishable
    from interior differences (no blocking artifact); > 1.0 when
    boundaries are noticeably more discontinuous than interior pixels
    (blocking artifact present).
    """
    h, w = arr.shape
    h_c, w_c = h - (h % block), w - (w % block)
    if h_c < block * 2 or w_c < block * 2:
        return 1.0
    cropped = arr[:h_c, :w_c]

    # Differences straddling a block boundary (vertical boundaries, i.e.
    # column block-index changes) vs. all horizontal-neighbor differences.
    v_edges = np.abs(cropped[:, block::block] - cropped[:, block - 1:-1:block])
    v_inner = np.abs(cropped[:, 1:] - cropped[:, :-1])
    # Same, for horizontal block boundaries (row block-index changes).
    h_edges = np.abs(cropped[block::block, :] - cropped[block - 1:-1:block, :])
    h_inner = np.abs(cropped[1:, :] - cropped[:-1, :])

    if v_edges.size == 0 or h_edges.size == 0:
        return 1.0

    edge_mean = (v_edges.mean() + h_edges.mean()) / 2.0
    inner_mean = (v_inner.mean() + h_inner.mean()) / 2.0 + 1e-8
    return float(edge_mean / inner_mean)


def _resample_periodicity(lap: np.ndarray) -> float:
    """
    Peak-to-mean ratio of the FFT magnitude spectrum of the Laplacian
    (high-frequency residual). Generator upsampling stages (transposed
    conv / repeated interpolation) tend to leave faint periodic structure
    in the residual, which shows up as one or more sharp peaks in its
    frequency spectrum; a real camera photo's high-frequency residual is
    closer to unstructured noise, without a strong periodic peak. `lap`
    is passed in (already computed by the caller) to avoid a redundant
    convolution pass. Like jpeg_blockiness, this targets *structure*
    rather than raw magnitude, so it holds up better than sharpness-only
    features under blur/resize/JPEG/noise.
    """
    mag = np.abs(np.fft.fft2(lap))
    mag[0, 0] = 0.0  # ignore the DC component
    mean_v = mag.mean() + 1e-8
    peak = mag.max()
    return float(peak / mean_v)


def compute_handcrafted_features(img: Image.Image) -> np.ndarray:
    """
    Returns a length-FEATURE_DIM float32 numpy array:
        [grad_direction_entropy, laplacian_var, fft_highfreq_ratio, grad_mag_std,
         radial_falloff_r2, radial_gradient_alignment, channel_saturation_ratio,
         local_blur_inconsistency, jpeg_blockiness, resample_periodicity]
    """
    arr = _to_gray_array(img)
    gx, gy = _sobel_gradients(arr)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    direction_entropy = _grad_direction_entropy(gx, gy)
    lap = _conv3x3(arr, _LAPLACIAN)
    lap_var = float(lap.var())
    fft_ratio = _fft_highfreq_ratio(arr)
    grad_mag_std = float(magnitude.std())

    cy, cx = _light_source_centroid(arr)
    radial_r2 = _radial_falloff_r2(arr, cy, cx)
    radial_alignment = _radial_gradient_alignment(gx, gy, cy, cx)

    rgb_arr = _to_rgb_array(img)
    saturation_ratio = _channel_saturation_ratio(rgb_arr)

    blur_inconsistency = _local_blur_inconsistency(lap)
    blockiness = _jpeg_blockiness(arr)
    periodicity = _resample_periodicity(lap)

    return np.array(
        [direction_entropy, lap_var, fft_ratio, grad_mag_std,
         radial_r2, radial_alignment, saturation_ratio, blur_inconsistency,
         blockiness, periodicity],
        dtype=np.float32,
    )