# Baowerful — Written Project Description (Devpost)

## Why this matters

AI-generated images rarely reach a viewer in their original form — they get re-compressed by messaging apps, thumbnailed by feed algorithms, cropped for profile pictures, and color-adjusted by filters, long before anyone tries to tell whether they're real. A detector that only performs well on clean, unmodified images is solving an easier problem than the one platforms actually face. Baowerful is built around that gap: robustness to realistic post-processing is treated as a first-class design constraint, not an afterthought measured after the fact.

## How our solution addresses the problem statement

Every one of the six transform families in the problem statement's robustness table — JPEG re-compression, Gaussian blur, resize/thumbnailing, Gaussian noise, color jitter, and center crop — is applied at training time as a random augmentation, so the model is explicitly trained to recognize AI-generated content after it's been degraded, not just when it's clean. The same six families, at every severity level the spec lists, are then used as the evaluation sweep (see the Robustness Evaluation Summary).

On a held-out test split, the model reaches **96.5% clean accuracy**, staying within 0.3–1pp of that on light JPEG compression and color jitter, and degrading gracefully rather than collapsing under all but the harshest conditions (0.25× downscaling, heavy blur, heavy noise — see the robustness table for exact numbers).

In the interest of an honest evaluation, we also stress-tested against the hackathon's official demonstration validation set (Section 5.4: COCO val2017 + DALL·E 3 Advanced) rather than only reporting numbers on our own training distribution. That test surfaced a real generalization gap — full details and root-cause analysis are in `docs/error_analysis_note.md` and the README's Limitations section, along with the concrete next step it points to.

**Impact & relevance.** The direct use case is content moderation and platform trust: a lightweight pre-filter that flags likely-AIGC content for review, or that surfaces a confidence score alongside a post, gives platforms and users a practical signal without claiming certainty. Because it's built and evaluated against realistic redistribution conditions rather than pristine benchmark images, it's aimed at the version of this problem that actually shows up in a feed — not just a lab setting.

**Feasibility & practicality.** The architecture is deliberately cheap to own: the ~87M-param CLIP backbone is frozen and never fine-tuned, so the only trainable component is a small (~1.7M-param) head — fast to retrain as new generators or datasets emerge, without needing to touch or re-host a large model. Inference is a single forward pass per image (no ensembling, no test-time augmentation), and the full model runs comfortably on a single consumer GPU or even CPU for small batches, which matters for a moderation pipeline that needs to process volume, not just get one image right in a demo.

## Approach

Core insight: a general-purpose semantic encoder like CLIP is a strong prior for "what's in the image," but it wasn't trained to notice the low-level physical inconsistencies (lighting, texture, compression artifacts) that generators tend to leave behind — so we pair it with a small set of forensic features built specifically to catch those, rather than relying on CLIP alone or on a forensics-only approach that would miss semantic context.

- **Backbone:** a frozen, pretrained CLIP ViT-B/32 image encoder (~87M params) — never fine-tuned, so training is fast and needs comparatively little data.
- **Handcrafted forensic features:** 10 hand-designed signals CLIP wasn't trained to capture, since CLIP optimizes for semantic content, not physical lighting/texture statistics: gradient-direction entropy, Laplacian variance, FFT high-frequency ratio, gradient-magnitude std, radial light-falloff R², radial gradient alignment, channel-saturation ratio, local blur inconsistency, JPEG blockiness, and resample periodicity. These are concatenated with the CLIP embedding (after batch-normalizing the handcrafted features, since they sit on a very different numeric scale).
- **Classifier head:** a small trainable head (a "widen-then-halve" MLP, ~1.7M params in the default configuration) on top of the fused 522-dim feature vector. Total model size is ~88.7M params — comfortably under the hackathon's 2B-parameter limit. A leaner ablation head (~400K params) is also implemented for comparison.
- **Robustness-by-training:** during training, each image is randomly degraded with the same transform families the hackathon tests against, so robustness is trained in rather than only measured after the fact.

## Development tools

Python scripts developed and run from the command line, plus a Colab notebook (`notebooks/colab_training.ipynb`) for GPU-accelerated training runs.

## Models / APIs used

CLIP ViT-B/32, via `open_clip`, using OpenAI's pretrained weights (frozen, used purely as a feature extractor).

## Libraries and frameworks used

PyTorch, torchvision, `open_clip_torch`, scikit-learn, NumPy, pandas, Pillow, tqdm.

## Datasets and assets used

- **CIFAKE** (real COCO-style photos vs. AI-generated images) — the dataset actually used for training and evaluation in the current results.
- **SID_Set** (Hugging Face) — a merge script (`merge_sid_set.py`) exists to fold it into the `data/train` / `data/test` layout, but per the current README this hasn't been incorporated into the reported results yet.
- **WildFake** (ModelScope) — available per the hackathon's resource list; not yet incorporated.
- The hackathon's WildFake-subset validation set (COCO val2017 + DALL·E Advanced) was noted as demonstration-only and excluded from training, per the problem statement.

*Note: since only CIFAKE was actually used for the reported 96.5% clean accuracy and the robustness sweep, that's the honest scope to state in the Devpost submission — mentioning SID_Set/WildFake as "available resources referenced" rather than "used," unless the team incorporates them before submission.*
