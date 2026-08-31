# Baowerful

AI-generated image detector built for the TikTok TechJam hackathon. Detects whether an image is AI-generated (AIGC) or authentic, and stays accurate after common post-processing: JPEG re-compression, blur, resizing, noise, color adjustment, and cropping.

## Approach

A frozen, pretrained CLIP image encoder (ViT-B/32, ~87M params, well under the 2B limit) extracts image features. These are concatenated with 10 handcrafted forensic features — signals CLIP wasn't trained to capture, since CLIP optimizes for semantic content, not physical lighting/texture statistics: gradient-direction entropy, Laplacian variance, FFT high-frequency ratio, gradient-magnitude std, radial light-falloff R², radial gradient alignment, channel-saturation ratio, local blur inconsistency, JPEG blockiness, and resample periodicity (see `src/handcrafted_features.py` for the full derivation of each). A small trainable classifier head sits on top of the fused features (~1.7M params by default; ~88.7M total with the frozen backbone) and is trained to predict real vs. AI-generated. During training, images are randomly degraded with the same kinds of transforms the hackathon tests robustness against (JPEG compression, blur, resize, noise, color jitter, cropping), so the model learns to recognize AIGC images even after they've been altered, not just when clean.

## Setup and installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Organize any dataset into this structure before training or evaluating:

```
data/
  train/
    real/   ...authentic images
    fake/   ...AI-generated images
  val/
    real/
    fake/
  test/
    real/
    fake/
```

CIFAKE already matches this layout. For SID_Set (Hugging Face) or WildFake (ModelScope), export their images into the same `real/` + `fake/` folders first (`merge_sid_set.py` does this for SID_Set).

`val/` is used for model selection during training (best-checkpoint tracking). `test/` should be a separate held-out split, never used during training or model selection — it's what the robustness evaluation and error analysis below are run against, so its results reflect genuinely unseen data.

## Steps to reproduce results

Train the classifier head:

```bash
python -m src.train --train_dir data/train --val_dir data/val --epochs 10
```

This saves the best checkpoint to `results/head_best.pt`.

Run inference on any folder of images (this is the required scoring script):

```bash
python -m src.infer --image_dir path/to/images --checkpoint results/head_best.pt --output predictions.json
```

Produces `predictions.json`: a list of `{"image_path": ..., "pred": <0-1 confidence>}` entries.

Produce the robustness evaluation summary (clean vs. each transform/severity) and error analysis samples, against the held-out `data/test` split:

```bash
python -m src.eval_robustness --data_dir data/test --checkpoint results/head_best.pt
```

Writes `docs/robustness_summary.csv` and `docs/error_examples.json`.

## Project structure

```
src/
  model.py                 frozen CLIP backbone + handcrafted features + trainable classifier head
  dataset.py                loads real/ + fake/ image folders
  handcrafted_features.py   the 10 forensic features fused with the CLIP embedding
  augmentations.py          the robustness transforms (train-time random + eval-time sweep)
  train.py                  trains the classifier head
  infer.py                  required inference script -> predictions.json
  eval_robustness.py        clean-vs-transformed accuracy table + error examples
data/       datasets (gitignored, not committed)
results/    checkpoints, training history (gitignored)
docs/       robustness summary and error analysis outputs
```

## Limitations & future work

Trained on CIFAKE (10k real COCO-style photos + 10k AI-generated, held out for validation), the classifier head (frozen CLIP ViT-B/32 + handcrafted features + trainable head) reaches 96.1% best validation accuracy during training, and 96.5% clean accuracy on the separate held-out test split used for the robustness sweep (see `docs/robustness_summary.csv`):

- **Most damaging**: aggressive downscaling (0.25x resize, -11.2pp), heavy blur (sigma=2.0, -10.1pp), and high sensor noise (sigma=0.10, -9.4pp). All three roughly compound in severity -- the model leans on fine-grained texture cues that these transforms destroy first.
- **Most robust**: color jitter (+/-20%, -0.9pp) and light JPEG re-compression (q=90, -0.3pp) barely move accuracy -- these transforms preserve the high-frequency detail the model seems to rely on.
- **Middle ground**: moderate JPEG compression and blur degrade gracefully (a few points per severity step) rather than collapsing, suggesting the training-time random augmentation is helping but not fully closing the gap at the most severe settings.

Representative false positive/negative examples per condition are in `docs/error_examples.json` for a closer look — see `docs/error_analysis_note.md` for a written breakdown, including a small set of images that account for a disproportionate share of the errors across almost every condition.

**Dataset gaps (tested, not just theoretical)**: only CIFAKE was used for training and evaluation. Its real images are low-resolution (32x32) COCO-style photos upscaled to the model's input size, and its fakes come from one generator family. We stress-tested this directly against the hackathon's official demonstration validation set (Section 5.4: COCO val2017 real photos + DALL·E 3 Advanced fakes) — on a 20-image sample, overall accuracy dropped to **40%** (70% on the fakes, but only **10%** on real photos: the model misclassified 9 of 10 genuine, native-resolution photos as AI-generated). See `docs/error_analysis_note.md` for the full breakdown and likely cause (a resolution/texture-statistics mismatch between CIFAKE's upscaled 32x32 real images and true high-resolution photos). SID_Set and WildFake were available and a merge script exists (`merge_sid_set.py`) but they weren't folded into training; doing so — specifically adding native higher-resolution real images — is now the clear highest-value next step, not a speculative one.

**Explainability**: the model currently outputs a single confidence score with no visual explanation of *why*. Given more time, a Grad-CAM-style saliency map over the CLIP features, or attention visualization, would help both debugging and the hackathon's explainability angle.

**Other improvements given more time**: calibrate the output probabilities (predictions cluster near 0 and 1, which may not reflect true confidence, and the error analysis shows the model is often confidently wrong on a small set of hard examples); evaluate against stacked/compounded transforms rather than one at a time, closer to real-world redistribution; and test on images from AIGC generators not represented in CIFAKE to check generalization beyond the training distribution.

## Team contributions

Baowerful was built by Tristan and Jeff.

**Jeff:** project scaffold and pipeline foundation — the CLIP-based detector architecture, dataset loading, robustness augmentations, and the train/infer/eval scripts; the Colab training notebook (Kaggle CIFAKE pull, train, evaluate); training infra fixes (checkpoint-resume behavior, an MPS float64 bug fix) and the classifier-head sizing decisions.

**Tristan:** the handcrafted forensic feature set (`src/handcrafted_features.py`) and the SID_Set merge tooling (`merge_sid_set.py`).

**Handcrafted forensic features — joint design, split implementation:** the 10 features were a shared design effort — brainstorming which physical/statistical properties of images (lighting, texture, compression artifacts) a CLIP embedding likely misses — then split roughly evenly to implement:
- **Tristan:** gradient-direction entropy, gradient-magnitude std, Laplacian variance (sharpness), FFT high-frequency ratio, radial light-source falloff (R²) and radial gradient alignment
- **Jeff:** channel saturation ratio, local blur/sharpness inconsistency, JPEG blockiness, resampling periodicity

Both team members contributed to testing, debugging, and validating the features against the robustness evaluation sweep (JPEG re-compression, blur, resize, noise), and collaborated on integrating the handcrafted features with the CLIP embedding pipeline.
