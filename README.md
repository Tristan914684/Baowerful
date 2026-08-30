# Baowerful

AI-generated image detector built for the TikTok TechJam hackathon. Detects whether an image is AI-generated (AIGC) or authentic, and stays accurate after common post-processing: JPEG re-compression, blur, resizing, noise, color adjustment, and cropping.

## Approach

A frozen, pretrained CLIP image encoder (ViT-B/32, ~87M params, well under the 2B limit) extracts image features. A small trainable classifier head (two linear layers) sits on top and is trained to predict real vs. AI-generated. During training, images are randomly degraded with the same kinds of transforms the hackathon tests robustness against (JPEG compression, blur, resize, noise, color jitter, cropping), so the model learns to recognize AIGC images even after they've been altered, not just when clean.

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
```

CIFAKE already matches this layout. For SID_Set (Hugging Face) or WildFake (ModelScope), export their images into the same `real/` + `fake/` folders first.

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

Produce the robustness evaluation summary (clean vs. each transform/severity) and error analysis samples:

```bash
python -m src.eval_robustness --data_dir data/val --checkpoint results/head_best.pt
```

Writes `docs/robustness_summary.csv` and `docs/error_examples.json`.

## Project structure

```
src/
  model.py            frozen CLIP backbone + trainable classifier head
  dataset.py           loads real/ + fake/ image folders
  augmentations.py     the robustness transforms (train-time random + eval-time sweep)
  train.py             trains the classifier head
  infer.py             required inference script -> predictions.json
  eval_robustness.py   clean-vs-transformed accuracy table + error examples
data/       datasets (gitignored, not committed)
results/    checkpoints, training history (gitignored)
docs/       robustness summary and error analysis outputs
```

## Limitations & future work

Trained on CIFAKE (10k real COCO-style photos + 10k AI-generated, held out for validation), the small classifier head (frozen CLIP ViT-B/32 + 512->256->1) reaches 96.1% clean validation accuracy. Robustness against the hackathon's transform sweep (see `docs/robustness_summary.csv`):

- **Most damaging**: aggressive downscaling (0.25x resize, -12.1pp), heavy blur (sigma=2.0, -9.2pp), and high sensor noise (sigma=0.10, -8.7pp). All three roughly compound in severity -- the model leans on fine-grained texture cues that these transforms destroy first.
- **Most robust**: color jitter (+/-20%, -0.5pp) and light JPEG re-compression (q=90, -0.1pp) barely move accuracy -- these transforms preserve the high-frequency detail CLIP features seem to rely on.
- **Middle ground**: moderate JPEG compression and blur degrade gracefully (a few points per severity step) rather than collapsing, suggesting the training-time random augmentation is helping but not fully closing the gap at the most severe settings.

Representative false positive/negative examples per condition are in `docs/error_examples.json` for a closer look.

**Dataset gaps**: only CIFAKE was used for training and evaluation. Its real images are low-resolution (32x32) COCO-style photos and its fakes come from one generator family, so the model hasn't been exposed to higher-resolution images, other generator architectures (diffusion vs. GAN), or more recent AIGC tools -- generalization to those is untested. SID_Set and WildFake were available but not incorporated; folding them in (per the README's data layout) is the highest-value next step.

**Explainability**: the model currently outputs a single confidence score with no visual explanation of *why*. Given more time, a Grad-CAM-style saliency map over the CLIP features, or attention visualization, would help both debugging and the hackathon's explainability angle.

**Other improvements given more time**: calibrate the output probabilities (predictions cluster near 0 and 1, which may not reflect true confidence); evaluate against stacked/compounded transforms rather than one at a time, closer to real-world redistribution; and test on images from AIGC generators not represented in CIFAKE to check generalization beyond the training distribution.

## Team contributions

TODO: list team members and what each person worked on (data pipeline, model/training, robustness eval, writeup, etc.).
