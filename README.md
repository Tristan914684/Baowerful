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

TODO: fill in after training runs -- e.g. which transforms hurt accuracy most, dataset gaps, false positive/negative patterns.

## Team contributions

TODO.
