# Error Analysis Note

## Generalization stress test: the hackathon's official demonstration validation set

Before writing up the failure modes found *within* CIFAKE, we ran the trained checkpoint against a small sample of the hackathon's official "for demonstration purposes only" validation set (Section 5.4): 10 real photos from COCO val2017 and 10 AI-generated images from DALL·E 3 Advanced, run through the exact required `src.infer` script.

| | Correct | Accuracy |
|---|---|---|
| Fake (DALL·E 3 Advanced) | 7 / 10 | 70% |
| Real (COCO val2017) | 1 / 10 | **10%** |
| Overall | 8 / 20 | **40%** |

This is a real, quantified generalization gap, not a rounding difference from the 96.5% clean accuracy reported on CIFAKE — the model misclassified 9 of 10 genuine, full-resolution real photos as AI-generated, several with over 95% confidence (average predicted "fake" probability on these real images: 0.80, nearly as high as the 0.70 average on the actual fakes in this sample).

**Why:** CIFAKE's real images are native 32×32 pixel photos, upscaled to CLIP's 224×224 input before both training and evaluation. COCO val2017's real photos are natively high-resolution, downsized to 224×224 instead. Those two operations leave different texture, sharpness, and compression statistics behind — and several of the handcrafted forensic features (Laplacian variance, FFT high-frequency ratio, JPEG blockiness) are exactly the kind of signal sensitive to that difference. The model appears to have learned "sharp, detailed, high-frequency content = AI-generated," which holds within CIFAKE's specific image pipeline but is backwards once real images arrive at native higher resolution.

**What this means:** the 96.5% clean accuracy and full robustness table in this repo are accurate *for the CIFAKE distribution they were measured on*, but should not be read as a general claim about real-world deployment accuracy. This is precisely the "dataset gaps" limitation already flagged in the README, now tested and quantified rather than speculative. The highest-value next step is retraining and re-evaluating with a training/eval mix that includes native higher-resolution real photos (COCO val2017 and/or SID_Set/WildFake, both already available per Section 5.4), not just CIFAKE.

*(Sample size caveat: 10 images per class is small and meant as a fast stress test, not a precise accuracy estimate — the true generalization accuracy on this domain could reasonably be a range around 40%, not exactly 40%. The direction and size of the gap is the important finding, not the third significant figure.)*

## Within-CIFAKE error analysis

Based on `docs/error_examples.json`, produced by the same `python -m src.eval_robustness` run as the robustness summary. For each of the 15 conditions (clean + 14 transform/severity combinations), the script saves the model's worst false positives (real images predicted as AI-generated) and false negatives (AI-generated images predicted as real) — up to 10 for clean, up to 5 for each transformed condition.

## Headline: a small set of images account for most of the errors

Rather than errors being spread evenly across the test set, a handful of images show up as errors again and again, across nearly every transform:

- `real/0001 (3).jpg` appears as a false positive in **9 of the 15** logged conditions, with confidence the model is "sure" it's AI-generated ranging from 0.59 to 0.96.
- `real/0002 (10).jpg` is a false positive in 9 of 15 conditions.
- `fake/10 (6).jpg` is a false negative in **10 of the 15** conditions, with the model rating it only 4–40% likely to be AI-generated even on the clean image.
- `fake/102 (9).jpg` is a false negative in 9 of 15 conditions, often scored under 0.15.

(Caveat: since only the top 5–10 errors per condition are logged, this "recurrence count" is partly a function of the cap — an image that's always among the worst offenders will show up in every list. It's still a meaningful signal that these are the model's persistently hardest examples, not one-off noise, but it isn't a full count over the whole test set.)

**Trade-off this points to:** confidence calibration, not just accuracy. These aren't borderline 0.51-vs-0.49 misses — the model is often *very* confident and *very* wrong on this small set. The README already flags that predictions cluster near 0 and 1; this data backs that up and shows it's not harmless overconfidence — it's concentrated on specific hard examples that stay hard across almost every transform.

## Representative cases

**False positive — `real/0001 (3).jpg`:** scored 0.87 on the clean image (already misclassified), and stays wrong under nearly every transform: 0.70 (JPEG q90), 0.96 (JPEG q50), 0.82 (blur σ=0.5), 0.79 (center crop 80%). This is a real photo the model treats as AI-generated regardless of what's done to it — suggesting the model has latched onto some property of the image itself (e.g. lighting or texture that reads as "too clean" to the handcrafted forensic features) rather than something introduced by post-processing.

**False negative — `fake/102 (9).jpg`:** scored 0.12 on the clean image and stays low under most transforms: 0.07 (JPEG q90), 0.10 (JPEG q70), 0.04 (noise σ=0.02), 0.15 (center crop). The model is confidently wrong in the *safe* direction here — it's convinced this AI-generated image is real.

**Condition-driven false negative — `fake/10 (6).jpg` under color jitter:** confidence rises from 0.18 (clean) to 0.40 (color jitter ±20%) — still a miss, but a case where a specific transform pushes an already-borderline example over the decision boundary, rather than the image being hard in every condition.

## False positives vs. false negatives — which is worse here

For a hackathon-scale AIGC detector, the two error types have different real-world costs: a false positive (flagging a real photo as AI-generated) risks wrongly labeling or removing authentic content, while a false negative (missing an AI-generated image) lets synthetic content through unflagged. This model doesn't show a strong skew toward one over the other — both the top false-positive and top false-negative examples recur with similar frequency (9–10 of 15 conditions) — but a production deployment would need to pick an operating threshold (currently 0.5) deliberately based on which error type matters more for the use case, rather than assuming the default threshold is right.

## What this suggests for next steps

1. **Retrain/re-evaluate with native higher-resolution real images included** (COCO val2017, SID_Set, and/or WildFake, all available per the hackathon's resource list) — this is now a confirmed gap, not a hypothetical one (see the generalization stress test above), and it's the single highest-value fix.
2. **Calibrate confidence**, not just threshold accuracy — the model's near-0/near-1 clustering (noted in the README) means a single global threshold can't distinguish "confidently right" from "confidently wrong on a hard example."
3. **Inspect the persistently-hard images directly** (`real/0001 (3).jpg`, `real/0002 (10).jpg`, `fake/10 (6).jpg`, `fake/102 (9).jpg`) to look for a shared visual property — this is a small enough set to check by eye.
