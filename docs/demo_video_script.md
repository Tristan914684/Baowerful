# Demo Video — Final Script & Navigation

Read-aloud script on the left of each beat, exactly what to have on screen and do on the right. Target ~2:30. Numbers are real, pulled from your repo (`predictions.json`, `docs/robustness_summary.csv`) — nothing here is made up, so you can say it with confidence.

**Live demo uses the CIFAKE sample set at `data/demo_video/` (8 images) — deliberately.** You now also have real WildFake validation images at `data/wildfake_val_demo/` (10 COCO val2017 real photos + 10 DALL·E 3 Advanced fakes), and running inference on those was genuinely useful — it surfaced a real generalization gap (see the 2:00–2:20 beat below, and `docs/error_analysis_note.md`). But that same run only gets 40% accuracy, so it's the wrong thing to show as your *working* demo. Keep the CIFAKE folder for the live run; the WildFake finding gets a verbal mention instead, framed as rigor rather than failure.

---

## Before you hit record

1. Open a Terminal window, sized large, font bumped up (Terminal → Preferences → Profile → Text → increase font size to ~16–18pt so it reads on video).
2. `cd ~/Documents/Baowerful && source .venv/bin/activate`
3. Run the inference command **once now**, silently, so the CLIP model is warm/cached and your recorded take doesn't sit on a slow first load:
   `python -m src.infer --image_dir data/demo_video --checkpoint results/head_best.pt --output data/demo_video_predictions.json`
4. Have `docs/robustness_chart.png` open in Preview (or the image viewer of your choice) in a second window/tab, ready to switch to.
5. Have `README.md` open in your editor (or the GitHub repo page in a browser tab) as a third window, scrolled to the top, ready to switch to.
6. Start your screen recorder (QuickTime: File → New Screen Recording). Leave 2 seconds of silence before you start talking and 2 seconds after you finish, for clean editing.

---

## 0:00–0:15 — Hook (camera or a title card, your call)

> "AI-generated images don't stay clean once they're online — they get compressed, resized, filtered, re-uploaded. We built Baowerful: a detector that tells real from AI-generated images, and keeps working after that kind of real-world processing."

**Navigate:** if using a title card, this is the only slide you need before switching to the terminal.

## 0:15–0:45 — Approach

> "The core idea: we use a frozen CLIP image encoder — it's great at understanding what's semantically in an image, but it was never trained to notice the physical fingerprints AI generators leave behind, like unnatural lighting or texture. So we pair it with ten handcrafted forensic features — things like lighting-direction consistency and JPEG blockiness — and train a small classifier head on top of both, combined. The whole model is about 89 million parameters, well under the hackathon's 2 billion limit. And during training, we randomly degrade every image with the same kinds of transforms the hackathon tests — compression, blur, noise, resizing, color changes, cropping — so robustness gets trained in, not just measured afterward."

**Navigate:** switch to `README.md` (or your editor), scrolled to the "Approach" section. You can point the cursor at it as you talk, or just have it visible as a backdrop.

## 0:45–1:30 — Live demo

**Navigate:** switch to the Terminal window.

Say, while typing (or just before running it):

> "Here's the script the hackathon requires — it takes a folder of images and outputs a confidence score for each one."

Type and run:
```
python -m src.infer --image_dir data/demo_video --checkpoint results/head_best.pt --output data/demo_video_predictions.json
```

(This should run fast since you warmed it up already.) Then:
```
cat data/demo_video_predictions.json
```

While the JSON prints, say:

> "Real photos score close to zero, AI-generated images score close to one. On a held-out batch of 100 images — 50 real, 50 AI-generated — this exact script got 98% accuracy. It's not perfect — in that same batch, one real photo scored 0.99 and got flagged as fake, and one AI image scored just 0.52, barely over the threshold — but those are the edge cases, not the norm."

## 1:30–2:00 — Robustness results

**Navigate:** switch to the `docs/robustness_chart.png` window/tab.

> "The hackathon specifically tests robustness to compression, blur, resizing, noise, color changes, and cropping, so we evaluated against every one of those. Clean accuracy is 96.5%. Light JPEG compression and color jitter barely move it — under a point off. The harder conditions — aggressive downscaling, heavy blur, heavy noise — bring it down to the mid-to-high 80s, which is where we'd focus improvement next."

## 2:00–2:20 — Error analysis & limitations

**Navigate:** stay on the chart, or switch back to the README's "Limitations" section.

> "Looking at where it's wrong: a small number of images account for most of the errors, and the model tends to be confidently wrong on them rather than borderline — that's a calibration issue we flag in the write-up. We also stress-tested against the hackathon's official demonstration validation set — real COCO photos and DALL·E 3 images — and found a real generalization gap: the model, trained only on CIFAKE's low-resolution images, misclassified most full-resolution real photos as AI-generated. We think that's a resolution mismatch in the training data, and it's the top item on our future-work list — full breakdown is in the repo."

## 2:20–2:30 — Close

> "Full code, setup instructions, and this robustness breakdown are all in the repo. Thanks for watching."

**Navigate:** end recording here, with ~1 second of silence before stopping.

---

## After recording

- [ ] Trim the dead air at the start/end
- [ ] Upload to YouTube, visibility set to **Public** (hackathon requirement — private or unlisted won't count)
- [ ] Copy the YouTube link into your Devpost description
- [ ] Double-check no copyrighted background music or third-party trademarks are in the video (hackathon requirement)
