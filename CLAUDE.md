# VFRW — Visual Font Recognition in the Wild

CV portfolio project: given an image (clean synthetic text or a real photo), detect text regions,
then classify which of 500+ fonts each region is set in. Core thesis: existing font-ID tools
(WhatTheFont, font-identifier sites) are inaccurate — this project's evaluation benchmarks
directly against them on real-photo test data.

Naming convention for write-ups: spell out "Visual Font Recognition in the Wild (VFRW)" once on
first use, then just "VFRW" thereafter.

Full project plan: `C:\Users\ibrah\.claude\plans\i-want-to-create-rustling-swing.md`

## Constraints
- Local GPU: GTX 1650, 4GB VRAM, 16GB RAM. Drives backbone choice (MobileNetV3/EfficientNet-B0)
  and training setup (mixed precision, small batch + grad accumulation).
- Timeline: ~1 week, heavy daily hours.
- Everything must stay free: font sources are free/open-license only; dataset hosting via
  Hugging Face Datasets Hub (or Kaggle) free tier.

## Architecture

```
Input image -> [Stage A: Text Detection] -> boxes -> [Stage B: Crop/normalize]
            -> [Stage C: Font Classifier (custom-trained CNN, 500+ classes)] -> prediction
```

- Stage A: pretrained detector first (EasyOCR/CRAFT), custom-trained detector added later as a
  bounded benchmark experiment (not a replacement).
- Stage C is the core deliverable — fully custom-trained, primarily on synthetic data with
  augmentation to close the synthetic-to-real gap.

## Setup
- Python 3.13, venv at `.venv/` (already created — do not recreate).
- Install/update deps: `./.venv/Scripts/python.exe -m pip install -r requirements.txt`
- After adding a new package: `./.venv/Scripts/python.exe -m pip freeze > requirements.txt` to keep it current.
- Current deps (as of Phase 1 setup): pillow, albumentations (opencv-python-headless, scipy, pydantic),
  requests, nltk, pandas, pyyaml, tqdm, huggingface_hub.

## Folder layout
- `data_gen/` — synthetic text-image rendering pipeline (fonts -> labeled crops)
- `data/` — gitignored; fonts, synthetic dataset, real-photo test set
- `models/` — model definitions + checkpoints (checkpoints gitignored)
- `training/` — training scripts
- `eval/` — evaluation scripts, incl. benchmark vs. existing font-ID tools
- `web_demo/` — FastAPI backend + frontend for the live demo
- `notebooks/` — exploratory work backing the write-up

## Working style
- Explain the theory behind each phase's decisions while building, not just deliver code —
  this project is a learning exercise as much as a deliverable.
- Smoke-test every pipeline at small scale (e.g. 10 fonts, few epochs) before scaling to the
  full run (500 fonts, full dataset).
- Long training runs go in the background; report on completion rather than being polled.
- Plan each major phase separately rather than re-deriving the whole project plan each time.
