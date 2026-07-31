"""
Phase 1 driver: renders the full (or a limited smoke-test subset of the) synthetic dataset.

Output layout (ImageFolder-compatible):
  data/synthetic/<family>__<role>/xxxxxxxx.jpg

Also writes:
  data/synthetic/manifest.csv   (path, family, role, class_name, split, source_mode)
  data_gen/config.yaml          (generation parameters, for reproducibility)

Resumability: each class is atomic -- its manifest rows are only appended (and flushed to
disk) after all of its images have been saved successfully. On a fresh run, any class folder
that exists but has no matching complete entry in the manifest is treated as a leftover partial
class from an interrupted run: it's deleted and regenerated from scratch. Completed classes are
skipped entirely. This means killing the process at any point (laptop sleep/shutdown, crash)
loses at most the one class that was in progress -- everything before it stays on disk and
labeled, and re-running this script just continues from there.

Usage:
  python generate_dataset.py --max-classes 10 --images-per-class 20   # smoke test
  python generate_dataset.py --images-per-class 150                  # full run
"""

import argparse
import csv
import random
import shutil
import yaml
from pathlib import Path

from tqdm import tqdm

from font_loader import build_class_list, load_manifest
from render import render_crop
from backgrounds import ensure_stock_backgrounds

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "synthetic"
MANIFEST_CSV = OUT_DIR / "manifest.csv"
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
FIELDNAMES = ["path", "family", "role", "class_name", "split", "text"]

VAL_SPLIT = 0.15


def class_dir_name(family: str, role: str) -> str:
    return f"{family}__{role}"


def load_completed_counts() -> dict[str, int]:
    """class_name -> number of manifest rows already recorded for it."""
    if not MANIFEST_CSV.exists():
        return {}
    counts = {}
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["class_name"]] = counts.get(row["class_name"], 0) + 1
    return counts


def generate(max_classes: int | None, images_per_class: int, seed: int = 42, stock_bg_count: int = 300):
    random.seed(seed)

    manifest = load_manifest()
    classes = build_class_list(manifest)
    classes.sort()  # stable, deterministic order so resuming skips the right ones
    random.Random(seed).shuffle(classes)
    if max_classes is not None:
        classes = classes[:max_classes]

    ensure_stock_backgrounds(count=stock_bg_count)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # config is safe to write up-front: it only reflects the run's parameters, not progress
    config = {
        "seed": seed,
        "num_classes": len(classes),
        "images_per_class": images_per_class,
        "val_split": VAL_SPLIT,
        "output_size": 224,
        "max_classes_arg": max_classes,
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

    completed = load_completed_counts()
    manifest_is_new = not MANIFEST_CSV.exists()
    manifest_file = open(MANIFEST_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_file, fieldnames=FIELDNAMES)
    if manifest_is_new:
        writer.writeheader()
        manifest_file.flush()

    skipped = 0
    generated_classes = 0
    total_images_written = 0

    try:
        for family, role in tqdm(classes, desc="classes"):
            class_name = class_dir_name(family, role)
            class_dir = OUT_DIR / class_name

            if completed.get(class_name, 0) == images_per_class:
                skipped += 1
                continue  # already fully done from a previous run

            if class_dir.exists():
                shutil.rmtree(class_dir)  # partial/stale leftover from an interrupted run
            class_dir.mkdir()

            n_val = max(1, int(images_per_class * VAL_SPLIT))
            n_train = images_per_class - n_val
            split_assignment = ["train"] * n_train + ["val"] * n_val

            class_rows = []
            for i in range(images_per_class):
                img, text = render_crop(family, role, manifest)
                fname = f"{i:05d}.jpg"
                img.save(class_dir / fname, quality=90)
                class_rows.append({
                    "path": f"{class_name}/{fname}",
                    "family": family,
                    "role": role,
                    "class_name": class_name,
                    "split": split_assignment[i],
                    "text": text,
                })

            writer.writerows(class_rows)
            manifest_file.flush()  # class is now atomically recorded as complete
            generated_classes += 1
            total_images_written += len(class_rows)
    finally:
        manifest_file.close()

    print(f"\nthis run: generated {generated_classes} classes ({total_images_written} images), "
          f"skipped {skipped} already-complete classes")
    print(f"manifest -> {MANIFEST_CSV}")
    print(f"config -> {CONFIG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-classes", type=int, default=None, help="limit classes (smoke test)")
    parser.add_argument("--images-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stock-bg-count", type=int, default=300)
    args = parser.parse_args()

    generate(args.max_classes, args.images_per_class, args.seed, args.stock_bg_count)
