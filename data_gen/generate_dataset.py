"""
Phase 1 driver: renders the full (or a limited smoke-test subset of the) synthetic dataset.

Output layout (ImageFolder-compatible):
  data/synthetic/<family>__<role>/xxxxxxxx.jpg

Also writes:
  data/synthetic/manifest.csv   (path, family, role, class_name, split, source_mode)
  data_gen/config.yaml          (generation parameters, for reproducibility)

Usage:
  python generate_dataset.py --max-classes 10 --images-per-class 20   # smoke test
  python generate_dataset.py --images-per-class 500                  # full run
"""

import argparse
import csv
import random
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

VAL_SPLIT = 0.15


def class_dir_name(family: str, role: str) -> str:
    return f"{family}__{role}"


def generate(max_classes: int | None, images_per_class: int, seed: int = 42, stock_bg_count: int = 300):
    random.seed(seed)

    manifest = load_manifest()
    classes = build_class_list(manifest)
    random.shuffle(classes)
    if max_classes is not None:
        classes = classes[:max_classes]

    ensure_stock_backgrounds(count=stock_bg_count)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for family, role in tqdm(classes, desc="classes"):
        class_name = class_dir_name(family, role)
        class_dir = OUT_DIR / class_name
        class_dir.mkdir(exist_ok=True)

        n_val = max(1, int(images_per_class * VAL_SPLIT))
        n_train = images_per_class - n_val
        split_assignment = ["train"] * n_train + ["val"] * n_val

        for i in range(images_per_class):
            img, text = render_crop(family, role, manifest)
            fname = f"{i:05d}.jpg"
            rel_path = f"{class_name}/{fname}"
            img.save(class_dir / fname, quality=90)
            rows.append({
                "path": rel_path,
                "family": family,
                "role": role,
                "class_name": class_name,
                "split": split_assignment[i],
                "text": text,
            })

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "family", "role", "class_name", "split", "text"])
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "seed": seed,
        "num_classes": len(classes),
        "images_per_class": images_per_class,
        "total_images": len(rows),
        "val_split": VAL_SPLIT,
        "output_size": 224,
        "max_classes_arg": max_classes,
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

    print(f"\ngenerated {len(rows)} images across {len(classes)} classes -> {OUT_DIR}")
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
