"""
Background generation for synthetic crops: mostly procedural (solid/gradient/noise),
supplemented by a cached one-time batch of free Lorem Picsum stock photos, per Phase 1 plan.
"""

import random
from pathlib import Path

import numpy as np
import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = REPO_ROOT / "data" / "backgrounds" / "stock"
STOCK_CACHE_COUNT = 300  # one-time downloaded batch, reused across the whole generation run

PROCEDURAL_PROB = 0.75  # majority procedural, per plan


def _random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def solid_background(size) -> Image.Image:
    return Image.new("RGB", size, _random_color())


def gradient_background(size) -> Image.Image:
    w, h = size
    c1 = np.array(_random_color(), dtype=np.float32)
    c2 = np.array(_random_color(), dtype=np.float32)
    if random.random() < 0.5:
        # linear horizontal gradient
        t = np.linspace(0, 1, w).reshape(1, w, 1)
    else:
        # linear vertical gradient
        t = np.linspace(0, 1, h).reshape(h, 1, 1)
        c1, c2 = c1.reshape(1, 1, 3), c2.reshape(1, 1, 3)
    arr = (c1 * (1 - t) + c2 * t)
    arr = np.broadcast_to(arr, (h, w, 3)).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def noise_texture_background(size) -> Image.Image:
    w, h = size
    base_color = np.array(_random_color(), dtype=np.float32)
    noise = np.random.normal(loc=0, scale=random.uniform(10, 40), size=(h, w, 3))
    arr = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def procedural_background(size) -> Image.Image:
    fn = random.choice([solid_background, gradient_background, noise_texture_background])
    return fn(size)


def ensure_stock_backgrounds(count=STOCK_CACHE_COUNT):
    STOCK_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(STOCK_DIR.glob("*.jpg"))
    if len(existing) >= count:
        return
    needed = count - len(existing)
    print(f"downloading {needed} stock backgrounds from picsum.photos...")
    for i in range(len(existing), count):
        dest = STOCK_DIR / f"{i:04d}.jpg"
        if dest.exists():
            continue
        url = f"https://picsum.photos/seed/vfrw{i}/512/512"
        try:
            r = requests.get(url, timeout=30, headers={"Accept-Encoding": "identity"})
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"skip {i}: {e}")


def stock_background(size) -> Image.Image:
    files = list(STOCK_DIR.glob("*.jpg"))
    if not files:
        return procedural_background(size)  # fallback if cache not populated
    path = random.choice(files)
    img = Image.open(path).convert("RGB")
    return img.resize(size)


def sample_background(size) -> Image.Image:
    if random.random() < PROCEDURAL_PROB:
        return procedural_background(size)
    return stock_background(size)


if __name__ == "__main__":
    ensure_stock_backgrounds(count=20)  # small smoke-test batch
    out_dir = REPO_ROOT / "data_gen" / "_bg_smoketest"
    out_dir.mkdir(exist_ok=True)
    for i in range(10):
        img = sample_background((224, 224))
        img.save(out_dir / f"bg_{i}.png")
    print(f"wrote 10 sample backgrounds to {out_dir}")
