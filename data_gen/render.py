"""
Renders a single synthetic text crop: text_corpus content -> font_loader font -> composited
onto a backgrounds.py background -> letterboxed to a fixed square -> augmented via albumentations.
"""

import random

import albumentations as A
import numpy as np
from PIL import Image, ImageDraw

from backgrounds import sample_background
from font_loader import get_font
from text_corpus import sample_text

OUTPUT_SIZE = 224  # final square size, matches Phase 2 backbone input


def _random_text_color(bg_sample_color):
    # bias toward contrast with the background so most crops are legible, but allow some
    # low-contrast "hard" cases for realism
    if random.random() < 0.85:
        brightness = sum(bg_sample_color) / 3
        if brightness > 127:
            return tuple(random.randint(0, 60) for _ in range(3))
        else:
            return tuple(random.randint(195, 255) for _ in range(3))
    return tuple(random.randint(0, 255) for _ in range(3))


def _letterbox(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


_AUGMENT = A.Compose([
    A.Affine(rotate=(-8, 8), shear=(-6, 6), scale=(0.9, 1.1), p=0.6),
    A.Perspective(scale=(0.02, 0.08), p=0.4),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.35),
    A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.ImageCompression(quality_range=(35, 90), p=0.5),
])


def render_crop(family: str, role: str, manifest=None) -> Image.Image:
    text = sample_text()
    font_size = random.randint(28, 60)
    font = get_font(family, role, font_size, manifest)

    # measure text first (throwaway canvas), then size the real canvas around it with a random
    # margin -- avoids both clipping long text and wasting most of the frame as letterbox padding
    measurer = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox = measurer.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    margin_x = random.randint(15, 50)
    margin_y = random.randint(15, 50)
    canvas_size = (text_w + 2 * margin_x, text_h + 2 * margin_y)

    bg = sample_background(canvas_size)
    bg_sample_color = bg.getpixel((canvas_size[0] // 2, canvas_size[1] // 2))
    color = _random_text_color(bg_sample_color)

    canvas = bg.copy()
    draw = ImageDraw.Draw(canvas)
    pos_x = margin_x - bbox[0]
    pos_y = margin_y - bbox[1]
    draw.text((pos_x, pos_y), text, font=font, fill=color)

    letterboxed = _letterbox(canvas, OUTPUT_SIZE)

    arr = np.array(letterboxed)
    augmented = _AUGMENT(image=arr)["image"]
    return Image.fromarray(augmented), text


if __name__ == "__main__":
    from pathlib import Path
    out_dir = Path(__file__).resolve().parent / "_render_smoketest"
    out_dir.mkdir(exist_ok=True)
    for i, (fam, role) in enumerate([("roboto", "Regular"), ("roboto", "BoldItalic"), ("lato", "Bold")]):
        for j in range(3):
            img, text = render_crop(fam, role)
            img.save(out_dir / f"{fam}_{role}_{j}.png")
            print(fam, role, repr(text))
    print(f"wrote samples to {out_dir}")
