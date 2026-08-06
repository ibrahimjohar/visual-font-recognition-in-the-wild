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


# minimum glyph height (pixels, in the final size x size output) below which text is no longer
# reliably legible -- a real bug let ~25% of the dataset ship under this without anyone noticing,
# because the old letterbox scaled by max(canvas_w, canvas_h): for long text, canvas width dwarfs
# height, so a single-line sentence could get crushed to ~6-8px tall. this floor is asserted in
# _crop_or_pad_to_square below so a future change to font_size/margin ranges can't silently
# regress this again without the assertion firing during generation, not months later in an audit.
MIN_LEGIBLE_GLYPH_HEIGHT = 20


def _crop_or_pad_to_square(img: Image.Image, size: int, font_size: int) -> Image.Image:
    """Scales by HEIGHT ONLY (not max(w, h)), then crops or pads width to reach size x size.
    This decouples glyph height from text length entirely -- a 5-character word and a
    90-character sentence both come out with the same effective glyph height, since neither
    text width nor number of characters enters the height calculation at all. Long text simply
    shows a (randomly positioned) horizontal window of the sentence rather than being crushed to
    fit the whole thing -- which is also a more realistic training signal for real photos, where
    text is routinely cropped by the frame edge rather than always fully visible."""
    w, h = img.size
    scale = size / h
    new_w, new_h = max(1, int(w * scale)), size
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    effective_glyph_height = font_size * scale
    assert effective_glyph_height >= MIN_LEGIBLE_GLYPH_HEIGHT, (
        f"effective glyph height {effective_glyph_height:.1f}px fell below the "
        f"{MIN_LEGIBLE_GLYPH_HEIGHT}px legibility floor (font_size={font_size}, scale={scale:.3f}) "
        f"-- font_size or margin ranges changed without re-checking this invariant"
    )

    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    if new_w <= size:
        paste_x = (size - new_w) // 2
        canvas.paste(resized, (paste_x, 0))
    else:
        crop_x = random.randint(0, new_w - size)
        cropped = resized.crop((crop_x, 0, crop_x + size, size))
        canvas.paste(cropped, (0, 0))
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


def render_crop(family: str, role: str, manifest=None, text: str | None = None) -> Image.Image:
    # text override lets the regenerate script re-render an existing manifest row with the same
    # label (fixed letterbox logic, fresh random font size/margin/background/augmentation) rather
    # than resampling new corpus content and silently changing what that row's text field means
    if text is None:
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

    letterboxed = _crop_or_pad_to_square(canvas, OUTPUT_SIZE, font_size)

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
