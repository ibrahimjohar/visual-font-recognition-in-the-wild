"""
Resolves (family, role) -> loaded PIL ImageFont, handling both static per-role files
(Regular.ttf, Bold.ttf, ...) and variable-font named-instance selection (variable_upright.ttf
holds Regular/Bold via Pillow's set_variation_by_name, variable_italic.ttf holds
Italic/BoldItalic the same way). Also builds the final (family, role) class list.
"""

import json
from collections import Counter
from pathlib import Path

from PIL import ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data_gen" / "font_manifest.json"
FONTS_DIR = REPO_ROOT / "data" / "fonts" / "google"

ROLES = ["Regular", "Bold", "Italic", "BoldItalic"]

# Non-alphabetic glyph fonts (emoji/symbol/musical-notation) that survive rendering without
# throwing but produce meaningless font-recognition training data (tofu boxes, unrelated
# glyph sets). Excluded at class-list build time rather than caught by exception handling,
# since they render "successfully" -- see data_gen/broken_fonts.log for the crash-only cases.
EXCLUDED_FAMILIES = {
    "notoemoji",
    "notocoloremoji",
    "notocoloremojicompattest",  # also fails to load (invalid pixel size); listed for clarity
    "notosanssymbols",
    "notosanssymbols2",
    "notoznamennymusicalnotation",
    # non-Latin-script fonts that render blank on English text (no Latin glyph coverage) --
    # confirmed via automated render check against the other 269 non-Latin families in the
    # manifest, which DO render clean Latin glyphs (Noto's shared Latin base design) and are
    # correctly kept. See data_gen/README or session notes for the check methodology.
    "karlatamilinclined",
    "karlatamilupright",
    "phetsarath",
}

VARIATION_NAME_MAP = {
    "Regular": b"Regular",
    "Bold": b"Bold",
    "Italic": b"Italic",
    "BoldItalic": b"Bold Italic",
}

_variation_name_cache = {}


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _get_variation_names(path: Path):
    key = str(path)
    if key in _variation_name_cache:
        return _variation_name_cache[key]
    try:
        font = ImageFont.truetype(str(path), 40)
        names = font.get_variation_names()
    except Exception:
        names = None
    _variation_name_cache[key] = names
    return names


def build_class_list(manifest=None) -> list[tuple[str, str]]:
    """Returns every available (family, role) class, checking variable fonts' actual
    named instances rather than assuming all 4 roles exist."""
    if manifest is None:
        manifest = load_manifest()

    classes = []
    for family, files in manifest.items():
        if family in EXCLUDED_FAMILIES:
            continue

        if "variable_upright" in files:
            names = _get_variation_names(FONTS_DIR / family / "variable_upright.ttf")
            if names:
                if b"Regular" in names:
                    classes.append((family, "Regular"))
                if b"Bold" in names:
                    classes.append((family, "Bold"))
            else:
                classes.append((family, "Regular"))  # fallback: treat as default instance

        if "variable_italic" in files:
            names = _get_variation_names(FONTS_DIR / family / "variable_italic.ttf")
            if names:
                if b"Italic" in names:
                    classes.append((family, "Italic"))
                if b"Bold Italic" in names:
                    classes.append((family, "BoldItalic"))
            else:
                classes.append((family, "Italic"))

        if "variable_upright" not in files and "variable_italic" not in files:
            for role in ROLES:
                if role in files:
                    classes.append((family, role))

    return classes


def get_font(family: str, role: str, size: int, manifest=None) -> ImageFont.FreeTypeFont:
    if manifest is None:
        manifest = load_manifest()
    files = manifest[family]

    if role in ("Regular", "Bold") and "variable_upright" in files:
        path = FONTS_DIR / family / "variable_upright.ttf"
        font = ImageFont.truetype(str(path), size)
        variation = VARIATION_NAME_MAP[role]
        names = font.get_variation_names()
        if variation in names:
            font.set_variation_by_name(variation)
        return font

    if role in ("Italic", "BoldItalic") and "variable_italic" in files:
        path = FONTS_DIR / family / "variable_italic.ttf"
        font = ImageFont.truetype(str(path), size)
        variation = VARIATION_NAME_MAP[role]
        names = font.get_variation_names()
        if variation in names:
            font.set_variation_by_name(variation)
        return font

    if role in files:
        path = FONTS_DIR / family / f"{role}.ttf"
        return ImageFont.truetype(str(path), size)

    raise ValueError(f"no file available for {family}/{role}")


if __name__ == "__main__":
    manifest = load_manifest()
    classes = build_class_list(manifest)
    print(f"total classes: {len(classes)}")
    print(Counter(r for _, r in classes))

    # sanity check: load every role available for a known variable family + a static one
    for fam in ["roboto", "lato"]:
        for role in ROLES:
            if (fam, role) in classes:
                font = get_font(fam, role, 32, manifest)
                print(fam, role, "loaded OK ->", font.getname() if hasattr(font, "getname") else font)
