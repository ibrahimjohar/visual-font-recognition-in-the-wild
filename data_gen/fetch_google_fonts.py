"""
Download Google Fonts (ofl/apache/ufl) font files into data/fonts/google/.

Selection rule per family:
  - static_named families: download whichever of Regular/Bold/Italic/BoldItalic exist as
    literal files (e.g. Lato-Regular.ttf, Lato-Bold.ttf).
  - variable-only families (single file with a weight axis, e.g. Roboto[wdth,wght].ttf):
    download the upright variable file and the italic variable file (if present). Regular/Bold
    instances are extracted at render time via Pillow's set_variation_by_axes, not at download
    time, since a single file already covers both weights.
  - mixed families (have both): prefer the static named files, ignore the variable file.

Source of file paths: data_gen/google_fonts_ttf_paths.txt (cached `git ls-tree` output, see
Phase 1 planning notes) — avoids re-hitting GitHub's rate-limited REST API.
"""

import re
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PATHS_FILE = REPO_ROOT / "data_gen" / "google_fonts_ttf_paths.txt"
RAW_BASE = "https://raw.githubusercontent.com/google/fonts/main/"

STATIC_STYLES = ["Regular", "Bold", "Italic", "BoldItalic"]


def load_family_files() -> dict[str, list[str]]:
    fam_files = defaultdict(list)
    with open(PATHS_FILE) as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            parts = path.split("/")
            fam = parts[1]
            fam_files[fam].append(path)
    return fam_files


def classify_family(files: list[str]) -> str:
    has_var = any("[" in f for f in files)
    has_static = any(re.search(r"-(Regular|Bold|Italic|BoldItalic)\.ttf$", f) for f in files)
    if has_var and not has_static:
        return "variable_only"
    if has_static and not has_var:
        return "static_named"
    return "mixed"


def select_files_for_family(family: str, files: list[str]) -> dict[str, str]:
    """Returns {style_or_role: repo_path} for the files we want to download."""
    kind = classify_family(files)
    selected = {}

    if kind in ("static_named", "mixed"):
        for style in STATIC_STYLES:
            for f in files:
                if f.endswith(f"-{style}.ttf") and "[" not in f:
                    selected[style] = f
                    break
    else:  # variable_only
        upright = next((f for f in files if "[" in f and "-Italic[" not in f), None)
        italic = next((f for f in files if "-Italic[" in f), None)
        if upright:
            selected["variable_upright"] = upright
        if italic:
            selected["variable_italic"] = italic

    return selected


def build_manifest() -> dict[str, dict[str, str]]:
    fam_files = load_family_files()
    manifest = {}
    for fam, files in fam_files.items():
        selected = select_files_for_family(fam, files)
        if selected:
            manifest[fam] = selected
    return manifest


if __name__ == "__main__":
    manifest = build_manifest()

    kind_counts = defaultdict(int)
    file_count = 0
    for fam, sel in manifest.items():
        file_count += len(sel)
        if "variable_upright" in sel or "variable_italic" in sel:
            kind_counts["variable_only"] += 1
        else:
            kind_counts["static_named_or_mixed"] += 1

    print(f"families selected: {len(manifest)}")
    print(f"total files to download: {file_count}")
    print(dict(kind_counts))

    out_path = REPO_ROOT / "data_gen" / "font_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out_path}")
