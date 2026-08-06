"""
Regenerates every manifest row whose effective glyph height was measured below the 20px
legibility floor (see the render.py fix and the eff_height measurement pass in session notes),
using the fixed height-scaled letterbox logic. Same text/family/role/class_name/split as before
-- only the rendered image content changes, so the manifest itself needs no edits.

Resumable: writes a progress log of completed paths, skips them on restart, same crash-safety
philosophy as generate_dataset.py.
"""

import csv
import pickle
import time
from pathlib import Path

from font_loader import load_manifest
from render import render_crop

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_CSV = REPO_ROOT / "data" / "synthetic" / "manifest.csv"
DATA_DIR = REPO_ROOT / "data" / "synthetic"
EFF_HEIGHT_PKL = Path(
    "C:/Users/ibrah/AppData/Local/Temp/claude/"
    "C--Ibrahim-Personal-University-Stuff-Portfolio-Projects-font-identifier/"
    "922e54b9-b003-4051-8623-0b8b44a90b77/scratchpad/eff_height_results.pkl"
)
PROGRESS_LOG = Path(__file__).resolve().parent / "regenerate_progress.log"
HEIGHT_THRESHOLD = 20.0


def main():
    with open(EFF_HEIGHT_PKL, "rb") as f:
        eff_results = pickle.load(f)  # list of (class_name, eff_height, textlen), manifest row order

    manifest_rows = []
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            manifest_rows.append(row)

    assert len(manifest_rows) == len(eff_results), (
        f"manifest has {len(manifest_rows)} rows but eff_height results has {len(eff_results)} "
        f"-- manifest changed since the measurement pass, re-run the measurement first"
    )

    to_regen = [row for row, (_, h, _) in zip(manifest_rows, eff_results) if h < HEIGHT_THRESHOLD]
    print(f"{len(to_regen)} of {len(manifest_rows)} rows need regeneration "
          f"(effective height < {HEIGHT_THRESHOLD}px)")

    done = set()
    if PROGRESS_LOG.exists():
        done = set(PROGRESS_LOG.read_text(encoding="utf-8").splitlines())
        print(f"resuming: {len(done)} already regenerated")

    font_manifest = load_manifest()
    progress_file = open(PROGRESS_LOG, "a", encoding="utf-8")

    t0 = time.time()
    n_done = 0
    try:
        for row in to_regen:
            path = row["path"]
            if path in done:
                continue

            img, _ = render_crop(row["family"], row["role"], font_manifest, text=row["text"])
            img.save(DATA_DIR / path, quality=90)

            progress_file.write(path + "\n")
            progress_file.flush()
            n_done += 1
            if n_done % 5000 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(to_regen) - len(done) - n_done) / rate if rate > 0 else float("inf")
                print(f"{n_done} regenerated this run, {elapsed:.0f}s elapsed, "
                      f"~{remaining/60:.1f}min remaining")
    finally:
        progress_file.close()

    print(f"done: {n_done} images regenerated this run")


if __name__ == "__main__":
    main()
