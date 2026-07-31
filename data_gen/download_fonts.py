"""
Download the font files listed in data_gen/font_manifest.json into data/fonts/google/.

Layout: data/fonts/google/<family>/<role>.ttf
  where role is one of: Regular, Bold, Italic, BoldItalic, variable_upright, variable_italic

Safety/resumability:
  - Each file streams to a .part temp path and is only renamed to its final name after a
    complete, verified download — so a truncated/interrupted download never looks "done".
  - Before downloading, any file that already exists at its final path is skipped, so
    re-running this script after an interruption just picks up where it left off.
  - Transient connection errors (we've seen ChunkedEncodingError/timeouts on this network)
    are retried a few times with backoff before giving up on a single file; a failure on one
    file doesn't stop the run — it's logged and skipped, reported in the summary at the end.
"""

import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data_gen" / "font_manifest.json"
OUT_DIR = REPO_ROOT / "data" / "fonts" / "google"
RAW_BASE = "https://raw.githubusercontent.com/google/fonts/main/"

HEADERS = {"Accept-Encoding": "identity", "User-Agent": "vfrw-fontfetch"}
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 3


def download_one(repo_path: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True  # already downloaded, skip

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    url = RAW_BASE + repo_path

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, headers=HEADERS, timeout=60, stream=True) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            if tmp_path.stat().st_size == 0:
                raise IOError("downloaded file is empty")
            tmp_path.rename(dest)
            return True
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == MAX_RETRIES:
                print(f"FAILED after {MAX_RETRIES} attempts: {repo_path} ({e})")
                return False
            time.sleep(RETRY_BACKOFF_SEC * attempt)

    return False


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    jobs = []
    for family, files in manifest.items():
        for role, repo_path in files.items():
            ext = ".ttf"
            dest = OUT_DIR / family / f"{role}{ext}"
            jobs.append((repo_path, dest))

    print(f"{len(jobs)} files to check/download -> {OUT_DIR}")

    already = sum(1 for _, dest in jobs if dest.exists() and dest.stat().st_size > 0)
    print(f"{already} already present, {len(jobs) - already} remaining")

    failed = []
    downloaded = 0
    for repo_path, dest in tqdm(jobs, desc="downloading fonts"):
        was_present = dest.exists() and dest.stat().st_size > 0
        ok = download_one(repo_path, dest)
        if ok and not was_present:
            downloaded += 1
        if not ok:
            failed.append(repo_path)

    print(f"\nnewly downloaded this run: {downloaded}")
    print(f"failed: {len(failed)}")
    if failed:
        fail_log = REPO_ROOT / "data_gen" / "download_failures.txt"
        with open(fail_log, "w") as f:
            f.write("\n".join(failed))
        print(f"failed paths written to {fail_log} -- re-run this script to retry them")


if __name__ == "__main__":
    main()
