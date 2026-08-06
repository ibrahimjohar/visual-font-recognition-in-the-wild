"""
Empirical VRAM probe -- per the council's Executor critique, batch size shouldn't be
discovered by babysitting a training run into OOM. Binary-searches the largest batch size
that survives a real forward+backward step (with AMP) for each phase, on the actual model
and actual GPU, then writes the results to a JSON file that train.py reads instead of
trusting config.py's placeholder numbers.

Each candidate batch size is tested in its own subprocess (_probe_one_batch.py), not inline
in this process -- a hard CUDA OOM can leave a process's CUDA context unusable for further
calls (including torch.cuda.empty_cache() itself), so isolating each attempt is what actually
makes the binary search reliable rather than crashing on the first OOM it hits.

Usage: python training/probe_batch_size.py
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.config import CONFIG, PHASES

PYTHON = sys.executable
PROBE_SCRIPT = REPO_ROOT / "training" / "_probe_one_batch.py"


def fits(phase_name: str, batch_size: int) -> bool:
    result = subprocess.run(
        [PYTHON, str(PROBE_SCRIPT), phase_name, str(batch_size)],
        capture_output=True, timeout=120,
    )
    return result.returncode == 0


def probe_max_batch(phase_name: str, low: int = 2, high: int = 512) -> int:
    while fits(phase_name, high):
        print(f"  batch {high}: fits, doubling")
        low = high
        high *= 2
        if high > 4096:
            break
    print(f"  batch {high}: does not fit, narrowing between {low} and {high}")

    while high - low > 4:
        mid = (low + high) // 2
        if fits(phase_name, mid):
            print(f"  batch {mid}: fits")
            low = mid
        else:
            print(f"  batch {mid}: does not fit")
            high = mid

    return low


def main():
    if not torch.cuda.is_available():
        print("no CUDA device available -- cannot probe VRAM. Aborting.")
        sys.exit(1)

    print(f"probing on {torch.cuda.get_device_name(0)}")
    results = {}

    for phase in PHASES:
        print(f"\n{phase.name}: probing max batch size...")
        max_batch = probe_max_batch(phase.name)
        # leave headroom -- don't run training at the exact OOM boundary, dataloader/pinned
        # memory/other processes need slack. Use 80% of the discovered max.
        safe_batch = max(1, int(max_batch * 0.8))
        print(f"{phase.name}: max viable batch ~{max_batch}, using safe batch {safe_batch}")
        results[phase.name] = {"max_batch_probed": max_batch, "safe_batch": safe_batch}

    out_path = REPO_ROOT / "training" / "probed_batch_sizes.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
