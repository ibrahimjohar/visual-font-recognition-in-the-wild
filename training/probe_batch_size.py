"""
Empirical VRAM probe -- per the council's Executor critique, batch size shouldn't be
discovered by babysitting a training run into OOM. Binary-searches the largest batch size
that survives a real forward+backward step (with AMP) for each phase, on the actual model
and actual GPU, then writes the results to a JSON file that train.py reads instead of
trusting config.py's placeholder numbers.

Usage: python training/probe_batch_size.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models"))

from models.classifier import build_model, freeze_backbone, unfreeze_last_blocks
from training.config import CONFIG, PHASES


def probe_max_batch(model: nn.Module, device: str, input_size: int, low: int = 2, high: int = 512) -> int:
    """Binary search: largest batch size where a forward+backward AMP step doesn't OOM."""
    model = model.to(device)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()

    def fits(batch_size: int) -> bool:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(batch_size, 3, input_size, input_size, device=device)
            y = torch.randint(0, CONFIG.num_classes, (batch_size,), device=device)
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            model.zero_grad(set_to_none=True)
            del x, y, out, loss
            torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False

    # find an upper bound that fails first
    while fits(high):
        low = high
        high *= 2
        if high > 4096:
            break

    while high - low > 4:
        mid = (low + high) // 2
        if fits(mid):
            low = mid
        else:
            high = mid

    return low


def main():
    if not torch.cuda.is_available():
        print("no CUDA device available -- cannot probe VRAM. Aborting.")
        sys.exit(1)

    device = "cuda"
    print(f"probing on {torch.cuda.get_device_name(0)}")
    results = {}

    for phase in PHASES:
        model = build_model(CONFIG.num_classes, pretrained=False)  # weights don't matter for a memory probe
        if phase.unfreeze_blocks == 0:
            freeze_backbone(model)
        else:
            unfreeze_last_blocks(model, phase.unfreeze_blocks)

        print(f"\n{phase.name}: probing max batch size...")
        max_batch = probe_max_batch(model, device, CONFIG.input_size)
        # leave headroom -- don't run training at the exact OOM boundary, dataloader/pinned
        # memory/other processes need slack. Use 80% of the discovered max.
        safe_batch = max(1, int(max_batch * 0.8))
        print(f"{phase.name}: max viable batch ~{max_batch}, using safe batch {safe_batch}")
        results[phase.name] = {"max_batch_probed": max_batch, "safe_batch": safe_batch}

        del model
        torch.cuda.empty_cache()

    out_path = REPO_ROOT / "training" / "probed_batch_sizes.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
