"""
Runs a single forward+backward AMP step at a given batch size, in an isolated process, and
exits 0 (fits) or 1 (OOM). Invoked as a subprocess by probe_batch_size.py so a hard CUDA OOM
-- which can leave the parent process's CUDA context unusable for further calls, including
torch.cuda.empty_cache() itself -- only kills this throwaway process, not the whole probe.

Usage: python training/_probe_one_batch.py <phase_name> <batch_size>
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from models.classifier import build_model, freeze_backbone, unfreeze_last_blocks, freeze_bn_stats
from training.config import CONFIG, PHASES


def main():
    phase_name, batch_size = sys.argv[1], int(sys.argv[2])
    phase = next(p for p in PHASES if p.name == phase_name)

    device = "cuda"
    model = build_model(CONFIG.num_classes, pretrained=False).to(device)
    if phase.unfreeze_blocks == 0:
        freeze_backbone(model)
    else:
        unfreeze_last_blocks(model, phase.unfreeze_blocks)
    freeze_bn_stats(model)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 does not need loss scaling, kept as a structural no-op
    criterion = nn.CrossEntropyLoss()

    try:
        x = torch.randn(batch_size, 3, CONFIG.input_size, CONFIG.input_size, device=device)
        y = torch.randint(0, CONFIG.num_classes, (batch_size,), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            sys.exit(1)
        raise

    sys.exit(0)


if __name__ == "__main__":
    main()
