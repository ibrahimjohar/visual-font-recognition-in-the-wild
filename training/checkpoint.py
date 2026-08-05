"""
Crash-safe checkpointing: same atomic-write philosophy as data_gen/generate_dataset.py --
write to a temp file, then rename over the target, so a kill mid-write never leaves a
half-written checkpoint that --resume would load and crash on.
"""

import json
import os
import time
from pathlib import Path

import torch


def save_checkpoint(checkpoint_dir: str, phase: str, epoch: int, step: int,
                     model, optimizer, scaler, scheduler, best_metric: float,
                     patience_counter: int, extra: dict | None = None,
                     tag: str = "last") -> Path:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "phase": phase,
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "patience_counter": patience_counter,
        "saved_at": time.time(),
        "extra": extra or {},
    }

    target = ckpt_dir / f"{tag}.pt"
    tmp = ckpt_dir / f".{tag}.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, target)  # atomic on both windows and posix

    # keep a small metadata file for quick inspection without loading the full checkpoint
    meta = {"phase": phase, "epoch": epoch, "step": step, "best_metric": best_metric,
             "saved_at": state["saved_at"]}
    meta_path = ckpt_dir / f"{tag}.meta.json"
    meta_tmp = ckpt_dir / f".{tag}.meta.json.tmp"
    with open(meta_tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_tmp, meta_path)

    return target


def load_checkpoint(checkpoint_dir: str, tag: str = "last") -> dict | None:
    path = Path(checkpoint_dir) / f"{tag}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def latest_checkpoint_exists(checkpoint_dir: str, tag: str = "last") -> bool:
    return (Path(checkpoint_dir) / f"{tag}.pt").exists()
