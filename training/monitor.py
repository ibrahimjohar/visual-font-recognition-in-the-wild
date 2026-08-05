"""
Per-step and per-epoch CSV logging, plus the drift/anomaly checks the plan calls for:
train/val loss-gap widening (overfitting drift), NaN/Inf guard, and macro-vs-micro accuracy
gap tracking. CSV rather than TensorBoard to match the project's lightweight dependency
footprint (pandas/pyyaml are already in requirements.txt; TensorBoard isn't).
"""

import csv
import math
from pathlib import Path


class StepLogger:
    FIELDS = ["phase", "epoch", "step", "loss", "lr", "grad_norm_preclip", "samples_per_sec", "gpu_mem_mb"]

    def __init__(self, log_dir: str):
        self.path = Path(log_dir) / "steps.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, **kwargs):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(kwargs)


class EpochLogger:
    FIELDS = ["phase", "epoch", "train_loss", "train_acc", "val_loss", "val_top1", "val_top5",
              "hard_subset_acc", "macro_acc", "micro_acc", "macro_micro_gap",
              "train_val_loss_gap", "drift_flag", "real_photo_top1", "real_photo_top5"]

    def __init__(self, log_dir: str):
        self.path = Path(log_dir) / "epochs.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()
        self._loss_gap_history: list[float] = []

    def _ensure_header(self):
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, drift_gap_epochs: int = 4, **kwargs):
        train_loss = kwargs.get("train_loss")
        val_loss = kwargs.get("val_loss")
        gap = None
        drift = False
        if train_loss is not None and val_loss is not None:
            gap = val_loss - train_loss
            self._loss_gap_history.append(gap)
            recent = self._loss_gap_history[-drift_gap_epochs:]
            if len(recent) == drift_gap_epochs and all(
                recent[i] < recent[i + 1] for i in range(len(recent) - 1)
            ):
                drift = True

        macro = kwargs.get("macro_acc")
        micro = kwargs.get("micro_acc")
        macro_micro_gap = (micro - macro) if (macro is not None and micro is not None) else None

        row = {field: kwargs.get(field) for field in self.FIELDS}
        row["train_val_loss_gap"] = gap
        row["drift_flag"] = drift
        row["macro_micro_gap"] = macro_micro_gap

        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)

        if drift:
            print(f"[drift warning] train/val loss gap has widened for "
                  f"{drift_gap_epochs} consecutive epochs -- possible overfitting")
        return drift


def check_nan_inf(loss_value: float) -> bool:
    """Returns True if the loss is NaN or Inf -- caller should emergency-checkpoint and stop."""
    return math.isnan(loss_value) or math.isinf(loss_value)
