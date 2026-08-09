"""
Training configuration for VFRW Stage C classifier (EfficientNet-B0, 2-phase progressive
unfreezing). All numbers here are the finalized/council-reviewed defaults; batch sizes get
overwritten by the empirical VRAM probe (see probe_batch_size.py) before a real run starts.
"""

from dataclasses import dataclass, field


@dataclass
class PhaseConfig:
    name: str
    unfreeze_blocks: int  # 0 = backbone fully frozen, 2 = last 2 MBConv blocks trainable
    lr_head: float
    lr_backbone: float | None  # None when backbone is frozen
    weight_decay: float
    batch_size: int  # placeholder -- replaced by the empirical probe
    grad_accum_steps: int
    warmup_pct: float = 0.05
    label_smoothing: float = 0.1
    grad_clip_norm: float = 5.0
    max_epochs: int = 30
    early_stop_patience: int = 3
    max_hours: float = 12.0


PHASE_1 = PhaseConfig(
    name="phase1_head_warmup",
    unfreeze_blocks=0,
    lr_head=1e-3,
    lr_backbone=None,
    weight_decay=0.03,
    batch_size=128,
    grad_accum_steps=1,
    max_epochs=5,
    early_stop_patience=3,
    max_hours=6.0,
)

PHASE_2 = PhaseConfig(
    name="phase2_finetune",
    unfreeze_blocks=2,
    lr_head=1e-4,
    lr_backbone=1e-5,
    weight_decay=0.05,
    # VRAM probe confirmed both phases fit a batch far beyond what's needed (~1550 before OOM,
    # dominated by early-layer forward activations, not by the 2 unfrozen blocks' backward
    # cost) -- real headroom means phase 2 doesn't need grad accumulation at all, so this runs
    # at physical batch 256 directly rather than 24-32 x 5 accumulation steps. Kept at 256, not
    # the full probed ceiling, since the reviewed LR schedule was tuned around this effective
    # batch and pushing further would need a linear-scaling-rule re-derivation that hasn't
    # happened.
    batch_size=256,
    grad_accum_steps=1,
    max_epochs=40,
    early_stop_patience=5,
    max_hours=30.0,
)


@dataclass
class GlobalConfig:
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    input_size: int = 224
    num_classes: int = 3407  # data/synthetic, post emoji/symbol/blank-script cleanup
    data_dir: str = "data/synthetic"
    manifest_csv: str = "data/synthetic/manifest.csv"
    real_test_dir: str = "data/real_test"
    checkpoint_dir: str = "training/checkpoints"
    log_dir: str = "training/logs"
    checkpoint_every_epoch: bool = True
    checkpoint_every_minutes: int = 15
    stop_sentinel: str = "training/STOP"
    seed: int = 42
    drift_gap_epochs: int = 4  # consecutive epochs of widening train/val loss gap -> flag
    # was 0 ("windows multiprocessing dataloader crashes") but that was never actually
    # re-verified against this project's dataset/transform code. num_workers=0 was almost
    # certainly the real cause of earlier phase-2 stalls: single-threaded synchronous JPEG
    # decode/transform left the GPU idle between batches, worse under memory pressure. Raising
    # this to 4 hit a real MemoryError on this machine (only ~1.7GB free system RAM at the
    # time) -- Windows spawns each worker as a fresh process that re-imports the whole script,
    # and that used to include timm's heavy dependency chain before the lazy-import fix in
    # models/classifier.py. 2 is the conservative number given how tight memory is here;
    # revisit upward only after confirming more headroom (Task Manager, not a guess).
    num_workers: int = 2


CONFIG = GlobalConfig()
PHASES = [PHASE_1, PHASE_2]
