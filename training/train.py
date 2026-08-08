"""
VFRW Stage C training driver. Hierarchical family/role classifier (see models/classifier.py's
docstring for why) with full crash-safety (checkpoint/resume/pause), drift monitoring, and the
two-part smoke test (Check A: pipeline/VRAM sanity, Check B: architecture-viability on the hard
subset).

Usage:
  python training/train.py --check-a                       # ~30-50 steps, pipeline/VRAM only
  python training/train.py --check-b                       # hard-subset convergence test
  python training/train.py --phase 1                       # full phase 1 run
  python training/train.py --phase 2 --resume               # full phase 2, resuming if checkpointed
  python training/train.py --phase 2 --max-hours 10          # wall-clock cutoff override

Drop a file named training/STOP to pause cleanly at the next check point (checked every epoch
and periodically mid-epoch).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.classifier import (build_model, freeze_backbone, unfreeze_last_blocks,
                                param_groups_for_phase, count_trainable, freeze_bn_stats)
from training.config import CONFIG, PHASE_1, PHASE_2, PhaseConfig
from training.dataset import FontDataset, load_class_list, load_family_role_lists
from training.checkpoint import save_checkpoint, load_checkpoint
from training.monitor import StepLogger, EpochLogger, check_nan_inf
from training.hard_subset import (build_confusable_subset, build_family_complete_random_subset,
                                   SMOKE_TEST_THRESHOLDS)


class LabelSpace:
    """Bundles the family/role index mappings plus the flat class_name <-> (family, role) lookup
    tables that let hierarchical (family_logits, role_logits) predictions be scored against the
    same flat-class top-1/top-5 metrics used throughout the project (and against the
    pre-committed Check B thresholds, which were defined in terms of flat-class top-5)."""

    def __init__(self, manifest_csv: str):
        families, roles = load_family_role_lists(manifest_csv)
        self.family_to_idx = {f: i for i, f in enumerate(families)}
        self.role_to_idx = {r: i for i, r in enumerate(roles)}
        self.num_families = len(families)
        self.num_roles = len(roles)

        class_names = load_class_list(manifest_csv)
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.num_classes = len(class_names)

        # for every flat class index, which family index and role index does it correspond to --
        # used to gather combined (family_logit + role_logit) scores per flat class in one shot
        family_idx_per_class = torch.zeros(self.num_classes, dtype=torch.long)
        role_idx_per_class = torch.zeros(self.num_classes, dtype=torch.long)
        self.family_role_to_flat: dict[tuple[int, int], int] = {}
        for name, flat_idx in self.class_to_idx.items():
            family, role = name.split("__")
            fam_idx, role_idx = self.family_to_idx[family], self.role_to_idx[role]
            family_idx_per_class[flat_idx] = fam_idx
            role_idx_per_class[flat_idx] = role_idx
            self.family_role_to_flat[(fam_idx, role_idx)] = flat_idx
        self.family_idx_per_class = family_idx_per_class
        self.role_idx_per_class = role_idx_per_class

    def combined_logits(self, family_logits: torch.Tensor, role_logits: torch.Tensor) -> torch.Tensor:
        """(batch, num_families) + (batch, num_roles) -> (batch, num_classes) by summing each
        valid (family, role) pair's two scores -- the hierarchical equivalent of the old flat
        head's per-class logit, so topk/top1/top5 work exactly as before."""
        device = family_logits.device
        fam_idx = self.family_idx_per_class.to(device)
        role_idx = self.role_idx_per_class.to(device)
        return family_logits[:, fam_idx] + role_logits[:, role_idx]

    def flat_target(self, family_y: torch.Tensor, role_y: torch.Tensor) -> torch.Tensor:
        flat = torch.empty_like(family_y)
        for i in range(family_y.size(0)):
            flat[i] = self.family_role_to_flat[(family_y[i].item(), role_y[i].item())]
        return flat


def topk_correct(output: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> dict[int, int]:
    maxk = max(ks)
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return {k: correct[:k].reshape(-1).float().sum().item() for k in ks}


def load_probed_batch_size(phase_name: str, configured: int) -> int:
    """The probe verifies headroom, it doesn't dictate the batch size -- config.py's number was
    chosen alongside the reviewed LR schedule, and silently swapping in the raw probed ceiling
    would need a linear-scaling-rule LR re-derivation that hasn't happened. If the configured
    batch doesn't fit within the probed safe ceiling, that's a real problem worth failing loudly
    on rather than silently shrinking."""
    probe_path = REPO_ROOT / "training" / "probed_batch_sizes.json"
    if not probe_path.exists():
        print(f"[warn] no probed_batch_sizes.json found -- using config value ({configured}) "
              f"unverified. Run training/probe_batch_size.py first to confirm it actually fits.")
        return configured
    with open(probe_path) as f:
        data = json.load(f)
    safe_ceiling = data.get(phase_name, {}).get("safe_batch")
    if safe_ceiling is not None and configured > safe_ceiling:
        raise RuntimeError(
            f"[{phase_name}] configured batch_size={configured} exceeds the probed safe "
            f"ceiling of {safe_ceiling} for this GPU -- lower config.py's batch_size or "
            f"re-probe if the GPU/model changed."
        )
    return configured


def build_loaders(class_filter: set[str] | None, batch_size: int, label_space: LabelSpace):
    train_ds = FontDataset(CONFIG.manifest_csv, CONFIG.data_dir, "train",
                            label_space.family_to_idx, label_space.role_to_idx, class_filter)
    val_ds = FontDataset(CONFIG.manifest_csv, CONFIG.data_dir, "val",
                          label_space.family_to_idx, label_space.role_to_idx, class_filter)
    # persistent_workers keeps the worker pool alive across epochs instead of respawning it each
    # time (real cost on Windows, where workers start via spawn not fork); prefetch_factor gives
    # each worker a small queue of batches ready ahead of the GPU asking, so a slow JPEG decode
    # doesn't stall the training step waiting on it.
    workers = CONFIG.num_workers
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=workers, pin_memory=True, drop_last=True,
                               persistent_workers=workers > 0,
                               prefetch_factor=2 if workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True,
                             persistent_workers=workers > 0,
                             prefetch_factor=2 if workers > 0 else None)
    return train_loader, val_loader


def evaluate(model, loader, device, label_space: LabelSpace) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    top1_correct, top5_correct = 0, 0
    family_top1_correct, role_top1_correct = 0, 0
    per_class_correct = torch.zeros(label_space.num_classes)
    per_class_total = torch.zeros(label_space.num_classes)

    with torch.no_grad():
        for x, family_y, role_y in loader:
            x = x.to(device, non_blocking=True)
            family_y = family_y.to(device, non_blocking=True)
            role_y = role_y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # no labels at eval -- plain cosine similarity, no margin applied
                family_logits, role_logits = model(x)
                loss = criterion(family_logits, family_y) + criterion(role_logits, role_y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)

            combined = label_space.combined_logits(family_logits, role_logits)
            flat_y = label_space.flat_target(family_y.cpu(), role_y.cpu()).to(device)
            correct = topk_correct(combined, flat_y, ks=(1, 5))
            top1_correct += correct[1]
            top5_correct += correct[5]
            family_top1_correct += (family_logits.argmax(dim=1) == family_y).sum().item()
            role_top1_correct += (role_logits.argmax(dim=1) == role_y).sum().item()

            pred1 = combined.argmax(dim=1)
            for c in flat_y.unique():
                mask = flat_y == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (pred1[mask] == c).sum().item()

    seen = per_class_total > 0
    macro_acc = (per_class_correct[seen] / per_class_total[seen]).mean().item() if seen.any() else 0.0
    micro_acc = top1_correct / n if n else 0.0

    return {
        "loss": total_loss / n if n else float("nan"),
        "top1": top1_correct / n if n else 0.0,
        "top5": top5_correct / n if n else 0.0,
        "family_top1": family_top1_correct / n if n else 0.0,
        "role_top1": role_top1_correct / n if n else 0.0,
        "macro_acc": macro_acc,
        "micro_acc": micro_acc,
    }


def train_phase(phase_cfg: PhaseConfig, label_space: LabelSpace,
                 class_filter: set[str] | None = None, checkpoint_tag: str | None = None,
                 max_epochs_override: int | None = None, resume: bool = False,
                 max_hours_override: float | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_tag = checkpoint_tag or phase_cfg.name
    max_hours = max_hours_override if max_hours_override is not None else phase_cfg.max_hours
    max_epochs = max_epochs_override if max_epochs_override is not None else phase_cfg.max_epochs

    model = build_model(label_space.num_families, label_space.num_roles,
                         pretrained=CONFIG.pretrained).to(device)
    if phase_cfg.unfreeze_blocks == 0:
        freeze_backbone(model)
    else:
        unfreeze_last_blocks(model, phase_cfg.unfreeze_blocks)
    trainable, total = count_trainable(model)
    print(f"[{phase_cfg.name}] trainable params: {trainable:,} / {total:,}")

    batch_size = load_probed_batch_size(phase_cfg.name, phase_cfg.batch_size)
    train_loader, val_loader = build_loaders(class_filter, batch_size, label_space)
    steps_per_epoch = max(1, len(train_loader) // phase_cfg.grad_accum_steps)

    param_groups = param_groups_for_phase(model, phase_cfg.lr_head, phase_cfg.lr_backbone)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=phase_cfg.weight_decay)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = max(1, int(total_steps * phase_cfg.warmup_pct))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in param_groups], total_steps=total_steps,
        pct_start=phase_cfg.warmup_pct, anneal_strategy="cos",
    ) if total_steps > warmup_steps else None
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 does not need loss scaling, kept as a structural no-op
    criterion = nn.CrossEntropyLoss(label_smoothing=phase_cfg.label_smoothing)

    step_logger = StepLogger(CONFIG.log_dir)
    epoch_logger = EpochLogger(CONFIG.log_dir)

    start_epoch, global_step, best_metric, patience_counter = 0, 0, -1.0, 0
    if resume:
        ckpt = load_checkpoint(CONFIG.checkpoint_dir, tag=checkpoint_tag)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if ckpt["scaler_state"]:
                scaler.load_state_dict(ckpt["scaler_state"])
            if scheduler is not None and ckpt["scheduler_state"]:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt["step"]
            best_metric = ckpt["best_metric"]
            patience_counter = ckpt["patience_counter"]
            print(f"[{phase_cfg.name}] resumed from epoch {ckpt['epoch']}, step {global_step}")

    stop_sentinel = REPO_ROOT / CONFIG.stop_sentinel
    start_time = time.time()
    last_checkpoint_time = start_time

    for epoch in range(start_epoch, max_epochs):
        model.train()
        freeze_bn_stats(model)  # model.train() re-enables BN training mode on frozen layers too -- undo that
        running_loss, running_correct, running_n = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for i, (x, family_y, role_y) in enumerate(train_loader):
            step_start = time.time()
            x = x.to(device, non_blocking=True)
            family_y = family_y.to(device, non_blocking=True)
            role_y = role_y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # ArcFace needs labels to apply the angular margin during training
                family_logits, role_logits = model(x, family_y, role_y)
                loss = (criterion(family_logits, family_y) + criterion(role_logits, role_y)) \
                    / phase_cfg.grad_accum_steps

            if check_nan_inf(loss.item()):
                print(f"[EMERGENCY] NaN/Inf loss at epoch {epoch} step {i} -- checkpointing and stopping")
                save_checkpoint(CONFIG.checkpoint_dir, phase_cfg.name, epoch, global_step,
                                 model, optimizer, scaler, scheduler, best_metric,
                                 patience_counter, tag=f"{checkpoint_tag}_emergency")
                raise RuntimeError("NaN/Inf loss encountered, emergency checkpoint saved")

            scaler.scale(loss).backward()

            if (i + 1) % phase_cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), phase_cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

                elapsed = time.time() - step_start
                samples_per_sec = x.size(0) * phase_cfg.grad_accum_steps / max(elapsed, 1e-6)
                gpu_mem = torch.cuda.memory_allocated(device) / 1e6 if device == "cuda" else 0
                step_logger.log(phase=phase_cfg.name, epoch=epoch, step=global_step,
                                 loss=loss.item() * phase_cfg.grad_accum_steps,
                                 lr=optimizer.param_groups[0]["lr"],
                                 grad_norm_preclip=float(grad_norm), samples_per_sec=samples_per_sec,
                                 gpu_mem_mb=gpu_mem)

            running_loss += loss.item() * phase_cfg.grad_accum_steps * x.size(0)
            # combined-correct: family AND role both right == the flat class right, for a train_acc
            # number comparable to the old flat-head metric
            both_correct = (family_logits.argmax(dim=1) == family_y) & (role_logits.argmax(dim=1) == role_y)
            running_correct += both_correct.sum().item()
            running_n += x.size(0)

            # mid-epoch checkpoint + stop-sentinel check, time-based not step-based
            if time.time() - last_checkpoint_time > CONFIG.checkpoint_every_minutes * 60:
                save_checkpoint(CONFIG.checkpoint_dir, phase_cfg.name, epoch, global_step,
                                 model, optimizer, scaler, scheduler, best_metric,
                                 patience_counter, tag=checkpoint_tag)
                last_checkpoint_time = time.time()
                if stop_sentinel.exists():
                    print("[stop] STOP sentinel found mid-epoch -- checkpointed and exiting")
                    stop_sentinel.unlink()
                    return {"stopped": True, "epoch": epoch}

            if time.time() - start_time > max_hours * 3600:
                print(f"[wall-clock] max_hours={max_hours} reached -- checkpointing and stopping")
                save_checkpoint(CONFIG.checkpoint_dir, phase_cfg.name, epoch, global_step,
                                 model, optimizer, scaler, scheduler, best_metric,
                                 patience_counter, tag=checkpoint_tag)
                return {"stopped": True, "reason": "max_hours", "epoch": epoch}

        train_loss = running_loss / max(running_n, 1)
        train_acc = running_correct / max(running_n, 1)

        val_metrics = evaluate(model, val_loader, device, label_space)
        drift = epoch_logger.log(
            drift_gap_epochs=CONFIG.drift_gap_epochs, phase=phase_cfg.name, epoch=epoch,
            train_loss=train_loss, train_acc=train_acc, val_loss=val_metrics["loss"],
            val_top1=val_metrics["top1"], val_top5=val_metrics["top5"],
            macro_acc=val_metrics["macro_acc"], micro_acc=val_metrics["micro_acc"],
        )
        print(f"[{phase_cfg.name}] epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.3f} "
              f"val_top5={val_metrics['top5']:.3f} family_top1={val_metrics['family_top1']:.3f} "
              f"role_top1={val_metrics['role_top1']:.3f}" + (" [DRIFT]" if drift else ""))

        save_checkpoint(CONFIG.checkpoint_dir, phase_cfg.name, epoch, global_step, model,
                         optimizer, scaler, scheduler, best_metric, patience_counter,
                         tag=checkpoint_tag)
        last_checkpoint_time = time.time()

        if val_metrics["top5"] > best_metric:
            best_metric = val_metrics["top5"]
            patience_counter = 0
            save_checkpoint(CONFIG.checkpoint_dir, phase_cfg.name, epoch, global_step, model,
                             optimizer, scaler, scheduler, best_metric, patience_counter,
                             tag=f"{checkpoint_tag}_best")
        else:
            patience_counter += 1
            if patience_counter >= phase_cfg.early_stop_patience:
                print(f"[{phase_cfg.name}] early stopping at epoch {epoch} "
                      f"(no val_top5 improvement for {phase_cfg.early_stop_patience} epochs)")
                break

        if stop_sentinel.exists():
            print("[stop] STOP sentinel found -- exiting after epoch checkpoint")
            stop_sentinel.unlink()
            return {"stopped": True, "epoch": epoch}

    return {"stopped": False, "best_val_top5": best_metric, "final_epoch": epoch}


def run_check_a(label_space: LabelSpace, n_steps: int = 600):
    """Check A: pipeline/VRAM sanity only -- does loss trend down, does the batch size hold.
    n_steps=600 and a first-50-vs-last-50 average comparison, not first-vs-last single step:
    with thousands of families and a fresh linear head, individual steps are noisy and loss can
    even rise briefly before descending as AdamW's moment estimates warm up. A 40-step
    single-point comparison flagged this as a false FAIL during development; confirmed via a
    longer run and a single-batch overfit test that the training mechanics were correct and it
    was just too short a window to see through the noise."""
    print("=== Check A: pipeline + VRAM sanity ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(label_space.num_families, label_space.num_roles,
                         pretrained=CONFIG.pretrained).to(device)
    freeze_backbone(model)
    freeze_bn_stats(model)
    batch_size = load_probed_batch_size(PHASE_1.name, PHASE_1.batch_size)
    train_loader, _ = build_loaders(None, batch_size, label_space)

    optimizer = torch.optim.AdamW(param_groups_for_phase(model, PHASE_1.lr_head, None))
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 does not need loss scaling, kept as a structural no-op
    criterion = nn.CrossEntropyLoss()

    losses = []
    for i, (x, family_y, role_y) in enumerate(train_loader):
        if i >= n_steps:
            break
        x, family_y, role_y = x.to(device), family_y.to(device), role_y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            family_logits, role_logits = model(x, family_y, role_y)
            loss = criterion(family_logits, family_y) + criterion(role_logits, role_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if device == "cuda" and i % 10 == 0:
            print(f"  step {i}: loss={loss.item():.3f} gpu_mem={torch.cuda.memory_allocated()/1e6:.0f}MB")

    window = min(50, len(losses) // 4) or 1
    first_avg = sum(losses[:window]) / window
    last_avg = sum(losses[-window:]) / window
    decreasing = last_avg < first_avg
    print(f"Check A result: avg loss (first {window} steps) {first_avg:.3f} -> "
          f"avg loss (last {window} steps) {last_avg:.3f} "
          f"({'PASS' if decreasing else 'FAIL'} -- trend should be down over {n_steps} steps)")
    return decreasing


def run_check_b(manifest_csv: str, label_space: LabelSpace):
    """Check B: architecture viability, on two subsets in parallel, to a pre-committed
    convergence threshold (see hard_subset.SMOKE_TEST_THRESHOLDS). Thresholds were defined
    against flat-class top-5, which is exactly what LabelSpace.combined_logits reproduces from
    the two hierarchical heads, so the same numbers are directly comparable to the flat-head
    Check B runs."""
    print("=== Check B: architecture viability (hard-subset convergence) ===")
    thresholds = SMOKE_TEST_THRESHOLDS

    confusable = build_confusable_subset(manifest_csv)
    # family-complete, not a naive per-class random sample -- see build_family_complete_random_subset's
    # docstring. A per-class random sample left most families with zero training coverage, which is
    # a real handicap specific to the hierarchical family/role head (its family ArcFace weight
    # vectors for uncovered families never get a gradient and can noisily outrank trained ones),
    # not a fair test of whether the architecture scales.
    random500 = build_family_complete_random_subset(manifest_csv, size=500)
    print(f"confusable subset: {len(confusable)} classes, random subset: {len(random500)} classes "
          f"(family-complete)")

    results = {}
    for name, subset in [("confusable", confusable), ("random500", random500)]:
        print(f"\n--- training on {name} subset ---")
        result = train_phase(PHASE_1, label_space, class_filter=subset,
                              checkpoint_tag=f"checkb_{name}_p1",
                              max_epochs_override=min(5, thresholds["max_epochs_cap"]))
        result2 = train_phase(PHASE_2, label_space, class_filter=subset,
                               checkpoint_tag=f"checkb_{name}_p2",
                               max_epochs_override=thresholds["max_epochs_cap"])
        results[name] = result2.get("best_val_top5", 0.0)
        print(f"{name}: best val top5 = {results[name]:.3f}")

    confusable_pass = results.get("confusable", 0.0) >= thresholds["confusable_top5_min"]
    random_pass = results.get("random500", 0.0) >= thresholds["random500_top5_min"]

    print(f"\nCheck B verdict:")
    print(f"  confusable subset top5={results.get('confusable', 0):.3f} "
          f"(threshold {thresholds['confusable_top5_min']}) -> {'PASS' if confusable_pass else 'FAIL'}")
    print(f"  random500 subset top5={results.get('random500', 0):.3f} "
          f"(threshold {thresholds['random500_top5_min']}) -> {'PASS' if random_pass else 'FAIL'}")

    if confusable_pass and random_pass:
        print("  BOTH PASS -- hierarchical head viable, proceed with full 3,407-class run.")
    elif random_pass and not confusable_pass:
        print("  random500 passes but confusable pairs fail -- scales broadly but still can't "
              "separate near-duplicate fonts even with the family/role split.")
    elif confusable_pass and not random_pass:
        print("  confusable pairs pass but random500 fails -- unexpected; investigate before "
              "proceeding, this combination suggests a bug rather than a capacity limit.")
    else:
        print("  BOTH FAIL -- reconsider architecture before committing to the full run.")

    return confusable_pass and random_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-a", action="store_true")
    parser.add_argument("--check-b", action="store_true")
    parser.add_argument("--phase", type=int, choices=[1, 2])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-hours", type=float, default=None)
    args = parser.parse_args()

    label_space = LabelSpace(CONFIG.manifest_csv)
    print(f"loaded {label_space.num_classes} classes ({label_space.num_families} families x "
          f"up to {label_space.num_roles} roles) from manifest")

    if args.check_a:
        run_check_a(label_space)
    elif args.check_b:
        run_check_b(CONFIG.manifest_csv, label_space)
    elif args.phase == 1:
        train_phase(PHASE_1, label_space, resume=args.resume, max_hours_override=args.max_hours)
    elif args.phase == 2:
        train_phase(PHASE_2, label_space, resume=args.resume, max_hours_override=args.max_hours)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
