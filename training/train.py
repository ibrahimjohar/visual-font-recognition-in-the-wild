"""
VFRW Stage C training driver. Implements the 2-phase progressive-unfreezing plan with full
crash-safety (checkpoint/resume/pause), drift monitoring, and the two-part smoke test
(Check A: pipeline/VRAM sanity, Check B: architecture-viability on the hard subset).

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
from training.dataset import FontDataset, load_class_list
from training.checkpoint import save_checkpoint, load_checkpoint
from training.monitor import StepLogger, EpochLogger, check_nan_inf
from training.hard_subset import build_confusable_subset, build_random_subset, SMOKE_TEST_THRESHOLDS


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


def build_loaders(class_filter: set[str] | None, batch_size: int, class_to_idx: dict[str, int]):
    train_ds = FontDataset(CONFIG.manifest_csv, CONFIG.data_dir, "train", class_to_idx, class_filter)
    val_ds = FontDataset(CONFIG.manifest_csv, CONFIG.data_dir, "val", class_to_idx, class_filter)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=CONFIG.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=CONFIG.num_workers, pin_memory=True)
    return train_loader, val_loader


def evaluate(model, loader, device, num_classes: int) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    top1_correct, top5_correct = 0, 0
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            correct = topk_correct(out, y, ks=(1, 5))
            top1_correct += correct[1]
            top5_correct += correct[5]

            pred1 = out.argmax(dim=1)
            for c in y.unique():
                mask = y == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (pred1[mask] == c).sum().item()

    seen = per_class_total > 0
    macro_acc = (per_class_correct[seen] / per_class_total[seen]).mean().item() if seen.any() else 0.0
    micro_acc = top1_correct / n if n else 0.0

    return {
        "loss": total_loss / n if n else float("nan"),
        "top1": top1_correct / n if n else 0.0,
        "top5": top5_correct / n if n else 0.0,
        "macro_acc": macro_acc,
        "micro_acc": micro_acc,
    }


def train_phase(phase_cfg: PhaseConfig, class_to_idx: dict[str, int], num_classes: int,
                 class_filter: set[str] | None = None, checkpoint_tag: str | None = None,
                 max_epochs_override: int | None = None, resume: bool = False,
                 max_hours_override: float | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_tag = checkpoint_tag or phase_cfg.name
    max_hours = max_hours_override if max_hours_override is not None else phase_cfg.max_hours
    max_epochs = max_epochs_override if max_epochs_override is not None else phase_cfg.max_epochs

    model = build_model(num_classes, pretrained=CONFIG.pretrained).to(device)
    if phase_cfg.unfreeze_blocks == 0:
        freeze_backbone(model)
    else:
        unfreeze_last_blocks(model, phase_cfg.unfreeze_blocks)
    trainable, total = count_trainable(model)
    print(f"[{phase_cfg.name}] trainable params: {trainable:,} / {total:,}")

    batch_size = load_probed_batch_size(phase_cfg.name, phase_cfg.batch_size)
    train_loader, val_loader = build_loaders(class_filter, batch_size, class_to_idx)
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
        epoch_step_start = time.time()

        for i, (x, y) in enumerate(train_loader):
            step_start = time.time()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = criterion(out, y) / phase_cfg.grad_accum_steps

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
            running_correct += (out.argmax(dim=1) == y).sum().item()
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

        val_metrics = evaluate(model, val_loader, device, num_classes)
        drift = epoch_logger.log(
            drift_gap_epochs=CONFIG.drift_gap_epochs, phase=phase_cfg.name, epoch=epoch,
            train_loss=train_loss, train_acc=train_acc, val_loss=val_metrics["loss"],
            val_top1=val_metrics["top1"], val_top5=val_metrics["top5"],
            macro_acc=val_metrics["macro_acc"], micro_acc=val_metrics["micro_acc"],
        )
        print(f"[{phase_cfg.name}] epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.3f} "
              f"val_top5={val_metrics['top5']:.3f}" + (" [DRIFT]" if drift else ""))

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


def run_check_a(class_to_idx: dict, num_classes: int, n_steps: int = 600):
    """Check A: pipeline/VRAM sanity only -- does loss trend down, does the batch size hold.
    n_steps=600 and a first-50-vs-last-50 average comparison, not first-vs-last single step:
    with 3,407 classes and a fresh linear head, individual steps are noisy (each batch of ~128
    barely samples the class space) and loss can even rise briefly before descending as AdamW's
    moment estimates warm up. A 40-step single-point comparison flagged this as a false FAIL
    during development; confirmed via a longer run and a single-batch overfit test that the
    training mechanics were correct and it was just too short a window to see through the noise."""
    print("=== Check A: pipeline + VRAM sanity ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes, pretrained=CONFIG.pretrained).to(device)
    freeze_backbone(model)
    freeze_bn_stats(model)
    batch_size = load_probed_batch_size(PHASE_1.name, PHASE_1.batch_size)
    train_loader, _ = build_loaders(None, batch_size, class_to_idx)

    optimizer = torch.optim.AdamW(param_groups_for_phase(model, PHASE_1.lr_head, None))
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 does not need loss scaling, kept as a structural no-op
    criterion = nn.CrossEntropyLoss()

    losses = []
    for i, (x, y) in enumerate(train_loader):
        if i >= n_steps:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = criterion(out, y)
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


def run_check_b(manifest_csv: str, class_to_idx: dict):
    """Check B: architecture viability, on two subsets in parallel, to a pre-committed
    convergence threshold (see hard_subset.SMOKE_TEST_THRESHOLDS)."""
    print("=== Check B: architecture viability (hard-subset convergence) ===")
    thresholds = SMOKE_TEST_THRESHOLDS

    confusable = build_confusable_subset(manifest_csv)
    random500 = build_random_subset(manifest_csv, size=500)
    print(f"confusable subset: {len(confusable)} classes, random subset: {len(random500)} classes")

    results = {}
    for name, subset in [("confusable", confusable), ("random500", random500)]:
        print(f"\n--- training on {name} subset ---")
        n_classes_subset = len(subset)
        # short phase1 + phase2 run, capped by max_epochs_cap, early-stopped on convergence
        result = train_phase(PHASE_1, class_to_idx, CONFIG.num_classes, class_filter=subset,
                              checkpoint_tag=f"checkb_{name}_p1",
                              max_epochs_override=min(5, thresholds["max_epochs_cap"]))
        result2 = train_phase(PHASE_2, class_to_idx, CONFIG.num_classes, class_filter=subset,
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
        print("  BOTH PASS -- flat softmax viable, proceed with full 3,407-class run.")
    elif random_pass and not confusable_pass:
        print("  random500 passes but confusable pairs fail -- softmax scales broadly but "
              "can't separate near-duplicate fonts. Consider: accept an embarrassing-pair "
              "failure list, OR add a metric-learning head for the confusable clusters specifically.")
    elif confusable_pass and not random_pass:
        print("  confusable pairs pass but random500 fails -- unexpected; investigate before "
              "proceeding, this combination suggests a bug rather than a capacity limit.")
    else:
        print("  BOTH FAIL -- flat softmax not viable at this scale. Reconsider architecture "
              "(embedding/metric-learning head) before committing to the full run.")

    return confusable_pass and random_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-a", action="store_true")
    parser.add_argument("--check-b", action="store_true")
    parser.add_argument("--phase", type=int, choices=[1, 2])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-hours", type=float, default=None)
    args = parser.parse_args()

    class_to_idx = {name: i for i, name in enumerate(load_class_list(CONFIG.manifest_csv))}
    num_classes = len(class_to_idx)
    print(f"loaded {num_classes} classes from manifest")

    if args.check_a:
        run_check_a(class_to_idx, num_classes)
    elif args.check_b:
        run_check_b(CONFIG.manifest_csv, class_to_idx)
    elif args.phase == 1:
        train_phase(PHASE_1, class_to_idx, num_classes, resume=args.resume,
                    max_hours_override=args.max_hours)
    elif args.phase == 2:
        train_phase(PHASE_2, class_to_idx, num_classes, resume=args.resume,
                    max_hours_override=args.max_hours)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
