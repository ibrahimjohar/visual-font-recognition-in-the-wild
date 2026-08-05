# VFRW Stage C training

EfficientNet-B0, 2-phase progressive unfreezing, on `data/synthetic` (3,407 classes,
511,050 images, post emoji/symbol/blank-script cleanup — see `data_gen/font_loader.py`'s
`EXCLUDED_FAMILIES`). Full parameter reasoning lives in `config.py`'s comments and the
project's council-review history; this is just the run order.

## Order of operations

1. **`python training/probe_batch_size.py`** — binary-searches the real max batch size for
   both phases on this GPU, writes `training/probed_batch_sizes.json`. Run once; `train.py`
   reads this instead of trusting `config.py`'s placeholder numbers.
2. **`python training/train.py --check-a`** — ~40 steps, pipeline/VRAM sanity only. Just
   checks loss decreases and nothing crashes. Not a judgment call.
3. **`python training/train.py --check-b`** — architecture-viability gate. Trains two subsets
   in parallel to a pre-committed convergence threshold (see `hard_subset.py`'s
   `SMOKE_TEST_THRESHOLDS`, written down before any run so the pass/fail call can't be
   rationalized after seeing numbers): a hand-curated confusable-pairs subset (~18 classes,
   intra-family role clusters + known geometric-sans lookalikes) and a random 500-class
   subset. **Both must clear their threshold** — a small subset passing alone doesn't tell you
   softmax scales to 3,407 classes; a broad subset passing alone doesn't tell you it can
   separate near-duplicates.
4. **`python training/train.py --phase 1`** — full head-warmup run, frozen backbone.
5. **`python training/train.py --phase 2 --resume`** — full fine-tune run, last 2 MBConv
   blocks unfrozen. `--resume` picks up the latest checkpoint automatically if interrupted.

## Pause / resume / recovery

- Drop a file named `training/STOP` to pause cleanly at the next checkpoint (checked every
  epoch and every 15 min mid-epoch). Delete it before resuming.
- `--max-hours N` caps wall-clock time per phase; checkpoints and exits at the limit rather
  than running unbounded.
- Any NaN/Inf loss triggers an immediate emergency checkpoint (`*_emergency.pt`) and a hard
  stop rather than continuing to corrupt further steps.
- Checkpoints live in `training/checkpoints/`, logs in `training/logs/` (`steps.csv` per-step,
  `epochs.csv` per-epoch — includes the train/val loss-gap drift flag and macro/micro accuracy
  gap tracked per the monitoring plan).

## Known gaps not yet closed (see session notes / council review)

- Real-photo eval (`data/real_test/`) is currently empty — Check B's threshold is
  synthetic-only until real photos are added. This is a known limitation flagged explicitly,
  not an oversight: re-anchor Check B (or add a Check C) to real-photo accuracy once photos
  exist, before phase 2 is treated as fully validated.
