# 0004 — M3 Hyperparameter Sweep Design

Status: Accepted
Date: 2026-06-17

## Context

Assignment §10 requires a sweep over at least 3 LRs, 3 LoRA ranks, and
2 additional hyperparameters, for at least 8 runs total. Two constraints
shape the design:

1. **Kaggle T4 quota**: 30 hours/week. An M2-equivalent run is ~4 hours
   (75 min train + 165 min eval). Eight such runs would consume the
   whole quota with no margin for failed runs or the final eval rerun.
2. **M2 collapse signal**: at LR=2e-4 with adamw_8bit on T4 (fp16 only),
   the model overshot at peak warmup LR (~step 50) and got stuck in a
   near-zero-gradient region. This rules out anchoring the sweep at 2e-4.

## Decision

### Anchor + single-axis deltas

All eight runs share max_seq_length=768, per_device_batch=1, grad_accum=8,
adamw_8bit, fp16. The anchor cell (LR=1e-4, rank=16, alpha=16, all-7
target modules, warmup_ratio=0.05) is the middle of the LR sweep, middle
of the rank sweep, and baseline for the three variations. Every other
run differs from anchor in exactly one dimension. This makes the
per-run delta attributable to a single hyperparameter.

| Run | LR | rank | α | targets | warmup | Purpose |
|---|---|---|---|---|---|---|
| m3-lr5e5-r16 | 5e-5 | 16 | 16 | all-7 | 0.05 | LR sweep — low |
| m3-lr1e4-r16 | 1e-4 | 16 | 16 | all-7 | 0.05 | LR sweep — mid (anchor) |
| m3-lr2e4-r16 | 2e-4 | 16 | 16 | all-7 | 0.05 | LR sweep — high (M2 reproduce) |
| m3-lr1e4-r8 | 1e-4 | 8 | 8 | all-7 | 0.05 | Rank — half |
| m3-lr1e4-r32 | 1e-4 | 32 | 32 | all-7 | 0.05 | Rank — double |
| m3-lr1e4-r16-attn | 1e-4 | 16 | 16 | attn-4 | 0.05 | Variation: scope |
| m3-lr1e4-r16-warmup10 | 1e-4 | 16 | 16 | all-7 | 0.10 | Variation: warmup |
| m3-lr1e4-r16-alpha32 | 1e-4 | 16 | 32 | all-7 | 0.05 | Variation: α/r ratio |

### Trimmed per-run eval

- num_train_epochs: 0.5 (M2 showed full epochs aren't needed for diagnosis)
- early_stopping_patience: 2 (kills collapsed runs faster)
- n_predictions: 50, n_judge_samples: 10

Estimated per-run wall: 75–90 min train + ~25 min eval ≈ 100–115 min.
Total sweep ≈ 15 hours. Comfortable inside 30-hour weekly quota.

After identifying the winner, `scripts/run_eval.py` reruns it with full
M2-style eval (200 predictions, 50 judge) so the reported number is
directly comparable to the M1 baseline.

### Resumability

`scripts/run_sweep.py` skips runs whose results JSON already exists
when invoked with `--resume`. This handles Kaggle's 12-hour session
limit: if the session times out mid-sweep, re-running the cell with
`--resume` picks up exactly where it stopped.

### Per-run isolation

Each run is executed as a fresh subprocess. GPU memory state, fragmented
allocators, lingering tensors from PEFT — none of these can leak between
runs. A bad config that OOMs or crashes does not poison the sweep.

## Alternatives considered

- **Full 3×3 LR × rank grid**: 9 runs leaves no slots for the two
  "other" hyperparameters required by §10. Rejected.
- **Random search**: cheaper exploration but the final report needs a
  clean delta table; structured sweep produces that naturally. Rejected.
- **Larger eval per sweep run**: would force fewer total runs. Picked
  "small eval + final full-eval rerun" as the better exploration /
  exploitation trade-off.

## Consequences

- The sweep characterizes the LR × rank surface near LR=1e-4 only. It
  will not catch a hypothetical productive region at e.g. LR=3e-4 with
  low rank. Acceptable: M2 already showed instability above LR=2e-4.
- The "trim eval to 50 + final rerun" pattern means M5 requires two
  separate invocations: `run_sweep.py` for exploration, `run_eval.py`
  for the winner's full eval. Documented in this ADR.