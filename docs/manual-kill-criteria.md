# Manual Kill Criteria for Training Runs

Operator-driven decisions. These complement `EarlyStoppingCallback` but
catch failure modes the callback cannot see (memory leaks, generation
collapse, etc.). Refine over time.

1. **Training loss does not decrease in the first 200 steps after warmup.**
   *Cause:* LR too low, or `target_modules` too restrictive.
   *Action:* Kill, raise LR by 2×, restart.

2. **Training loss spikes >50% step-over-step.**
   *Cause:* LR too high, or a bad batch.
   *Action:* Kill, halve LR, restart.

3. **Validation loss has not improved for 5 consecutive eval windows while
   training loss continues to fall.**
   *Cause:* Overfitting.
   *Action:* Kill, use best checkpoint.

4. **GPU memory climbs every step instead of staying flat.**
   *Cause:* Memory leak in callback or collator.
   *Action:* Kill, fix leak, restart.

5. **Printed samples become repetitive, single-token, or nonsensical.**
   *Cause:* Catastrophic forgetting or gradient explosion.
   *Action:* Kill immediately, do not let the run finish.

6. **Train-val gap doubles within a single eval window.**
   *Cause:* Sudden overfitting on a memorized batch.
   *Action:* Kill, reduce LR or rank, restart.