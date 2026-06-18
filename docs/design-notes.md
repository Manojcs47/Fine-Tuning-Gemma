## 2026-06-XX — M1 baseline results

**Run metadata**
- Config: `configs/baseline.yaml` (n_predictions=200, n_judge=50, ppl on 100 val examples)
- Base model: `unsloth/gemma-4-E2B-it`, no LoRA, bf16 on T4
- Deterministic generation (`do_sample=False`, max_new_tokens=1024)

**Quantitative (n=200)**

| Metric | Value | Notes |
|---|---|---|
| Exact match | 0.000 | Strict EM is hopeless on long free-text answers |
| Contains match | 0.020 | Only 4/200 gold answers literally inside predictions |
| Token F1 | 0.267 | The honest task signal — moderate overlap |
| Perplexity (val) | 32.51 | High; model has never seen this format |
| Mean output tokens | 495.5 | Verbose; some outputs hit the 1024-token cap |

**Judge (n=50, parsed 49/50)**

| Dimension | Score / 5 |
|---|---|
| conclusion_correctness | 3.10 |
| reasoning_validity | 3.67 |
| no_fabrication | 4.92 |
| **aggregate** | **3.90** |

**Qualitative observations from 10 generations**

1. **No `<reasoning>` blocks.** Base model never emits the tagged structure used in
   training data. It defaults to markdown — headers, bullets, bold. This single
   format mismatch explains most of the exact-match collapse.
2. **Verbose with truncation.** Multiple outputs are cut off mid-sentence at the
   1024-token cap (Examples 2, 8, 10). Mean 495 tokens vs gold's flowing
   100–250-word prose. Fine-tuning should compress this.
3. **Right knowledge, wrong target.** On the Triangle-of-Doom question (Ex. 1)
   the model confidently described *Hesselbach's triangle* — a different
   structure entirely. Knowledge is rich but mis-routed.
4. **Doesn't commit on "which is strongest" questions.** On the cellulitis
   question (Ex. 9), gold asks for the single strongest predisposing factor
   (tinea pedis). Model returned a ranked list and explicitly refused to pick:
   "If forced to choose the single strongest…" — never names tinea pedis.
5. **Reasoning paths sometimes lead to wrong specifics.** On the isoniazid
   hepatotoxicity question (Ex. 10), reasoning is structurally fine but lands
   on "enzyme induction/inhibition" instead of the correct "impaired
   acetylation." Mechanism-level errors that no_fabrication doesn't catch.
6. **Conservatism is real.** `no_fabrication=4.92` matches what the qualitative
   reads show: very few invented drug names or doses.

**Implication for fine-tuning**

- **Easy wins likely:** format adherence (`<reasoning>` tags), terser answers,
  committing to a single answer when asked.
- **Harder wins:** specific mechanism-level accuracy (acetylation vs CYP,
  triangle-of-Doom vs Hesselbach). SFT on 25k examples should move these.
- **Probably won't fix with SFT alone:** ambiguous-question handling, novel
  combinations not in the training distribution.

**Known issue — judge sample correspondence**

The first version of `run_baseline.py` sampled 50 random indices for the judge
but the qualitative markdown displayed the *first 10 generations*, pairing them
with `judge_results[0..9]` (which belong to the random sample, not the first
10). The aggregate judge metrics are correct; only the per-sample lines in
`qualitative.md` were misaligned. Fixed in the Part 6 refactor by using the
first N predictions for the judge as well — no random sampling.

---

## 2026-06-12 — M2 bug bash: four crashes between code-pull and first train step

Four distinct failures, all triggered by the Unsloth 2026.6.x + TRL 0.24.0 +
transformers 5.5.0 + T4 combination in the current Kaggle image. Documented
here because each has a different signature and is independently debuggable.

### Crash 1 — `cannot pickle 'ConfigModuleInstance' object` (fixed earlier)

**Where:** During `SFTTrainer._prepare_dataset` → `dataset.map(tokenize_fn, ...)`.

**Why:** Unsloth's `FastModel.from_pretrained` returns a `Gemma4Processor`
whose internals reference `torch._dynamo.config`, a `ConfigModuleInstance`
that `dill` cannot pickle. Even with `dataset_num_proc=1` in SFTConfig,
TRL 0.24 + datasets 3.x still triggers a `multiprocess.Pool` spawn under
some code paths, forcing the pickle.

**Fix:** `gemma_medical.data.pre_tokenize_for_training` runs the
tokenization ourselves with `num_proc=1` (no `multiprocess.Pool` at all)
and hands SFTTrainer a dataset that already has `input_ids` /
`attention_mask` / `labels`. TRL detects `is_processed=True` and skips its
own tokenization map entirely.

### Crash 2 — `TypeError: 'function' object is not subscriptable`

**Where:**
```
File ".../trl/trainer/sft_trainer.py", line 1105, in compute_loss
    per_token_entropy = entropy_from_logits(outputs.logits)
File ".../trl/trainer/utils.py", line 1542, in entropy_from_logits
    original_shape = logits.shape[:-1]
```

**Why:** Unsloth's cut-cross-entropy optimization computes the loss directly
from hidden states and never materializes the full `[batch × seq × vocab]`
logits tensor. To preserve the HF API surface, `outputs.logits` is set to
an `EmptyLogits` placeholder whose `.shape` is a *method*, not a property.
TRL ≥ 0.20 added per-token entropy logging in `compute_loss`, calling
`entropy_from_logits(outputs.logits)` → `logits.shape[:-1]`. On the
placeholder, that errors as shown above.

**Fix (current):** Do *not* force logits. Instead, bypass TRL's logit-reading
entropy/accuracy block entirely and compute the loss straight through the
`transformers.Trainer` path, which Unsloth patches with its fused, logit-free
cross-entropy. This is implemented by `_MemoryEfficientSFTTrainer` (a
`SFTTrainer` subclass) in `src/gemma_medical/train.py`, whose `compute_loss`
pops the private `_prediction_loss_only` flag, sets `use_cache=False`, and
delegates to `Trainer.compute_loss`. We keep Unsloth's memory savings and never
hit `entropy_from_logits`. Cost: `entropy` and `mean_token_accuracy` are no
longer logged; `loss`, `grad_norm` and `learning_rate` are unaffected.

> **Superseded approach (kept for history):** earlier we set
> `UNSLOTH_RETURN_LOGITS=1` before every Unsloth import to force real logits.
> It removed this TypeError but materialized a multi-GB fp32 logits tensor that
> caused Crash 4 (below). It has been removed from `__init__.py`,
> `run_train.py`, `run_baseline.py`, `run_eval.py`, and `train.run_training`.
> Perplexity never needed it either — `evaluate.compute_perplexity` reads the
> model's fused `.loss`, not `.logits`.

### Crash 3 — `ValueError: No valid checkpoint found in output directory`

**Where:** `transformers.Trainer.train(resume_from_checkpoint=True)` on a
directory that exists but contains no `checkpoint-*` subdirs.

**Why:** After crash 2 fired on step 1, no checkpoint was ever written.
Re-running with `--resume` then triggered Trainer's strict resume check.

**Fix:** `gemma_medical.train._has_valid_checkpoint(output_dir)` now scans
for `checkpoint-*` subdirs. If `--resume` was requested but none exist,
the script logs a warning and silently starts a fresh run. Matches the
Kaggle workflow expectation: "continue if possible, start fresh otherwise."

### Crash 4 — `torch.OutOfMemoryError` in `convert_to_fp32` (3.04 GiB)

**Where:**
```
File ".../accelerate/utils/operations.py", line 902, in _convert_to_fp32
    return tensor.float()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.04 GiB.
```

**Why:** Crash 2's fix (`UNSLOTH_RETURN_LOGITS=1`) has a cost: Unsloth now
materializes the full `[batch × seq × vocab]` logits tensor. T4 doesn't
support bf16 so we're using fp16 autocast, which means accelerate wraps the
model's forward with `ConvertOutputsToFp32`. That wrapper upcasts the
logits to fp32 on the way out — doubling the tensor's memory at exactly
the moment when activations + gradients + 8-bit optimizer + 5.1 B model
params are all already resident.

For Gemma 4 E2B at `per_device_batch_size=4, seq_len=2048`, the fp32
conversion needs ~3 GiB of additional contiguous memory. Free VRAM at that
point is ~2.9 GiB. Overflow.

The YAML had `per_device_train_batch_size: 2`, but the runtime kept
reporting batch=4 — turned out the Kaggle copy of the YAML had been
edited locally. The 4 ↔ 2 ↔ 1 difference doesn't help much when the
fp32 conversion of the logits is the dominant non-model memory cost: 4 →
~3 GiB, 2 → ~1.5 GiB, 1 → ~0.76 GiB. Only 1 reliably fits.

**Fix:** Three changes, in priority order:

1. **`per_device_train_batch_size: 1`** + **`gradient_accumulation_steps: 8`**
   in both `configs/lora_default.yaml` and `configs/qlora_default.yaml`.
   Effective batch stays at 8, peak VRAM drops by ~75%.
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** set in
   `__init__.py` and at the top of every script. The CUDA OOM error
   message itself recommends this — the PyTorch caching allocator coalesces
   free segments instead of holding fragmented "available but unusable"
   chunks.
3. **`per_device_eval_batch_size`** wired as an explicit config field with
   sensible default fallback to `per_device_train_batch_size`.
   `TrainingArguments` defaults to 8 if unset; at batch=8/seq=2048/fp32 the
   eval logits tensor alone is ~17 GiB — the first eval would OOM even if
   training was fine.

A `_free_cuda_memory()` call (gc + `torch.cuda.empty_cache()`) was also
added immediately before `trainer.train()` so any cached allocations from
dataset prep are released before the first training step competes for VRAM.

> **Update (2026-06-16):** Crash 4's *root cause* — the materialized fp32
> logits tensor — disappears entirely once `UNSLOTH_RETURN_LOGITS=1` is
> removed (see Crash 2's current fix). With Unsloth's logit-free path there is
> no full-vocab tensor to upcast, which frees ~3 GB of steady-state VRAM. The
> mitigations above (batch=1 + grad-accum=8, `expandable_segments:True`,
> explicit eval batch size, pre-train `_free_cuda_memory()`) are all kept as
> cheap, robust hygiene, and the headroom they now buy is what lets
> `max_seq_length` go back up to 1024. A post-training `del trainer, model` +
> `_free_cuda_memory()` was also added in `run_training` so the evaluation
> reload doesn't have to hold the training model and an inference copy at once.

### What we did NOT do (and why)

- **Pin TRL to an older version.** Unsloth 2026.6.x deps require
  `trl>=0.18.2,<=0.24.0,!=0.19.0`. We could pin to 0.18.2 to predate the
  entropy logging code path. Rejected: this fragility cuts off later TRL
  features (chunked NLL, batch eval metrics) and silently encourages
  copying old code into the future.
- **Use `loss_type="chunked_nll"`.** TRL has a memory-efficient loss type
  that chunks the LM-head matmul. Considered but rejected: chunked_nll
  also avoids materializing the full logits tensor, which means
  `outputs.logits` would again be empty/placeholder — bringing back crash
  2. The combination doesn't compose cleanly.
- **Disable mixed precision.** Setting `fp16=False, bf16=False` would skip
  the fp32 conversion entirely. Rejected: the model itself would then run
  in full fp32, which would OOM the moment it's loaded (~20 GiB vs T4's
  ~14.5 GiB).
- **Keep `max_seq_length` at 2048.** The earliest configs used 2048; under
  the forced-logits regime that was the dominant memory term and it was
  temporarily cut to 768 to survive. With logits no longer materialized the
  pressure is gone, so it now sits at **1024** — a deliberate middle ground
  that captures the long medical CoT chains (a minority of examples exceed
  1024 and are truncated cleanly) while leaving generous headroom for the
  generation spike during in-training sampling and post-training eval.
  ## M2 — First LoRA fine-tune (2026-06-17)

### Acceptance status
PASS (pipeline-level). M2's acceptance criteria are pipeline end-to-end +
loss decrease after warmup, both met. Hyperparameter quality is M3's
problem, not M2's.

### Training behavior — divergence at LR peak

Trained `lora-default-r16-lr2e4` (LR=2e-4, r=16, full 7 target modules,
adamw_8bit, fp16 on T4) for 1 epoch, expected 1169 steps. Early stopping
fired at step 400 after 3 consecutive evals plateaued.

Loss trajectory (logging_steps=10 windows):
- step 10: train_loss=1.54, grad_norm=2.3 (warming up)
- step 30: train_loss=0.79, grad_norm=0.9 (best — model is learning)
- step 50: train_loss=1.51, grad_norm=0.03 (overshoot, gradient collapse)
- step 100→400: train_loss flat at ~1.555, grad_norm ~1e-3 to 1e-4 (frozen)

Eval loss flat at 1.93 from step 100 onward; deltas below the 0.005 early
stopping threshold; patience=3 fired correctly at step 400.

### Root cause

Classic fp16 + adamw_8bit instability at peak LR. The 5% warmup ramps LR
from 0 to 2e-4 over the first ~58 steps. The model learned productively
during low-LR warmup (steps 0-30, LR going through ~1e-4), then overshot
around steps 40-50 as LR approached peak (1.5-2e-4). The 8-bit optimizer
states couldn't recover precision; gradients underflowed; the model got
stuck in a near-zero-gradient region for the rest of training.

The smoke test (12 steps, all within warmup) showed clean learning and
better judge metrics than the full run, which is consistent with this
diagnosis — the failure mode is specifically at-peak-LR.

### M2 results vs M1 baseline

| Metric              | M1 baseline | M2 (collapsed) | Delta  |
|---------------------|-------------|----------------|--------|
| Exact match         | 0.000       | 0.000          | flat   |
| Contains match      | 0.020       | 0.020          | flat   |
| Token F1            | 0.267       | 0.230          | -0.037 |
| Perplexity          | 32.51       | 9.64           | -22.9  |
| Mean output tokens  | 495.5       | 429.6          | -65.9  |
| Judge aggregate     | 3.90        | 3.77           | -0.13  |
| Judge: correctness  | 3.10        | 3.12           | flat   |
| Judge: validity     | 3.67        | 3.38           | -0.29  |
| Judge: fabrication  | 4.92        | 4.82           | flat   |

Interpretation: perplexity dropped 70% (the model learned the dataset's
text distribution) but task-correctness metrics did not improve. The model
became more *fluent at this style* without becoming *more correct*. This
is distribution overfit caused by the early training collapse.

### Implications for M3

Search range needs to focus BELOW 2e-4. The productive learning region
was around LR=1e-4 (peak grad_norm ~1, loss dropping from 1.5 to 0.8).
M3 sweep should include 5e-5, 1e-4, and 2e-4 (where 2e-4 is included
specifically to confirm/reject the divergence finding with different
ranks). Also include a lower warmup_ratio config to test whether faster
warmup is causing the overshoot.

### Engineering wins from M2

- Eight separate integration bugs surfaced and fixed (UNSLOTH_RETURN_LOGITS,
  alloc fragmentation, dill pickle, TRL 0.16+ rename, Gemma4Processor
  positional binding, eval batch default, resume guard, env var clearing).
  All documented inline as NOTE: comments.
- Pre-tokenization workaround in data.py decouples us from TRL's broken
  multiprocess tokenization. Sweep runs will inherit this fix for free.
- Early stopping fired exactly as designed. Without it, we would have
  wasted ~2.5 more hours of T4 time on a frozen model.