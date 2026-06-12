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

## 2026-06-12 — M2 bug bash: three crashes between code-pull and first train step

Three distinct failures, none in our code, all triggered by Unsloth 2026.6.3
+ TRL 0.24.0 + transformers 5.5.0 in the current Kaggle image. Documented
here because each one has a different signature and is independently
debuggable.

### Crash 1 — `cannot pickle 'ConfigModuleInstance' object` (already fixed)

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

**Fix:** Set `UNSLOTH_RETURN_LOGITS=1` *before* any Unsloth import. This is
the documented Unsloth workaround (see
`unsloth/models/_utils.py` and the Unsloth env-flags docs). It forces
Unsloth to materialize real logits tensors, trading a few hundred MB of
VRAM for compatibility with TRL's entropy logging path.

Applied in **four** places (defense in depth, because Unsloth issue #3071
reports the env var occasionally getting cleared during training):

1. `src/gemma_medical/__init__.py` — set at first package import
2. `scripts/run_train.py` — set at file-top, before any `from gemma_medical` imports
3. `scripts/run_baseline.py` and `scripts/run_eval.py` — same
4. `src/gemma_medical/train.py:run_training` — re-asserted immediately
   before `trainer.train()`

### Crash 3 — `ValueError: No valid checkpoint found in output directory`

**Where:** `transformers.Trainer.train(resume_from_checkpoint=True)` on a
directory that exists but contains no `checkpoint-*` subdirs.

**Why:** After crash 2 fired on step 1, no checkpoint was ever written.
Re-running with `--resume` then triggered Trainer's strict resume check.

**Fix:** `gemma_medical.train._has_valid_checkpoint(output_dir)` now scans
for `checkpoint-*` subdirs. If `--resume` was requested but none exist,
the script logs a warning and silently starts a fresh run. Matches the
Kaggle workflow expectation: "continue if possible, start fresh otherwise."

### What we did NOT do (and why)

- **Pin TRL to an older version.** Unsloth 2026.6.3's deps require
  `trl>=0.18.2,<=0.24.0,!=0.19.0`, so we could pin to 0.18.2 to predate
  the entropy code path. Rejected: this fragility cuts off later TRL
  features (chunked NLL, batch eval metrics) and silently encourages
  copying old code into the future. The env var is a 1-line, targeted
  workaround.
- **Disable entropy at the SFTConfig level.** Current TRL has no clean
  flag for this (entropy logging is woven into `compute_loss`), so the
  cleanest knob is on the Unsloth side.