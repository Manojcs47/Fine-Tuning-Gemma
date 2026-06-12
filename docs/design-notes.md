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