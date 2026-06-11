# ADR-0003: Layered evaluation — exact-match + LLM-judge + qualitative

**Status:** Accepted
**Date:** 2026-06-11

## Context

Medical reasoning quality cannot be reduced to a single number. Training
loss alone is famously misleading; perplexity is interpretable but doesn't
measure correctness of the final answer.

## Decision

Use a three-layer evaluation, computed at M1 (baseline) and after every
M3 sweep run on the held-out 500-example test slice:

1. **Exact-match accuracy** on the final answer line — deterministic,
   cheap, cross-run comparable.
2. **LLM-as-judge** score on a 50-example sample, rubric covering
   conclusion correctness, validity of reasoning steps, and absence of
   fabricated entities (drug names, dosages).
3. **Qualitative read** of 10–20 generations per major run, recorded in
   `docs/design-notes.md`.

Plus training-time signals: training loss, validation loss, train-val gap,
and perplexity = exp(eval_loss).

## Rationale

- Exact-match catches gross regressions cheaply.
- LLM-judge catches reasoning-quality changes that exact-match misses
  (right answer, wrong reasoning, or vice versa).
- Qualitative reads catch failure modes neither metric surfaces
  (repetition collapse, fabricated drug names, format breakage).

## Consequences

- Each evaluation pass costs a few hundred inference calls + ~50 judge
  calls. Budget ~5–10 minutes of T4 time per run for eval.
- The judge model is a confounder; we fix the judge across runs and
  document which model/version we used.