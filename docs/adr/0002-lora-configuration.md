# ADR-0002: LoRA as default fine-tuning technique

**Status:** Accepted
**Date:** 2026-06-11

## Context

Three viable techniques for the T4: full SFT, LoRA, QLoRA. Full SFT needs
~24 GB and does not fit. LoRA (8–10 GB) and QLoRA (6–8 GB) both fit.

## Decision

- **Default:** LoRA with `r=16`, `lora_alpha=16`, dropout `0`, targeting
  all attention + MLP projections (`q,k,v,o,gate,up,down`).
- **Gradient checkpointing:** `use_gradient_checkpointing="unsloth"` (the
  string, not a boolean — this is Unsloth's optimized path).
- **QLoRA (4-bit):** run for the M4 comparison; not the default.

## Rationale

- LoRA at r=16 with all-module targeting is the Unsloth-recommended starting
  point and has the best published quality/speed trade-off for Gemma 4 E2B.
- 4-bit quantization in QLoRA costs a small quality delta we want to
  measure, not assume. M4 is the milestone that quantifies it.
- Targeting MLP layers in addition to attention typically improves quality
  vs attention-only at modest extra cost.

## Consequences

- Adapter checkpoints are ~50–200 MB (cheap to version/push).
- We can compose multiple adapters later if we extend the experiment.
- We will not produce a merged full-weight checkpoint unless explicitly
  needed for the demo.