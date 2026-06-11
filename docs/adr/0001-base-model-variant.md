# ADR-0001: Use `unsloth/gemma-4-E2B-it` as the base model

**Status:** Accepted
**Date:** 2026-06-11

## Context

The assignment requires fine-tuning a Gemma 4 variant on the free Kaggle T4
(16 GB VRAM). The Gemma 4 family includes E2B, E4B, 12B, 26B-A4B, and 31B.
The instruction-tuned (`-it`) and pretrained variants are both available.

## Decision

Use the **instruction-tuned E2B** variant, specifically the Unsloth mirror
`unsloth/gemma-4-E2B-it`, in text-only mode.

## Rationale

- **E2B fits the T4.** Per Unsloth's docs, E2B LoRA needs 8–10 GB VRAM
  and E2B QLoRA needs 6–8 GB. E4B LoRA needs 17 GB and does not fit.
- **Instruction-tuned, not pretrained.** Starting from `-it` is closer to a
  realistic deployment scenario and the SFT step is genuinely additive —
  we teach domain reasoning on top of existing instruction-following ability.
- **Unsloth mirror over Google upstream.** Identical weights, avoids the
  HF license-acknowledgement gate, and is what Unsloth's FastModel loader
  is tested against.
- **Text-only mode.** Simplifies data preparation and memory budgeting;
  the medical reasoning dataset is text-only anyway.

## Consequences

- We cannot evaluate multimodal capability — out of scope.
- If E4B QLoRA later proves a better quality/cost trade-off (Unsloth docs
  hint at this), we revisit this decision in a follow-up ADR.