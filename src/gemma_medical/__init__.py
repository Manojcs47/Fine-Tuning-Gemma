"""Gemma 4 E2B medical reasoning fine-tuning package."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# CRITICAL — Unsloth + TRL 0.20+ compatibility shim
# ---------------------------------------------------------------------------
#
# Unsloth optimizes memory by using cut-cross-entropy / chunked NLL: the loss
# is computed from hidden states directly and the full [batch x seq x vocab]
# logits tensor is never materialized. To preserve the HF API surface,
# `outputs.logits` is set to an `EmptyLogits` placeholder where `.shape` is a
# *method*, not a property.
#
# TRL >= 0.20's `SFTTrainer.compute_loss` calls `entropy_from_logits(outputs.logits)`
# for per-token entropy logging, which does `logits.shape[:-1]`. On the
# placeholder that fails with:
#     TypeError: 'function' object is not subscriptable
#
# `UNSLOTH_RETURN_LOGITS=1` forces Unsloth to materialize real logits tensors,
# trading a little memory for compatibility with TRL's entropy logging.
#
# This MUST be set before `import unsloth` anywhere in the process. Setting it
# in this package's __init__ guarantees that any script importing anything
# from `gemma_medical` (which is everything that touches Unsloth — see
# `gemma_medical.model`) will have it in place.
#
# Defense in depth: each top-level script also sets this at the very top, in
# case Python's module-cache hands back a stale entry, and `train.py` re-sets
# it right before `trainer.train()` to work around Unsloth issue #3071, where
# the env var sometimes gets cleared by an internal Unsloth code path.
# ---------------------------------------------------------------------------
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

__version__ = "0.1.0"