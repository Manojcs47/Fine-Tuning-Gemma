"""Gemma 4 E2B medical reasoning fine-tuning package."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# CRITICAL — Unsloth + TRL 0.20+ compatibility shim (M2 bug bash #2)
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
# trading memory for compatibility with TRL's entropy logging.
# ---------------------------------------------------------------------------
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

# ---------------------------------------------------------------------------
# CUDA memory fragmentation knob (M2 bug bash #3 — OOM during fp32 conversion)
# ---------------------------------------------------------------------------
#
# Once UNSLOTH_RETURN_LOGITS=1 is set, the full fp16 [batch x seq x vocab]
# logits tensor is real. T4 doesn't support bf16, so we're using fp16
# autocast — which means accelerate's `ConvertOutputsToFp32` wrapper upcasts
# the logits to fp32 on the way out of the model. That doubles the tensor's
# memory and triggers OOM if there's any fragmentation.
#
# `expandable_segments:True` tells the PyTorch caching allocator to coalesce
# free segments instead of holding them as separate "available but unusable"
# chunks. The CUDA OOM error message itself recommends this setting.
#
# Note: the env var name is `PYTORCH_CUDA_ALLOC_CONF` (with `_CUDA_`), not
# `PYTORCH_ALLOC_CONF` as the error message says — the latter is wrong in
# older PyTorch error strings.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"