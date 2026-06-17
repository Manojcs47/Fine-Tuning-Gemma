"""Gemma 4 E2B medical reasoning fine-tuning package."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# CRITICAL ENV VARS — must be set before any torch import.
# ---------------------------------------------------------------------------
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   Without this, PyTorch's caching allocator fragments over training steps:
#   freed regions become unusable and OOM hits even when "free" memory in
#   nvidia-smi is several hundred MB. The error message itself recommends this
#   setting. PyTorch reads it ONCE, when the CUDA allocator first initializes,
#   so the authoritative place to set it is the shell (the `!VAR=... python`
#   prefix in the Kaggle cell). Setting it here is belt-and-suspenders for the
#   case where the package is imported before the first CUDA call.
#
# NOTE — we deliberately DO NOT set UNSLOTH_RETURN_LOGITS=1 anymore.
#   That flag forces Unsloth to materialize the full [batch, seq, ~262K] logits
#   tensor on every step (several GB in fp32). It was originally added to dodge
#   a TRL crash (SFTTrainer.compute_loss runs entropy_from_logits, which does
#   `logits.shape[:-1]` and blows up on Unsloth's EmptyLogits sentinel). But on
#   a 16 GB T4 the forced logits tensor OOMs within ~10 steps. The real fix is
#   to keep Unsloth's logit-free fused cross-entropy and skip TRL's logit-based
#   entropy metric instead — see gemma_medical.train._MemoryEfficientSFTTrainer.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Print at import time so the kernel log shows whether the alloc-conf stuck.
_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
print(f"[gemma_medical] env at import: PYTORCH_CUDA_ALLOC_CONF={_alloc!r}")

__version__ = "0.1.0"