"""Gemma 4 E2B medical reasoning fine-tuning package."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# CRITICAL ENV VARS — must be set before any torch import.
# ---------------------------------------------------------------------------
#
# UNSLOTH_RETURN_LOGITS=1
#   Forces Unsloth to materialize real logits tensors. Required because
#   TRL >= 0.20's SFTTrainer.compute_loss calls entropy_from_logits which
#   does logits.shape[:-1]; on Unsloth's EmptyLogits placeholder, `shape`
#   is a method, not a property, and that fails with
#   "TypeError: 'function' object is not subscriptable".
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   Without this, PyTorch's caching allocator fragments over training steps:
#   freed regions become unusable, and OOM hits even when "free" memory in
#   nvidia-smi is several hundred MB. The error message itself recommends
#   this setting. The env var name has _CUDA_ in it — the runtime error
#   text "PYTORCH_ALLOC_CONF" is wrong (a known PyTorch typo).
#
# NOTE on `=` vs `setdefault`:
#   We use direct assignment, not setdefault. Unsloth's package init clears
#   some env vars on its way through (issue #3071 et al.) — setdefault won't
#   re-set after that. The shell-level export in the Kaggle cell is the
#   authoritative source; this is belt-and-suspenders.
# ---------------------------------------------------------------------------
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Print at import time so the kernel log shows whether the env vars stuck.
# If you see them as None somewhere later, that's diagnostic — something in
# the import chain cleared them.
_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
_logits = os.environ.get("UNSLOTH_RETURN_LOGITS")
print(f"[gemma_medical] env at import: "
      f"PYTORCH_CUDA_ALLOC_CONF={_alloc!r} UNSLOTH_RETURN_LOGITS={_logits!r}")

__version__ = "0.1.0"