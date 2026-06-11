"""Deterministic seeding. Call set_seed() before any randomness in training."""
from __future__ import annotations

import os
import random


def set_seed(seed: int = 3407) -> None:
    """Seed every RNG we touch. Mirrors transformers.set_seed plus extras."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore[import-not-found]
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore[import-not-found]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        from transformers import set_seed as hf_set_seed  # type: ignore[import-not-found]
        hf_set_seed(seed)
    except ImportError:
        pass