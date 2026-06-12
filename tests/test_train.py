"""Tests for the CPU-only helpers in gemma_medical.train.

`build_trainer` and `run_training` themselves are GPU-only (they import
unsloth/trl/transformers) so we don't test them here. We test the small
helpers because they encode the assumptions that the M2 bug bash exposed:
the "missing checkpoint" case (cell 6) and the env-var setdefaults (cells
4 and 5).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from gemma_medical.config import TrainingConfig
from gemma_medical.train import _has_valid_checkpoint


# ---------------------------------------------------------------------------
# _has_valid_checkpoint
# ---------------------------------------------------------------------------


def test_returns_false_for_missing_dir(tmp_path: Path) -> None:
    assert _has_valid_checkpoint(tmp_path / "does-not-exist") is False


def test_returns_false_for_empty_dir(tmp_path: Path) -> None:
    assert _has_valid_checkpoint(tmp_path) is False


def test_returns_false_when_only_non_checkpoint_files(tmp_path: Path) -> None:
    (tmp_path / "trainer_state.json").write_text("{}")
    (tmp_path / "config.json").write_text("{}")
    assert _has_valid_checkpoint(tmp_path) is False


def test_returns_false_when_only_non_checkpoint_subdirs(tmp_path: Path) -> None:
    (tmp_path / "wandb").mkdir()
    (tmp_path / "logs").mkdir()
    assert _has_valid_checkpoint(tmp_path) is False


def test_returns_true_when_one_checkpoint_subdir(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-200").mkdir()
    assert _has_valid_checkpoint(tmp_path) is True


def test_returns_true_when_multiple_checkpoint_subdirs(tmp_path: Path) -> None:
    for step in (100, 200, 400):
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert _has_valid_checkpoint(tmp_path) is True


def test_ignores_checkpoint_prefix_on_file(tmp_path: Path) -> None:
    # A *file* named checkpoint-XXXX (e.g. someone manually copied one) is
    # not a valid resume target — checkpoints are directories.
    (tmp_path / "checkpoint-100").write_text("not a directory")
    assert _has_valid_checkpoint(tmp_path) is False


# ---------------------------------------------------------------------------
# Env-var setdefaults
# ---------------------------------------------------------------------------


def test_package_sets_unsloth_return_logits() -> None:
    """Importing gemma_medical must set UNSLOTH_RETURN_LOGITS=1.

    Smoke test for the M2 bug-bash #2 workaround. If this regresses,
    training on Kaggle will crash with `'function' object is not subscriptable`.
    """
    import gemma_medical  # noqa: F401
    assert os.environ.get("UNSLOTH_RETURN_LOGITS") == "1"


def test_package_sets_pytorch_cuda_alloc_conf() -> None:
    """Importing gemma_medical must set PYTORCH_CUDA_ALLOC_CONF.

    Smoke test for the M2 bug-bash #3 workaround. Without this, training
    on T4 OOMs during accelerate's fp32 conversion of the materialized
    logits tensor due to memory fragmentation.
    """
    import gemma_medical  # noqa: F401
    val = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    assert "expandable_segments:True" in val


# ---------------------------------------------------------------------------
# TrainingConfig — per_device_eval_batch_size fallback
# ---------------------------------------------------------------------------


def test_effective_eval_batch_size_defaults_to_train() -> None:
    cfg = TrainingConfig(per_device_train_batch_size=1, per_device_eval_batch_size=None)
    assert cfg.effective_eval_batch_size == 1


def test_effective_eval_batch_size_uses_explicit_value() -> None:
    cfg = TrainingConfig(per_device_train_batch_size=1, per_device_eval_batch_size=4)
    assert cfg.effective_eval_batch_size == 4


def test_effective_eval_batch_size_rejects_zero() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(per_device_eval_batch_size=0)  # ge=1 constraint