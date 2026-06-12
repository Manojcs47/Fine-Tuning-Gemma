"""Tests for the CPU-only helpers in gemma_medical.train.

`build_trainer` and `run_training` themselves are GPU-only (they import
unsloth/trl/transformers) so we don't test them here. We test the small
helper `_has_valid_checkpoint` because it's pure I/O on a directory and
because the "missing checkpoint" case is exactly the regression the M2
bug bash exposed (cell 6).
"""
from __future__ import annotations

from pathlib import Path

from gemma_medical.train import _has_valid_checkpoint


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


def test_package_sets_unsloth_return_logits() -> None:
    """Importing gemma_medical must set UNSLOTH_RETURN_LOGITS=1.

    This is a smoke test for the env-var workaround documented in
    src/gemma_medical/__init__.py. If this regresses, training on Kaggle
    will fail again with `'function' object is not subscriptable`.
    """
    import os
    import gemma_medical  # noqa: F401  (just need the side effect)
    assert os.environ.get("UNSLOTH_RETURN_LOGITS") == "1"