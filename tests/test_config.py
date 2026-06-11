"""Tests for config loading and validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from gemma_medical.config import (
    ExperimentConfig,
    RuntimeSettings,
    load_experiment_config,
)

CONFIGS = Path(__file__).parent.parent / "configs"


def test_lora_default_loads() -> None:
    cfg = load_experiment_config(CONFIGS / "lora_default.yaml")
    assert cfg.technique == "lora"
    assert cfg.model.load_in_4bit is False
    assert cfg.lora.r == 16
    assert cfg.lora.use_gradient_checkpointing == "unsloth"


def test_qlora_default_loads_in_4bit() -> None:
    cfg = load_experiment_config(CONFIGS / "qlora_default.yaml")
    assert cfg.technique == "qlora"
    assert cfg.model.load_in_4bit is True  # enforced by model_post_init


def test_full_sft_implies_not_lora_path() -> None:
    cfg = ExperimentConfig(name="test-full-sft", technique="full_sft")
    assert cfg.model.full_finetuning is True
    assert cfg.model.load_in_4bit is False


def test_invalid_learning_rate_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(name="bad", training={"learning_rate": -1.0})  # type: ignore[arg-type]


def test_runtime_settings_loads() -> None:
    # Should not raise even if .env is missing — fields have defaults.
    settings = RuntimeSettings()
    assert isinstance(settings.wandb_project, str)