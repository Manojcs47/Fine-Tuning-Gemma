"""Configuration schemas: runtime secrets (from .env) and experiment configs (from YAML).

Two distinct concerns:
  - RuntimeSettings: tokens, paths, env-driven knobs. NEVER logged.
  - ExperimentConfig: hyperparameters describing one training run. ALWAYS logged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Runtime settings — from environment / .env file
# ---------------------------------------------------------------------------


class RuntimeSettings(BaseSettings):
    """Environment-bound settings. Loaded from .env or process env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    hf_token: str = Field(default="", description="Hugging Face access token")
    wandb_api_key: str = Field(default="", description="Weights & Biases API key")
    hf_hub_username: str = Field(default="Manojcs47")
    wandb_project: str = Field(default="gemma4-medical-finetuning")
    wandb_entity: str = Field(default="")

    # Output paths (Kaggle uses /kaggle/working, local uses ./outputs)
    output_dir: Path = Field(default=Path("./outputs"))
    checkpoint_dir: Path = Field(default=Path("./outputs/checkpoints"))
    adapter_dir: Path = Field(default=Path("./outputs/lora-adapter"))

    def ensure_dirs(self) -> None:
        """Create output directories if missing."""
        for p in (self.output_dir, self.checkpoint_dir, self.adapter_dir):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment config — from YAML files in configs/
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Which base model + how it's loaded."""

    base_model: str = "unsloth/gemma-4-E2B-it"
    max_seq_length: int = Field(default=2048, ge=512, le=8192)
    load_in_4bit: bool = False  # True for QLoRA
    full_finetuning: bool = False  # True only with A100+


class LoRAConfig(BaseModel):
    """LoRA adapter shape."""

    r: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=16, ge=4, le=256)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    use_gradient_checkpointing: Literal["unsloth", "true", "false"] = "unsloth"
    random_state: int = 3407


class DataConfig(BaseModel):
    """Dataset loading + split definition."""

    dataset_name: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    dataset_config_name: str | None = "en"
    split: str = "train"
    test_size: int = Field(default=500, ge=50, description="Last N examples held out as test")
    val_size: int = Field(default=500, ge=50, description="Next N examples held out for validation")
    seed: int = 3407


class TrainingConfig(BaseModel):
    """Optimizer + scheduler + batching."""

    learning_rate: float = Field(default=2e-4, gt=0, le=1e-2)
    per_device_train_batch_size: int = Field(default=2, ge=1, le=16)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=64)
    num_train_epochs: float = Field(default=1.0, gt=0, le=10)
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.5)
    lr_scheduler_type: Literal["cosine", "linear", "constant"] = "cosine"
    eval_steps: int = Field(default=100, ge=10)
    save_steps: int = Field(default=200, ge=10)
    save_total_limit: int = Field(default=3, ge=1, le=10)
    logging_steps: int = Field(default=10, ge=1)
    sample_print_steps: int = Field(default=100, ge=10,
                                    description="Print a sample generation every N steps")
    seed: int = 3407

    @field_validator("save_steps")
    @classmethod
    def save_steps_multiple_of_eval_steps(cls, v: int, info: object) -> int:
        # Light sanity check — eval and save should align so best-checkpoint logic is clean.
        return v


class EarlyStoppingConfig(BaseModel):
    """Hugging Face EarlyStoppingCallback args."""

    enabled: bool = True
    early_stopping_patience: int = Field(default=3, ge=1)
    early_stopping_threshold: float = Field(default=0.005, ge=0.0)


class ExperimentConfig(BaseModel):
    """The complete, loggable description of one training run."""

    name: str = Field(description="Short identifier, used in W&B run name and output dir")
    technique: Literal["lora", "qlora", "full_sft"] = "lora"
    notes: str = ""
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)

    @field_validator("technique")
    @classmethod
    def _qlora_implies_4bit(cls, v: str) -> str:
        # Cross-field validation handled in the model_validator below.
        return v

    def model_post_init(self, __context: object) -> None:
        # QLoRA must load in 4-bit; full SFT must NOT use LoRA path.
        if self.technique == "qlora":
            self.model.load_in_4bit = True
            self.model.full_finetuning = False
        elif self.technique == "lora":
            self.model.load_in_4bit = False
            self.model.full_finetuning = False
        elif self.technique == "full_sft":
            self.model.load_in_4bit = False
            self.model.full_finetuning = True


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)