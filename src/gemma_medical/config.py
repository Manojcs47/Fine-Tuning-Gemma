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
    """Dataset loading + split definition.

    Note on `dataset_config`: as of mid-2026 the medical-o1-reasoning-SFT dataset
    splits into four language configs (`en`, `zh`, `en_mix`, `zh_mix`) and the
    Hub requires picking one. We default to `en` since the assignment is in
    English; older versions of the dataset did not need this argument.
    """

    dataset_name: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    dataset_config: str | None = Field(
        default="en",
        description="HF dataset config/subset name. Use None for single-config datasets.",
    )
    split: str = "train"
    test_size: int = Field(default=500, ge=50, description="Last N examples held out as test")
    val_size: int = Field(default=500, ge=50, description="Next N examples held out for validation")
    seed: int = 3407


class TrainingConfig(BaseModel):
    """Optimizer + scheduler + batching."""

    learning_rate: float = Field(default=2e-4, gt=0, le=1e-2)
    # Per-device batch sizing. On T4 with UNSLOTH_RETURN_LOGITS=1, train batch must
    # stay at 1 (see configs/lora_default.yaml batch sizing note). eval batch
    # defaults to None which means "match train"; setting it explicitly avoids
    # TrainingArguments' default of 8 (which would OOM at the first eval).
    per_device_train_batch_size: int = Field(default=1, ge=1, le=16)
    per_device_eval_batch_size: int | None = Field(
        default=None, ge=1, le=16,
        description="Defaults to per_device_train_batch_size if None.",
    )
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=64)
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
    eval_dataset_size: int = Field(default=100, ge=10, le=500,
                                    description="Val examples used for training-time eval (smaller = faster eval)")

    # ----- Memory-safety knobs (M2 OOM fix) ----------------------------------
    eval_accumulation_steps: int | None = Field(
        default=1, ge=1, le=64,
        description=(
            "Move eval logits from GPU to CPU every N batches. With Gemma 4's "
            "262K vocab the per-batch logits tensor is ~1 GB at seq=1024; "
            "without this setting they accumulate on GPU and OOM."
        ),
    )
    disable_intraining_eval: bool = Field(
        default=False,
        description=(
            "Skip evaluation during training entirely (eval_strategy='no'). "
            "Useful for smoke tests on tight VRAM budgets — post-training eval "
            "still runs through evaluation_pipeline, which uses model.generate "
            "(not compute_loss) and has its own memory profile."
        ),
    )

    seed: int = 3407

    @field_validator("save_steps")
    @classmethod
    def save_steps_multiple_of_eval_steps(cls, v: int, info: object) -> int:
        return v

    @property
    def effective_eval_batch_size(self) -> int:
        """`per_device_eval_batch_size` falling back to `per_device_train_batch_size`."""
        return self.per_device_eval_batch_size or self.per_device_train_batch_size

class EarlyStoppingConfig(BaseModel):
    """Hugging Face EarlyStoppingCallback args."""

    enabled: bool = True
    early_stopping_patience: int = Field(default=3, ge=1)
    early_stopping_threshold: float = Field(default=0.005, ge=0.0)


class EvaluationConfig(BaseModel):
    """Inference + metric + judge parameters used at M1 and after every M3 run."""

    # How much to evaluate
    n_predictions: int = Field(default=200, ge=10, le=500,
                                description="Test examples to generate predictions on")
    n_judge_samples: int = Field(default=50, ge=5, le=200,
                                  description="Predictions to score with LLM-judge")
    n_qualitative_samples: int = Field(default=10, ge=1, le=50,
                                        description="Examples to dump to markdown for human reading")
    perplexity_samples: int = Field(default=100, ge=10, le=500,
                                     description="Val examples for perplexity computation")

    # Generation parameters (deterministic for fair cross-run comparison)
    max_new_tokens: int = Field(default=1024, ge=64, le=4096)
    do_sample: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=64, ge=0)
    batch_size: int = Field(default=4, ge=1, le=16,
                             description="Batch size for batched generation")

    # Judge
    judge_model: str = Field(default="unsloth/gemma-4-E2B-it",
                              description="HF model id used as LLM-judge")
    judge_max_new_tokens: int = Field(default=256, ge=64, le=1024)
    reuse_inference_model_as_judge: bool = Field(default=True,
                                                  description="Skip judge model reload if it matches the inference model")


class ExperimentConfig(BaseModel):
    """The complete, loggable description of one training run."""

    name: str = Field(description="Short identifier, used in W&B run name and output dir")
    technique: Literal["lora", "qlora", "full_sft", "baseline"] = "lora"
    notes: str = ""
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def model_post_init(self, __context: object) -> None:
        # Cross-technique constraints
        if self.technique == "qlora":
            self.model.load_in_4bit = True
            self.model.full_finetuning = False
        elif self.technique == "lora":
            self.model.load_in_4bit = False
            self.model.full_finetuning = False
        elif self.technique == "full_sft":
            self.model.load_in_4bit = False
            self.model.full_finetuning = True
        elif self.technique == "baseline":
            # Baseline: no training, no LoRA. Load base in bf16.
            self.model.load_in_4bit = False
            self.model.full_finetuning = False


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)