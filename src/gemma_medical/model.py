"""Model loading and adapter attachment.

This module is the single place where Unsloth's FastModel is touched.
It cannot be imported on a CPU-only laptop — `unsloth` requires CUDA.

Three modes:
  - lora:     base in bf16/fp16, LoRA adapters attached.
  - qlora:    base in 4-bit, LoRA adapters attached.
  - full_sft: base in bf16/fp16, all parameters trainable (no LoRA).
              Requires an A100 or larger; will OOM on a T4.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gemma_medical.config import ExperimentConfig, LoRAConfig, ModelConfig
from gemma_medical.logging_setup import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
    # These imports are CUDA-only; declare as TYPE_CHECKING to keep the module
    # importable for static analysis on CPU.
    pass


def load_base_model(config: ModelConfig) -> tuple[Any, Any]:
    """Load the base model + tokenizer via Unsloth FastModel.

    Returns:
        (model, tokenizer) — exact types depend on Unsloth's internals.
    """
    # Lazy import: Unsloth is GPU-only, must not be loaded at module import time.
    from unsloth import FastModel  # type: ignore[import-not-found]

    log.info(
        "loading_base_model",
        base_model=config.base_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        full_finetuning=config.full_finetuning,
    )

    model, tokenizer = FastModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        full_finetuning=config.full_finetuning,
    )

    log.info("base_model_loaded", dtype=str(getattr(model, "dtype", "unknown")))
    return model, tokenizer


def attach_lora(model: Any, config: LoRAConfig) -> Any:
    """Attach LoRA adapters to a loaded base model.

    The critical setting is `use_gradient_checkpointing="unsloth"` — the
    string value enables Unsloth's memory-optimized checkpointing path
    which is what makes 2048-token context fit on a 16 GB T4.
    """
    from unsloth import FastModel  # type: ignore[import-not-found]

    log.info(
        "attaching_lora",
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
    )

    model = FastModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.random_state,
    )

    _log_trainable_parameters(model)
    return model


def _log_trainable_parameters(model: Any) -> None:
    """Print and log the trainable parameter count — sanity check for LoRA."""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

    pct = 100.0 * trainable / total if total > 0 else 0.0
    log.info(
        "trainable_parameters",
        trainable=trainable,
        total=total,
        percentage=round(pct, 4),
    )
    print(
        f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)"
    )


def build_model_for_experiment(config: ExperimentConfig) -> tuple[Any, Any]:
    """Top-level entry point: load base, then attach LoRA if applicable.

    For technique='full_sft', returns the base model with no adapter attached
    (all params trainable). For 'lora' and 'qlora', returns the base wrapped
    with PEFT LoRA adapters.
    """
    model, tokenizer = load_base_model(config.model)

    if config.technique == "full_sft":
        log.info("full_sft_mode_no_lora_attached")
        return model, tokenizer

    model = attach_lora(model, config.lora)
    return model, tokenizer