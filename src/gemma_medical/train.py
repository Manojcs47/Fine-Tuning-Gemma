"""Top-level training orchestration.

Workflow (build_trainer + run_training):
  1. Load base model + tokenizer (Unsloth FastModel).
  2. Attach LoRA adapters.
  3. Load and chat-template the dataset; subset val for fast eval-during-training.
  4. Build SFTTrainer with all callbacks (sample printer, memory monitor,
     structured logger, optional EarlyStoppingCallback).
  5. Train (with optional resume_from_checkpoint).
  6. Save the LoRA adapter to disk.

GPU-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gemma_medical.config import ExperimentConfig, RuntimeSettings
from gemma_medical.data import Splits, prepare_datasets
from gemma_medical.logging_setup import get_logger
from gemma_medical.model import build_model_for_experiment
from gemma_medical.train_callbacks import (
    MemoryMonitorCallback,
    SamplePrinterCallback,
    StructuredLoggerCallback,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hardware autodetection
# ---------------------------------------------------------------------------


def _detect_dtype_flags() -> tuple[bool, bool]:
    """Return (bf16, fp16) flags based on what the GPU supports.

    T4 (Turing): no bf16 → use fp16.
    A100/H100 (Ampere+): bf16 preferred.
    """
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_bf16_supported():
            return True, False
        return False, True
    except ImportError:
        return False, False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_trainer(
    cfg: ExperimentConfig,
    settings: RuntimeSettings,
    output_dir: Path,
    *,
    use_wandb: bool,
) -> tuple[Any, Splits, Any, Any]:
    """Construct model + data + SFTTrainer. Returns (trainer, splits, model, tokenizer)."""
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
    from transformers import EarlyStoppingCallback  # type: ignore[import-not-found]

    # --- Model -------------------------------------------------------------
    log.info("build_trainer_loading_model")
    model, tokenizer = build_model_for_experiment(cfg)

# --- Data --------------------------------------------------------------
    log.info("build_trainer_preparing_data")
    splits = prepare_datasets(cfg.data, tokenizer)

    # Pre-tokenize train + eval so SFTTrainer treats them as `is_processed`
    # and skips its own tokenization map call. See
    # gemma_medical.data.pre_tokenize_for_training for the full story —
    # this is what actually fixes the dill ConfigModuleInstance crash.
    from gemma_medical.data import pre_tokenize_for_training

    train_tokenized = pre_tokenize_for_training(
        splits.train, tokenizer, max_length=cfg.model.max_seq_length
    )

    n_eval = min(cfg.training.eval_dataset_size, len(splits.val))
    eval_subset = splits.val.select(range(n_eval))
    eval_tokenized = pre_tokenize_for_training(
        eval_subset, tokenizer, max_length=cfg.model.max_seq_length
    )
    log.info("eval_subset_size", n=n_eval, full_val=len(splits.val))

    # --- SFTConfig ---------------------------------------------------------
    bf16, fp16 = _detect_dtype_flags()
    log.info("dtype_flags", bf16=bf16, fp16=fp16)

    report_to = ["wandb"] if use_wandb else ["none"]

    # NOTE on TRL API changes (TRL >= 0.16, bundled with Unsloth 2026.6+):
    #   - SFTConfig.max_seq_length was renamed to `max_length`.
    #   - SFTTrainer's `tokenizer=` kwarg was renamed to `processing_class=`.
    #
    # NOTE on `dataset_num_proc=1`:
    #   Defense in depth only — the real fix is the pre-tokenization above.
    #   Unsloth's tokenizer references torch._dynamo.config.ConfigModuleInstance,
    #   which dill cannot pickle. Setting dataset_num_proc=1 here is NOT enough
    #   on its own: TRL/datasets still spawn a multiprocess.Pool in some code
    #   paths regardless of this setting. We sidestep the whole problem by
    #   feeding SFTTrainer a dataset that already has input_ids — TRL detects
    #   is_processed=True and skips the broken tokenization map. This setting
    #   protects any subsequent map calls (EOS append, truncate) that don't
    #   drag the Dynamo config into their closures.
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        weight_decay=cfg.training.weight_decay,
        logging_steps=cfg.training.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        seed=cfg.training.seed,
        bf16=bf16,
        fp16=fp16,
        optim="adamw_8bit",
        dataset_text_field="text",
        max_length=cfg.model.max_seq_length,   # was: max_seq_length=...
        dataset_num_proc=1,                    # was: 2 — see note above
        packing=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        run_name=cfg.name,
    )

    # --- Callbacks ---------------------------------------------------------
    callbacks: list[Any] = [
        SamplePrinterCallback(
            tokenizer=tokenizer,
            val_dataset_raw=splits.test,  # use test slice (raw, has Question/Response) for sampling
            every_n_steps=cfg.training.sample_print_steps,
            max_new_tokens=300,
        ),
        MemoryMonitorCallback(log_every_n_steps=cfg.training.eval_steps),
        StructuredLoggerCallback(),
    ]

    if cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping.early_stopping_threshold,
            )
        )
        log.info(
            "early_stopping_enabled",
            patience=cfg.early_stopping.early_stopping_patience,
            threshold=cfg.early_stopping.early_stopping_threshold,
        )

    # --- Trainer -----------------------------------------------------------
    # `tokenizer=` was renamed to `processing_class=` in TRL >= 0.16.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_tokenized,     # was: splits.train
        eval_dataset=eval_tokenized,        # was: eval_subset
        args=sft_args,
        callbacks=callbacks,
    )

    return trainer, splits, model, tokenizer


def run_training(
    cfg: ExperimentConfig,
    settings: RuntimeSettings,
    *,
    use_wandb: bool = False,
    resume: bool = False,
) -> tuple[Path, Splits]:
    """Run end-to-end training. Returns (adapter_path, splits).

    The splits are returned so the caller can pass test/val to the evaluation
    pipeline without reloading the dataset.
    """
    # Per-experiment output dir under the runtime settings root.
    output_dir = settings.checkpoint_dir / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("training_output_dir", path=str(output_dir))

    trainer, splits, model, tokenizer = build_trainer(
        cfg, settings, output_dir, use_wandb=use_wandb
    )

    log.info("training_starting", resume=resume)
    train_result = trainer.train(resume_from_checkpoint=resume if resume else None)
    log.info("training_complete", metrics=train_result.metrics)

    # --- Save final (best) adapter ----------------------------------------
    adapter_dir = settings.adapter_dir / cfg.name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log.info("saving_adapter", path=str(adapter_dir))

    # For LoRA, model.save_pretrained writes only the adapter weights (~50-200 MB).
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    return adapter_dir, splits