"""Train a Gemma 4 E2B LoRA/QLoRA adapter and (optionally) evaluate it.

Usage:
  Smoke test:
    !PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python \\
        scripts/run_train.py --config configs/lora_default.yaml --smoke-test

  Full M2 run:
    !PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python \\
        scripts/run_train.py --config configs/lora_default.yaml

  Resume from latest checkpoint (silently starts fresh if none exists):
    !PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python \\
        scripts/run_train.py --config configs/lora_default.yaml --resume

The shell-inline `PYTORCH_CUDA_ALLOC_CONF=...` is the authoritative way to
set this — Python-side setting can be undone by Unsloth's package init
before PyTorch's CUDA allocator initializes.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Set the allocator config before ANY import that could transitively import
# torch. The canonical place to set PYTORCH_CUDA_ALLOC_CONF is the shell level
# (the `!VAR=... python` prefix in the Kaggle cell), since that's the only
# thing PyTorch is guaranteed to see before the CUDA allocator initializes;
# this is belt-and-suspenders for direct `python scripts/run_train.py` runs.
#
# We intentionally do NOT set UNSLOTH_RETURN_LOGITS=1 — that forces the full
# ~262K-vocab logits tensor and OOMs a T4. See src/gemma_medical/__init__.py
# and train._MemoryEfficientSFTTrainer for the full story.
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

# Make src/ importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemma_medical.config import (  # noqa: E402
    ExperimentConfig,
    RuntimeSettings,
    load_experiment_config,
)
from gemma_medical.logging_setup import configure_logging, get_logger  # noqa: E402
from gemma_medical.seeds import set_seed  # noqa: E402

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a LoRA/QLoRA adapter on Gemma 4 E2B")
    p.add_argument("--config", type=str, required=True, help="Path to YAML experiment config")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint; silently starts fresh if none found")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny config override: ~12 train steps, 10 eval samples")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip post-training evaluation")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    p.add_argument("--tags", type=str, default="m2,train",
                   help="Comma-separated W&B tags (default for M2; "
                        "sweep passes m3-specific tags)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output directory for evaluation artifacts")
    return p.parse_args()


def apply_smoke_test_overrides(cfg: ExperimentConfig) -> None:
    """Shrink everything so the pipeline runs end-to-end in ~5 minutes.

    Critically: in-training eval is DISABLED for smoke tests. The OOM at
    eval-step-10 is the dominant failure mode on T4, and a smoke test's
    job is to verify the pipeline plumbing, not to exercise the eval loop
    (post-training eval runs through evaluation_pipeline, which uses
    model.generate() and has a different memory profile).
    """
    cfg.training.num_train_epochs = 0.01      # ~12 train steps
    cfg.training.save_steps = 20              # save once
    cfg.training.logging_steps = 5
    cfg.training.sample_print_steps = 10      # operator-side signal still fires
    cfg.training.disable_intraining_eval = True   # <<< the OOM fix
    cfg.evaluation.n_predictions = 10
    cfg.evaluation.n_judge_samples = 5
    cfg.evaluation.n_qualitative_samples = 3
    cfg.evaluation.perplexity_samples = 10
    cfg.early_stopping.enabled = False
    log.info("smoke_test_overrides_applied",
             in_training_eval="disabled")


def init_wandb(cfg: ExperimentConfig, settings: RuntimeSettings, tags: list[str]) -> Any:
    """Initialize W&B if API key is available. Returns the run object or None."""
    key = settings.wandb_api_key or os.environ.get("WANDB_API_KEY")
    if not key:
        log.info("wandb_disabled_no_key")
        return None
    try:
        import wandb  # type: ignore[import-not-found]
        run = wandb.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity or None,
            name=cfg.name,
            tags=tags + [cfg.technique, cfg.model.base_model.split("/")[-1]],
            config=cfg.model_dump(),
            resume="allow",
        )
        log.info("wandb_initialized", run_id=run.id)
        return run
    except Exception as e:
        log.warning("wandb_init_failed", error=str(e))
        return None


def main() -> int:
    args = parse_args()
    configure_logging(level="INFO")

    settings = RuntimeSettings()
    settings.ensure_dirs()
    cfg: ExperimentConfig = load_experiment_config(args.config)

    if args.smoke_test:
        apply_smoke_test_overrides(cfg)

    set_seed(cfg.training.seed)

    # --- W&B init ----------------------------------------------------------
    use_wandb = not args.no_wandb
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()]
    wandb_run = init_wandb(cfg, settings, tags=tag_list) if use_wandb else None
    use_wandb = wandb_run is not None

    # --- Train -------------------------------------------------------------
    from gemma_medical.train import run_training  # GPU-only import

    log.info("starting_training", config_name=cfg.name, technique=cfg.technique)
    adapter_path, splits, eval_model, eval_tokenizer = run_training(
        cfg, settings, use_wandb=use_wandb, resume=args.resume
    )
    log.info("training_done", adapter_path=str(adapter_path))

    # --- Free leftover caches before eval ---------------------------------
    # run_training already dropped the trainer (optimizer + grad buffers) and
    # switched the model into inference mode; just sweep the allocator.
    try:
        import torch  # type: ignore[import-not-found]
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        pass

    # --- Post-training evaluation ------------------------------------------
    if args.skip_eval:
        log.info("skipping_post_training_eval")
        if wandb_run is not None:
            wandb_run.finish()
        return 0

    from gemma_medical.evaluation_pipeline import print_summary, run_evaluation

    # Confirm the adapter actually persisted to disk (an M2 acceptance
    # criterion) WITHOUT reloading a second ~10 GB base model — we evaluate the
    # already-trained, in-memory model, which is the same base+adapter a reload
    # would reconstruct. Two base-model copies do not fit on a 16 GB T4.
    adapter_ok = (Path(adapter_path) / "adapter_config.json").exists()
    log.info(
        "adapter_persisted_check",
        path=str(adapter_path),
        ok=adapter_ok,
        note="evaluating in-memory trained model (no reload, to fit T4 VRAM)",
    )

    out_dir = Path(args.output_dir) if args.output_dir else (
        settings.output_dir / cfg.name
    )
    summary = run_evaluation(
        model=eval_model,
        tokenizer=eval_tokenizer,
        test_split=splits.test,
        val_split_for_perplexity=splits.val,
        eval_cfg=cfg.evaluation,
        max_seq_length=cfg.model.max_seq_length,
        out_dir=out_dir,
        tag=cfg.name,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
    )
    print_summary(summary)

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())