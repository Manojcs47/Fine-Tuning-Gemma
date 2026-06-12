"""Evaluate any saved adapter (or just the base model) against the test slice.

Usage:
  Re-evaluate a trained adapter:
    python scripts/run_eval.py --config configs/lora_default.yaml \\
        --adapter /kaggle/working/outputs/lora-adapter/lora-default-r16-lr2e4

  Evaluate the base model (no adapter):
    python scripts/run_eval.py --config configs/baseline.yaml
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CRITICAL: env-var setdefaults must run BEFORE any import that could
# transitively import unsloth or initialize CUDA.
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemma_medical.config import (  # noqa: E402
    ExperimentConfig,
    RuntimeSettings,
    load_experiment_config,
)
from gemma_medical.data import load_raw_dataset, make_splits  # noqa: E402
from gemma_medical.logging_setup import configure_logging, get_logger  # noqa: E402
from gemma_medical.seeds import set_seed  # noqa: E402

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a (possibly adapter-wrapped) model")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--adapter", type=str, default=None,
                   help="Path to a saved LoRA adapter (omit for base-model eval)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Override n_predictions for quick testing")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--skip-perplexity", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", type=str, default=None,
                   help="Tag used in result filenames / W&B (defaults to config name)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(level="INFO")

    settings = RuntimeSettings()
    settings.ensure_dirs()
    cfg: ExperimentConfig = load_experiment_config(args.config)

    if args.max_samples is not None:
        cfg.evaluation.n_predictions = args.max_samples
        cfg.evaluation.n_judge_samples = min(args.max_samples, cfg.evaluation.n_judge_samples)
        cfg.evaluation.n_qualitative_samples = min(args.max_samples, cfg.evaluation.n_qualitative_samples)

    set_seed(cfg.training.seed)

    tag = args.tag or cfg.name
    out_dir = Path(args.output_dir) if args.output_dir else (settings.output_dir / tag)

    # --- W&B ---------------------------------------------------------------
    use_wandb = not args.no_wandb and bool(
        settings.wandb_api_key or os.environ.get("WANDB_API_KEY")
    )
    wandb_run = None
    if use_wandb:
        try:
            import wandb  # type: ignore[import-not-found]
            wandb_run = wandb.init(
                project=settings.wandb_project,
                entity=settings.wandb_entity or None,
                name=f"{tag}-eval",
                tags=["eval", cfg.technique],
                config=cfg.model_dump(),
            )
        except Exception as e:
            log.warning("wandb_init_failed", error=str(e))
            use_wandb = False

    # --- Data --------------------------------------------------------------
    raw = load_raw_dataset(cfg.data)
    splits = make_splits(raw, cfg.data)

    # --- Model + optional adapter -----------------------------------------
    from gemma_medical.evaluation_pipeline import print_summary, run_evaluation
    from gemma_medical.inference import load_model_for_inference

    model, tokenizer = load_model_for_inference(cfg.model, adapter_path=args.adapter)

    # --- Run ---------------------------------------------------------------
    summary = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        test_split=splits.test,
        val_split_for_perplexity=splits.val,
        eval_cfg=cfg.evaluation,
        max_seq_length=cfg.model.max_seq_length,
        out_dir=out_dir,
        tag=tag,
        skip_perplexity=args.skip_perplexity,
        skip_judge=args.skip_judge,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
    )
    print_summary(summary)

    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())