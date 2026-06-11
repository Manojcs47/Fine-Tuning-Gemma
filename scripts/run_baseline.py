"""M1 baseline measurement.

Workflow:
  1. Load base model + tokenizer.
  2. Generate predictions on the last N test examples.
  3. Compute exact-match, contains-match, token-F1.
  4. Compute perplexity on a val slice.
  5. Score N_judge predictions with LLM-judge (reuses the loaded model).
  6. Save predictions, metrics, qualitative samples to outputs/baseline/.
  7. Log everything to W&B (optional, if WANDB_API_KEY is set).

Run on Kaggle:
    !python scripts/run_baseline.py --config configs/baseline.yaml

Quick test (10 samples only):
    !python scripts/run_baseline.py --config configs/baseline.yaml --max-samples 10
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

# Make src/ importable without `pip install -e .` for raw script runs
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemma_medical.config import (  # noqa: E402
    ExperimentConfig,
    RuntimeSettings,
    load_experiment_config,
)
from gemma_medical.data import (  # noqa: E402
    apply_chat_template_to_dataset,
    load_raw_dataset,
    make_splits,
)
from gemma_medical.evaluate import (  # noqa: E402
    compute_perplexity,
    compute_task_metrics,
)
from gemma_medical.logging_setup import configure_logging, get_logger  # noqa: E402
from gemma_medical.seeds import set_seed  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1 baseline measurement")
    p.add_argument("--config", type=str, required=True, help="Path to YAML experiment config")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Override n_predictions for quick testing")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip LLM-as-judge phase (faster)")
    p.add_argument("--skip-perplexity", action="store_true",
                   help="Skip perplexity computation (faster)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output directory")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable W&B logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_predictions_jsonl(
    generations: list[Any], references: list[str], path: Path
) -> None:
    """Persist predictions to JSONL so a session crash doesn't lose them."""
    with path.open("w", encoding="utf-8") as f:
        for gen, ref in zip(generations, references, strict=True):
            row = {
                "question": gen.question,
                "raw_output": gen.raw_output,
                "reasoning": gen.reasoning,
                "answer": gen.answer,
                "n_tokens_out": gen.n_tokens_out,
                "reference": ref,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("predictions_saved", path=str(path), n=len(generations))


def save_qualitative_md(
    generations: list[Any],
    references: list[str],
    judge_results: list[Any] | None,
    n: int,
    path: Path,
) -> None:
    """Dump first N generations as markdown for human reading."""
    lines: list[str] = ["# Baseline Qualitative Samples", ""]
    for i in range(min(n, len(generations))):
        gen = generations[i]
        ref = references[i]
        lines.append(f"## Example {i + 1}")
        lines.append("")
        lines.append("**Question:**")
        lines.append(f"> {gen.question}")
        lines.append("")
        lines.append("**Gold answer:**")
        lines.append(f"> {ref}")
        lines.append("")
        lines.append("**Model output:**")
        lines.append("```")
        lines.append(gen.raw_output[:2000])
        lines.append("```")
        lines.append("")
        if judge_results and i < len(judge_results):
            jr = judge_results[i]
            lines.append("**Judge:**")
            lines.append(f"- conclusion_correctness: {jr.conclusion_correctness}")
            lines.append(f"- reasoning_validity: {jr.reasoning_validity}")
            lines.append(f"- no_fabrication: {jr.no_fabrication}")
            lines.append(f"- aggregate: {jr.aggregate:.2f}")
            lines.append(f"- justification: {jr.justification}")
            lines.append("")
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("qualitative_saved", path=str(path))


def save_results_json(results: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log.info("results_saved", path=str(path))


# ---------------------------------------------------------------------------
# Judge aggregation
# ---------------------------------------------------------------------------


def aggregate_judge_results(results: list[Any]) -> dict[str, float]:
    """Mean across all dimensions, plus the parse-success rate."""
    if not results:
        return {"n": 0}
    parsed = [r for r in results if r.parse_ok]
    n_total = len(results)
    n_parsed = len(parsed)
    if n_parsed == 0:
        return {"n": n_total, "n_parsed": 0, "parse_rate": 0.0}
    return {
        "n": n_total,
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / n_total,
        "conclusion_correctness": sum(r.conclusion_correctness for r in parsed) / n_parsed,
        "reasoning_validity": sum(r.reasoning_validity for r in parsed) / n_parsed,
        "no_fabrication": sum(r.no_fabrication for r in parsed) / n_parsed,
        "aggregate": sum(r.aggregate for r in parsed) / n_parsed,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    configure_logging(level="INFO")

    settings = RuntimeSettings()
    cfg: ExperimentConfig = load_experiment_config(args.config)

    if args.max_samples is not None:
        cfg.evaluation.n_predictions = args.max_samples
        cfg.evaluation.n_judge_samples = min(args.max_samples, cfg.evaluation.n_judge_samples)
        cfg.evaluation.n_qualitative_samples = min(args.max_samples, cfg.evaluation.n_qualitative_samples)

    set_seed(cfg.training.seed)

    # Output paths
    out_dir = Path(args.output_dir) if args.output_dir else (settings.output_dir / "baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output_dir", path=str(out_dir))

    # ----- W&B init (optional) -----
    use_wandb = (not args.no_wandb) and bool(settings.wandb_api_key or os.environ.get("WANDB_API_KEY"))
    wandb_run = None
    if use_wandb:
        try:
            import wandb  # type: ignore[import-not-found]
            wandb_run = wandb.init(
                project=settings.wandb_project,
                entity=settings.wandb_entity or None,
                name=cfg.name,
                tags=["baseline", "m1", cfg.model.base_model.split("/")[-1]],
                config=cfg.model_dump(),
            )
            log.info("wandb_initialized", run_id=wandb_run.id)
        except Exception as e:
            log.warning("wandb_init_failed", error=str(e))
            use_wandb = False

    # ----- Load data -----
    raw = load_raw_dataset(cfg.data)
    splits = make_splits(raw, cfg.data)

    n_pred = min(cfg.evaluation.n_predictions, len(splits.test))
    test_subset = splits.test.select(range(n_pred))
    questions = [ex["Question"] for ex in test_subset]
    references = [ex["Response"] for ex in test_subset]
    log.info("evaluation_subset", n_predictions=n_pred)

    # ----- Load model (GPU-only imports happen here) -----
    from gemma_medical.inference import (
        generate_predictions,
        load_model_for_inference,
    )

    model, tokenizer = load_model_for_inference(cfg.model)

    # ----- Phase 1: Generate predictions -----
    generations = generate_predictions(model, tokenizer, questions, cfg.evaluation)
    save_predictions_jsonl(generations, references, out_dir / "predictions.jsonl")

    # ----- Phase 2: Task metrics -----
    pred_answers = [g.answer for g in generations]
    out_token_counts = [g.n_tokens_out for g in generations]
    task_metrics = compute_task_metrics(pred_answers, references, out_token_counts)
    log.info("task_metrics", **task_metrics.to_dict())

    # ----- Phase 3: Perplexity (optional) -----
    perplexity = float("nan")
    if not args.skip_perplexity:
        n_ppl = min(cfg.evaluation.perplexity_samples, len(splits.val))
        val_subset = splits.val.select(range(n_ppl))
        val_formatted = apply_chat_template_to_dataset(val_subset, tokenizer, num_proc=1)
        perplexity = compute_perplexity(
            model, tokenizer,
            texts=[row["text"] for row in val_formatted],
            max_length=cfg.model.max_seq_length,
        )

    # ----- Phase 4: LLM-as-judge -----
    judge_results: list[Any] = []
    judge_summary: dict[str, Any] = {}
    if not args.skip_judge:
        from gemma_medical.judge import GemmaJudge

        n_judge = min(cfg.evaluation.n_judge_samples, len(generations))
        rng = random.Random(cfg.data.seed)
        sample_idx = rng.sample(range(len(generations)), n_judge)
        triples = [
            (generations[i].question, references[i], generations[i].raw_output)
            for i in sample_idx
        ]
        log.info("judge_sampling", n=n_judge)

        # Switch model back to inference mode (it already is, but be explicit)
        judge = GemmaJudge(model, tokenizer, cfg.evaluation)
        judge_results = judge.score_batch(triples)
        judge_summary = aggregate_judge_results(judge_results)
        log.info("judge_summary", **judge_summary)

    # ----- Phase 5: Save artifacts -----
    save_qualitative_md(
        generations,
        references,
        judge_results,
        n=cfg.evaluation.n_qualitative_samples,
        path=out_dir / "qualitative.md",
    )

    summary = {
        "config_name": cfg.name,
        "technique": cfg.technique,
        "base_model": cfg.model.base_model,
        "n_predictions": n_pred,
        "task_metrics": task_metrics.to_dict(),
        "perplexity": perplexity,
        "judge": judge_summary,
        "generation_params": {
            "max_new_tokens": cfg.evaluation.max_new_tokens,
            "do_sample": cfg.evaluation.do_sample,
            "batch_size": cfg.evaluation.batch_size,
        },
    }
    save_results_json(summary, out_dir / "baseline_results.json")

    # Also save raw judge results for later inspection
    if judge_results:
        with (out_dir / "judge_results.jsonl").open("w", encoding="utf-8") as f:
            for r in judge_results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # ----- W&B logging -----
    if use_wandb and wandb_run is not None:
        import wandb  # type: ignore[import-not-found]
        wandb.log({
            "baseline/exact_match": task_metrics.exact_match,
            "baseline/contains_match": task_metrics.contains_match,
            "baseline/token_f1": task_metrics.token_f1,
            "baseline/mean_output_tokens": task_metrics.mean_output_tokens,
            "baseline/perplexity": perplexity,
            **{f"baseline/judge_{k}": v for k, v in judge_summary.items()
               if isinstance(v, (int, float))},
        })
        # Attach the qualitative markdown as a W&B artifact
        artifact = wandb.Artifact("baseline_outputs", type="evaluation")
        artifact.add_file(str(out_dir / "qualitative.md"))
        artifact.add_file(str(out_dir / "baseline_results.json"))
        artifact.add_file(str(out_dir / "predictions.jsonl"))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    # ----- Final summary print -----
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Predictions:       {n_pred}")
    print(f"Exact match:       {task_metrics.exact_match:.4f}")
    print(f"Contains match:    {task_metrics.contains_match:.4f}")
    print(f"Token F1:          {task_metrics.token_f1:.4f}")
    print(f"Perplexity:        {perplexity:.2f}")
    if judge_summary:
        print(f"Judge aggregate:   {judge_summary.get('aggregate', 'N/A')}")
        print(f"Judge parse rate:  {judge_summary.get('parse_rate', 'N/A')}")
    print(f"\nArtifacts saved to: {out_dir}/")
    print("=" * 60)

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]
        torch.cuda.empty_cache()
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())