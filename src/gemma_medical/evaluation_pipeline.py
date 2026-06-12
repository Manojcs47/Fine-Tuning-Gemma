"""Reusable end-to-end evaluation: generate → metrics → perplexity → judge → save.

This module is the single place evaluation logic lives. Both run_baseline.py
(no adapter) and run_train.py / run_eval.py (with adapter) call it.

GPU-only — imports torch, calls generate.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets import Dataset

from gemma_medical.config import EvaluationConfig
from gemma_medical.data import apply_chat_template_to_dataset
from gemma_medical.evaluate import compute_perplexity, compute_task_metrics
from gemma_medical.inference import GenerationOutput, generate_predictions
from gemma_medical.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers (moved from run_baseline.py so all callers share them)
# ---------------------------------------------------------------------------


def save_predictions_jsonl(
    generations: list[GenerationOutput], references: list[str], path: Path
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
    generations: list[GenerationOutput],
    references: list[str],
    judge_results: list[Any],
    n: int,
    path: Path,
    title: str = "Qualitative Samples",
) -> None:
    """Dump first N generations + matching judge results as markdown.

    IMPORTANT: judge_results must be indexed identically to generations.
    The Part 5 implementation randomly sampled judge indices, which broke
    this correspondence in the markdown. Here we judge the first N
    sequentially — no sampling.
    """
    lines: list[str] = [f"# {title}", ""]
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
            lines.append(f"- parse_ok: {jr.parse_ok}")
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
# The pipeline
# ---------------------------------------------------------------------------


def run_evaluation(
    model: Any,
    tokenizer: Any,
    test_split: Dataset,
    val_split_for_perplexity: Dataset | None,
    eval_cfg: EvaluationConfig,
    max_seq_length: int,
    out_dir: Path,
    *,
    tag: str = "eval",
    skip_perplexity: bool = False,
    skip_judge: bool = False,
    use_wandb: bool = False,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """Run all five evaluation phases against a loaded model.

    Args:
        model, tokenizer: loaded model (base or adapter-wrapped).
        test_split: raw HF Dataset with Question / Complex_CoT / Response fields.
        val_split_for_perplexity: raw HF Dataset for perplexity (chat-templated inside).
        eval_cfg: evaluation parameters.
        max_seq_length: for perplexity truncation.
        out_dir: where to save artifacts.
        tag: short label used for W&B metric prefix and file naming.
        skip_perplexity, skip_judge: shortcut flags for smoke tests.
        use_wandb, wandb_run: optional W&B integration.

    Returns:
        Summary dict suitable for JSON dumping.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Generate predictions -------------------------------------
    n_pred = min(eval_cfg.n_predictions, len(test_split))
    subset = test_split.select(range(n_pred))
    questions = [ex["Question"] for ex in subset]
    references = [ex["Response"] for ex in subset]
    log.info("eval_phase_1_generate", tag=tag, n=n_pred)

    generations = generate_predictions(model, tokenizer, questions, eval_cfg)
    save_predictions_jsonl(generations, references, out_dir / "predictions.jsonl")

    # --- Phase 2: Task metrics ---------------------------------------------
    pred_answers = [g.answer for g in generations]
    out_token_counts = [g.n_tokens_out for g in generations]
    task_metrics = compute_task_metrics(pred_answers, references, out_token_counts)
    log.info("eval_phase_2_task_metrics", tag=tag, **task_metrics.to_dict())

    # --- Phase 3: Perplexity (optional) -------------------------------------
    perplexity = float("nan")
    if not skip_perplexity and val_split_for_perplexity is not None:
        n_ppl = min(eval_cfg.perplexity_samples, len(val_split_for_perplexity))
        val_subset = val_split_for_perplexity.select(range(n_ppl))
        val_formatted = apply_chat_template_to_dataset(val_subset, tokenizer, num_proc=1)
        log.info("eval_phase_3_perplexity", tag=tag, n=n_ppl)
        perplexity = compute_perplexity(
            model, tokenizer,
            texts=[row["text"] for row in val_formatted],
            max_length=max_seq_length,
        )

    # --- Phase 4: LLM-judge (optional) --------------------------------------
    # Judge the FIRST N predictions sequentially — no random sampling, so the
    # qualitative markdown can pair judge_results[i] with generations[i] cleanly.
    judge_results: list[Any] = []
    judge_summary: dict[str, Any] = {}
    if not skip_judge:
        from gemma_medical.judge import GemmaJudge  # lazy import (GPU)

        n_judge = min(eval_cfg.n_judge_samples, len(generations))
        triples = [
            (generations[i].question, references[i], generations[i].raw_output)
            for i in range(n_judge)
        ]
        log.info("eval_phase_4_judge", tag=tag, n=n_judge)

        judge = GemmaJudge(model, tokenizer, eval_cfg)
        judge_results = judge.score_batch(triples)
        judge_summary = aggregate_judge_results(judge_results)
        log.info("eval_phase_4_judge_done", tag=tag, **judge_summary)

        with (out_dir / "judge_results.jsonl").open("w", encoding="utf-8") as f:
            for r in judge_results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # --- Phase 5: Save artifacts -------------------------------------------
    save_qualitative_md(
        generations,
        references,
        judge_results,
        n=eval_cfg.n_qualitative_samples,
        path=out_dir / "qualitative.md",
        title=f"Qualitative Samples — {tag}",
    )

    summary = {
        "tag": tag,
        "n_predictions": n_pred,
        "task_metrics": task_metrics.to_dict(),
        "perplexity": perplexity,
        "judge": judge_summary,
        "generation_params": {
            "max_new_tokens": eval_cfg.max_new_tokens,
            "do_sample": eval_cfg.do_sample,
            "batch_size": eval_cfg.batch_size,
        },
    }
    save_results_json(summary, out_dir / f"{tag}_results.json")

    # --- W&B logging -------------------------------------------------------
    if use_wandb and wandb_run is not None:
        try:
            import wandb  # type: ignore[import-not-found]
            wandb.log({
                f"{tag}/exact_match": task_metrics.exact_match,
                f"{tag}/contains_match": task_metrics.contains_match,
                f"{tag}/token_f1": task_metrics.token_f1,
                f"{tag}/mean_output_tokens": task_metrics.mean_output_tokens,
                f"{tag}/perplexity": perplexity,
                **{f"{tag}/judge_{k}": v
                   for k, v in judge_summary.items()
                   if isinstance(v, (int, float))},
            })
            artifact = wandb.Artifact(f"{tag}_outputs", type="evaluation")
            for fname in (f"{tag}_results.json", "qualitative.md",
                          "predictions.jsonl", "judge_results.jsonl"):
                fpath = out_dir / fname
                if fpath.exists():
                    artifact.add_file(str(fpath))
            wandb_run.log_artifact(artifact)
        except Exception as e:
            log.warning("wandb_log_failed", error=str(e))

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print a results summary."""
    tag = summary.get("tag", "eval")
    tm = summary["task_metrics"]
    jg = summary.get("judge", {})
    print("\n" + "=" * 60)
    print(f"RESULTS — {tag}")
    print("=" * 60)
    print(f"Predictions:       {summary['n_predictions']}")
    print(f"Exact match:       {tm['exact_match']:.4f}")
    print(f"Contains match:    {tm['contains_match']:.4f}")
    print(f"Token F1:          {tm['token_f1']:.4f}")
    print(f"Mean out tokens:   {tm['mean_output_tokens']:.1f}")
    print(f"Perplexity:        {summary['perplexity']:.2f}")
    if jg:
        agg = jg.get("aggregate", float("nan"))
        pr = jg.get("parse_rate", float("nan"))
        print(f"Judge aggregate:   {agg:.2f}" if isinstance(agg, (int, float)) else f"Judge aggregate:   N/A")
        print(f"Judge parse rate:  {pr:.2%}" if isinstance(pr, (int, float)) else f"Judge parse rate:  N/A")
    print("=" * 60)