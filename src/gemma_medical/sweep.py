"""M3 sweep config schema, run materialization, and result aggregation.

Workflow (see scripts/run_sweep.py for the orchestrator that calls this):
  1. Load a SweepConfig from YAML (e.g. configs/sweep_m3.yaml).
  2. For each SweepRun:
     - deep-merge base config + sweep-wide overrides + per-run overrides
     - write the merged config to a temp YAML
     - subprocess.run scripts/run_train.py with that YAML
     - parse the resulting per-run results JSON
  3. Aggregate into sweep_results.json + sweep_comparison.md.

This module deliberately has no torch/unsloth imports — it can be unit-tested
on a laptop without GPU dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------


class SweepRun(BaseModel):
    """One run in a sweep — overrides applied on top of base + sweep-wide overrides."""

    name: str = Field(
        description="Unique run name; used as cfg.name, W&B run name, and output subdir",
    )
    description: str = Field(default="", description="One-line summary of this run")
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Nested override dict applied on top of base + sweep_overrides. "
            "Example: {'training': {'learning_rate': 5e-5}, 'lora': {'r': 8}}"
        ),
    )


class SweepConfig(BaseModel):
    """A sweep: a base config + sweep-wide overrides + a list of per-run overrides."""

    sweep_name: str = Field(description="Group name for all runs in this sweep")
    base_config_path: str = Field(
        description="Path to base ExperimentConfig YAML (e.g. configs/lora_default.yaml)"
    )
    description: str = Field(default="")
    sweep_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Applied to every run (e.g. trimmed eval sizes for sweep mode)",
    )
    runs: list[SweepRun] = Field(min_length=1)


def load_sweep_config(path: str | Path) -> SweepConfig:
    """Load and validate a sweep config from YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sweep config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Sweep config is empty: {p}")
    return SweepConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Deep merge (used by materialize_run_config)
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`. Returns a new dict; inputs not mutated.

    Rules:
      - For dict values: recurse.
      - For everything else (incl. lists): override REPLACES base wholesale.
        This is intentional — overriding target_modules to ['q_proj'] must
        produce ['q_proj'], not extend the existing list.
    """
    # Shallow-copy base; inner dicts are deep-copied via recursion below.
    result: dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Per-run config materialization
# ---------------------------------------------------------------------------


def materialize_run_config(
    sweep: SweepConfig,
    run: SweepRun,
    output_path: Path,
) -> dict[str, Any]:
    """Compute the final ExperimentConfig dict for one run and write it to YAML.

    Override order (later wins):
        base_config (from sweep.base_config_path)
        + sweep.sweep_overrides
        + run.overrides
        + name=run.name (always last)

    Returns the materialized dict (also written to `output_path`).
    """
    base_path = Path(sweep.base_config_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with base_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    if base is None:
        raise ValueError(f"Base config is empty: {base_path}")

    cfg = deep_merge(base, sweep.sweep_overrides)
    cfg = deep_merge(cfg, run.overrides)
    cfg["name"] = run.name  # always wins

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


def _try_get(d: dict[str, Any], *paths: str) -> Any:
    """Try several dotted paths; return the first one that resolves, else None.

    Earlier paths take precedence. Use this to be robust against minor schema
    differences in the results JSON (flat vs nested layouts).
    """
    for path in paths:
        keys = path.split(".")
        cur: Any = d
        try:
            for k in keys:
                cur = cur[k]
        except (KeyError, TypeError):
            continue
        return cur
    return None


def extract_run_summary(
    run_name: str,
    cfg_dict: dict[str, Any],
    results_path: Path,
) -> dict[str, Any]:
    """Build a summary dict for one sweep run from its results JSON + materialized config.

    Tolerates missing files, malformed JSON, and either flat-key or nested-key
    result layouts. Always returns a dict with the same shape, even on failure.
    """
    summary: dict[str, Any] = {
        "name": run_name,
        "learning_rate": _try_get(cfg_dict, "training.learning_rate"),
        "rank": _try_get(cfg_dict, "lora.r"),
        "lora_alpha": _try_get(cfg_dict, "lora.lora_alpha"),
        "target_modules": _try_get(cfg_dict, "lora.target_modules"),
        "warmup_ratio": _try_get(cfg_dict, "training.warmup_ratio"),
        "status": "unknown",
    }

    if not results_path.exists():
        summary["status"] = "failed"
        summary["error"] = "results file not found"
        return summary

    try:
        with results_path.open("r", encoding="utf-8") as f:
            r = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        summary["status"] = "failed"
        summary["error"] = f"results read failed: {e}"
        return summary

    summary["status"] = "ok"

    # Metric extraction tolerates two layouts: nested under {"task":..., "judge":...}
    # or flat at the top level. Nested paths tried first.
    summary["exact_match"] = _try_get(
        r, "task.exact_match", "task_metrics.exact_match", "exact_match",
    )
    summary["contains_match"] = _try_get(
        r, "task.contains_match", "task_metrics.contains_match", "contains_match",
    )
    summary["token_f1"] = _try_get(
        r, "task.token_f1", "task_metrics.token_f1", "token_f1",
    )
    summary["mean_output_tokens"] = _try_get(
        r, "task.mean_output_tokens", "task_metrics.mean_output_tokens",
        "mean_output_tokens",
    )
    summary["perplexity"] = _try_get(r, "perplexity.perplexity", "perplexity")
    summary["judge_aggregate"] = _try_get(r, "judge.aggregate", "judge_aggregate")
    summary["judge_correctness"] = _try_get(
        r, "judge.conclusion_correctness", "judge_conclusion_correctness",
        "judge_correctness",
    )
    summary["judge_reasoning"] = _try_get(
        r, "judge.reasoning_validity", "judge_reasoning_validity", "judge_reasoning",
    )
    summary["judge_fabrication"] = _try_get(
        r, "judge.no_fabrication", "judge_no_fabrication", "judge_fabrication",
    )
    return summary


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _fmt(v: Any, precision: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def _fmt_targets(targets: list[str] | None) -> str:
    if not targets:
        return "—"
    if len(targets) == 7:
        return "all"
    if len(targets) == 4 and set(targets) == {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attn"
    return f"{len(targets)} mods"


def _fmt_lr(v: Any) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:.0e}"


def write_comparison_table(
    summaries: list[dict[str, Any]],
    output_path: Path,
    sweep_name: str,
) -> None:
    """Write a markdown comparison table to `output_path`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# Sweep results — {sweep_name}",
        "",
        f"Total runs: {len(summaries)} "
        f"(ok: {sum(1 for s in summaries if s.get('status') == 'ok')}, "
        f"failed: {sum(1 for s in summaries if s.get('status') == 'failed')})",
        "",
        "| Run | LR | rank | α | targets | warmup | Token F1 | PPL | Judge | Status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['name']} "
            f"| {_fmt_lr(s.get('learning_rate'))} "
            f"| {_fmt(s.get('rank'))} "
            f"| {_fmt(s.get('lora_alpha'))} "
            f"| {_fmt_targets(s.get('target_modules'))} "
            f"| {_fmt(s.get('warmup_ratio'))} "
            f"| {_fmt(s.get('token_f1'))} "
            f"| {_fmt(s.get('perplexity'), 2)} "
            f"| {_fmt(s.get('judge_aggregate'), 2)} "
            f"| {s.get('status', '?')} |"
        )

    # "Best" lines — only consider successful runs.
    ok = [s for s in summaries if s.get("status") == "ok"]
    if ok:
        lines.append("")
        lines.append("## Best by metric")
        lines.append("")
        best_f1 = max(
            (s for s in ok if isinstance(s.get("token_f1"), (int, float))),
            key=lambda s: s["token_f1"], default=None,
        )
        best_judge = max(
            (s for s in ok if isinstance(s.get("judge_aggregate"), (int, float))),
            key=lambda s: s["judge_aggregate"], default=None,
        )
        best_ppl = min(
            (s for s in ok if isinstance(s.get("perplexity"), (int, float))),
            key=lambda s: s["perplexity"], default=None,
        )
        if best_f1 is not None:
            lines.append(f"- **Best Token F1**: `{best_f1['name']}` "
                         f"({best_f1['token_f1']:.4f})")
        if best_judge is not None:
            lines.append(f"- **Best Judge aggregate**: `{best_judge['name']}` "
                         f"({best_judge['judge_aggregate']:.2f})")
        if best_ppl is not None:
            lines.append(f"- **Lowest perplexity**: `{best_ppl['name']}` "
                         f"({best_ppl['perplexity']:.2f})")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")