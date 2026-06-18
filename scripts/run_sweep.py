"""Run an M3 hyperparameter sweep.

Workflow:
  - Load the sweep YAML (e.g. configs/sweep_m3.yaml).
  - For each run:
      * Materialize a per-run YAML (base + sweep_overrides + run.overrides).
      * subprocess.run scripts/run_train.py with that YAML.
      * Read the results JSON it produced.
  - Write outputs/m3_sweep/sweep_results.json + sweep_comparison.md.

Resumable: passing --resume skips runs whose results JSON already exists.
This survives Kaggle session timeouts (12h limit) — re-running picks up
where the previous session stopped.

Each run is a fresh subprocess for memory isolation: a bad config OOMing or
fragmenting the allocator cannot poison subsequent runs.

Usage:
  python scripts/run_sweep.py \\
      --sweep configs/sweep_m3.yaml \\
      --output-root /kaggle/working/outputs/m3_sweep

  # Resume after interrupt:
  python scripts/run_sweep.py \\
      --sweep configs/sweep_m3.yaml \\
      --output-root /kaggle/working/outputs/m3_sweep \\
      --resume

  # Only run specific entries:
  python scripts/run_sweep.py \\
      --sweep configs/sweep_m3.yaml \\
      --output-root /kaggle/working/outputs/m3_sweep \\
      --only m3-lr1e4-r16,m3-lr1e4-r16-warmup10
"""
from __future__ import annotations

# Set env vars BEFORE any import that could pull torch/unsloth into the parent
# process. The parent doesn't use the GPU, but we want subprocesses to inherit.
import os
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Make src/ importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemma_medical.logging_setup import configure_logging, get_logger  # noqa: E402
from gemma_medical.sweep import (  # noqa: E402
    SweepConfig,
    extract_run_summary,
    load_sweep_config,
    materialize_run_config,
    write_comparison_table,
)

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an M3 hyperparameter sweep")
    p.add_argument("--sweep", type=str, required=True, help="Path to sweep YAML")
    p.add_argument("--output-root", type=str, required=True,
                   help="Root dir for sweep outputs (per-run subdirs + aggregate)")
    p.add_argument("--resume", action="store_true",
                   help="Skip runs whose results JSON already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be executed; do not actually run")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable W&B for all runs in the sweep")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated run names to execute (all others skipped)")
    return p.parse_args()


def _per_run_paths(output_root: Path, run_name: str) -> dict[str, Path]:
    run_root = output_root / run_name
    return {
        "run_root": run_root,
        "tmp_config": run_root / "_config.yaml",
        "results_json": run_root / f"{run_name}_results.json",
    }


def _run_one(
    sweep_name: str,
    run_name: str,
    config_path: Path,
    output_dir: Path,
    use_wandb: bool,
) -> tuple[bool, float]:
    """Execute one run via subprocess. Returns (success_bool, wall_seconds).

    Stdout/stderr flow live to the parent (visible in the Kaggle cell output),
    so the user sees progress and can debug crashes without opening log files.
    """
    cmd = [
        sys.executable,
        "scripts/run_train.py",
        "--config", str(config_path),
        "--output-dir", str(output_dir),
        "--tags", f"m3,{sweep_name},{run_name}",
    ]
    if not use_wandb:
        cmd.append("--no-wandb")

    # Ensure subprocess inherits the env vars and the Kaggle secrets that the
    # notebook set in os.environ.
    env = os.environ.copy()
    env["UNSLOTH_RETURN_LOGITS"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log.info("subprocess_starting", run=run_name)
    print(f"\n{'=' * 70}\n[sweep] starting run: {run_name}\n{'=' * 70}\n", flush=True)
    t0 = time.time()

    try:
        result = subprocess.run(cmd, env=env, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        log.error("subprocess_launch_failed", run=run_name, error=str(e))
        return False, time.time() - t0

    elapsed = time.time() - t0
    if result.returncode != 0:
        log.warning("subprocess_nonzero_exit",
                    run=run_name, returncode=result.returncode,
                    elapsed_s=round(elapsed, 1))
        return False, elapsed

    log.info("subprocess_completed", run=run_name, elapsed_s=round(elapsed, 1))
    return True, elapsed


def _write_aggregate(
    output_root: Path,
    sweep_name: str,
    summaries: list[dict[str, Any]],
) -> None:
    """Write sweep_results.json + sweep_comparison.md to output_root."""
    json_path = output_root / "sweep_results.json"
    md_path = output_root / "sweep_comparison.md"
    output_root.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"sweep_name": sweep_name, "runs": summaries},
            f, indent=2, default=str,
        )
    write_comparison_table(summaries, md_path, sweep_name)


def _print_terminal_summary(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    print("\n" + "=" * 92)
    print(f"{'Run':<35} {'LR':>8} {'rank':>5} {'F1':>7} {'PPL':>7} {'Judge':>7}  Status")
    print("-" * 92)
    for s in summaries:
        lr = s.get("learning_rate")
        lr_s = f"{lr:.0e}" if isinstance(lr, (int, float)) else "—"
        rank = s.get("rank")
        rank_s = str(rank) if rank is not None else "—"
        f1 = s.get("token_f1")
        f1_s = f"{f1:.4f}" if isinstance(f1, (int, float)) else "—"
        ppl = s.get("perplexity")
        ppl_s = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else "—"
        ja = s.get("judge_aggregate")
        ja_s = f"{ja:.2f}" if isinstance(ja, (int, float)) else "—"
        print(f"{s['name']:<35} {lr_s:>8} {rank_s:>5} {f1_s:>7} {ppl_s:>7} {ja_s:>7}  {s.get('status', '?')}")
    print("=" * 92)


def main() -> int:
    args = parse_args()
    configure_logging(level="INFO")

    sweep: SweepConfig = load_sweep_config(args.sweep)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    only_set: set[str] | None = None
    if args.only:
        only_set = {n.strip() for n in args.only.split(",") if n.strip()}
        log.info("sweep_filtered", only=sorted(only_set))

    log.info("sweep_starting",
             sweep_name=sweep.sweep_name,
             n_runs=len(sweep.runs),
             output_root=str(output_root),
             resume=args.resume,
             dry_run=args.dry_run)

    summaries: list[dict[str, Any]] = []
    n_succeeded = 0
    n_failed = 0
    n_skipped = 0
    total_elapsed = 0.0

    for i, run in enumerate(sweep.runs, start=1):
        if only_set is not None and run.name not in only_set:
            log.info("run_skipped_filter", run=run.name)
            continue

        log.info("run_index", index=i, total=len(sweep.runs), run=run.name,
                 description=run.description)

        paths = _per_run_paths(output_root, run.name)
        paths["run_root"].mkdir(parents=True, exist_ok=True)

        # Materialize per-run config
        try:
            cfg_dict = materialize_run_config(sweep, run, paths["tmp_config"])
        except (FileNotFoundError, ValueError) as e:
            log.error("config_materialize_failed", run=run.name, error=str(e))
            summaries.append({
                "name": run.name, "status": "failed",
                "error": f"config materialize: {e}",
            })
            n_failed += 1
            _write_aggregate(output_root, sweep.sweep_name, summaries)
            continue

        # Resume: skip if results already exist
        if args.resume and paths["results_json"].exists():
            log.info("run_already_done_skipping", run=run.name,
                     results_path=str(paths["results_json"]))
            summaries.append(
                extract_run_summary(run.name, cfg_dict, paths["results_json"])
            )
            n_skipped += 1
            _write_aggregate(output_root, sweep.sweep_name, summaries)
            continue

        if args.dry_run:
            log.info("dry_run_would_execute", run=run.name,
                     config_path=str(paths["tmp_config"]),
                     output_dir=str(paths["run_root"]))
            continue

        # Execute the run
        ok, elapsed = _run_one(
            sweep_name=sweep.sweep_name,
            run_name=run.name,
            config_path=paths["tmp_config"],
            output_dir=paths["run_root"],
            use_wandb=not args.no_wandb,
        )
        total_elapsed += elapsed
        n_succeeded += int(ok)
        n_failed += int(not ok)

        # Build summary (works whether the run wrote results or crashed).
        summaries.append(
            extract_run_summary(run.name, cfg_dict, paths["results_json"])
        )

        # Persist aggregate after every run so partial sweeps are useful.
        _write_aggregate(output_root, sweep.sweep_name, summaries)

    log.info("sweep_complete",
             n_runs=len(sweep.runs),
             n_succeeded=n_succeeded,
             n_failed=n_failed,
             n_skipped=n_skipped,
             total_elapsed_s=round(total_elapsed, 1),
             total_elapsed_min=round(total_elapsed / 60, 1))

    _write_aggregate(output_root, sweep.sweep_name, summaries)
    _print_terminal_summary(summaries)

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())