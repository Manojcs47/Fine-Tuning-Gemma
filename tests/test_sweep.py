"""Tests for gemma_medical.sweep — config merging + result extraction.

These tests have zero GPU dependencies and should run on a laptop.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gemma_medical.sweep import (
    SweepConfig,
    SweepRun,
    deep_merge,
    extract_run_summary,
    load_sweep_config,
    materialize_run_config,
    write_comparison_table,
)


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_simple() -> None:
    base = {"a": 1, "b": 2}
    override = {"b": 20, "c": 3}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 20, "c": 3}
    # base not mutated
    assert base == {"a": 1, "b": 2}


def test_deep_merge_nested() -> None:
    base = {"training": {"lr": 1e-4, "wd": 0.01}, "model": {"size": "small"}}
    override = {"training": {"lr": 5e-5}}
    result = deep_merge(base, override)
    assert result == {
        "training": {"lr": 5e-5, "wd": 0.01},
        "model": {"size": "small"},
    }


def test_deep_merge_replaces_list_not_extends() -> None:
    # Lists are replaced wholesale (not concatenated). This matters for
    # target_modules: overriding to ["q_proj"] must produce ["q_proj"],
    # not the union with the default 7-module list.
    base = {"lora": {"target_modules": ["a", "b", "c"]}}
    override = {"lora": {"target_modules": ["q_proj"]}}
    result = deep_merge(base, override)
    assert result == {"lora": {"target_modules": ["q_proj"]}}


def test_deep_merge_empty_override() -> None:
    base = {"a": {"b": 1}}
    assert deep_merge(base, {}) == {"a": {"b": 1}}


def test_deep_merge_dict_replaces_nondict() -> None:
    base = {"a": 1}
    override = {"a": {"nested": 2}}
    assert deep_merge(base, override) == {"a": {"nested": 2}}


def test_deep_merge_nondict_replaces_dict() -> None:
    base = {"a": {"nested": 1}}
    override = {"a": 5}
    assert deep_merge(base, override) == {"a": 5}


# ---------------------------------------------------------------------------
# Sweep config loading
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f)


def test_load_sweep_config_minimal(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.yaml"
    _write_yaml(sweep_path, {
        "sweep_name": "test-sweep",
        "base_config_path": "configs/lora_default.yaml",
        "runs": [{"name": "r1", "overrides": {}}],
    })
    sweep = load_sweep_config(sweep_path)
    assert sweep.sweep_name == "test-sweep"
    assert len(sweep.runs) == 1
    assert sweep.runs[0].name == "r1"


def test_load_sweep_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sweep_config(tmp_path / "nope.yaml")


def test_load_sweep_config_empty_runs_list_rejected(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.yaml"
    _write_yaml(sweep_path, {
        "sweep_name": "t",
        "base_config_path": "x.yaml",
        "runs": [],
    })
    with pytest.raises(Exception):  # pydantic ValidationError
        load_sweep_config(sweep_path)


def test_load_sweep_config_empty_file(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("")
    with pytest.raises(ValueError):
        load_sweep_config(sweep_path)


# ---------------------------------------------------------------------------
# materialize_run_config
# ---------------------------------------------------------------------------


def test_materialize_run_config_merges_correctly(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, {
        "name": "base",
        "technique": "lora",
        "training": {"learning_rate": 2.0e-4, "warmup_ratio": 0.05},
        "lora": {"r": 16, "lora_alpha": 16},
    })
    sweep = SweepConfig(
        sweep_name="s",
        base_config_path=str(base_path),
        sweep_overrides={"training": {"num_train_epochs": 0.5}},
        runs=[SweepRun(
            name="r1",
            overrides={"training": {"learning_rate": 5.0e-5}},
        )],
    )
    out_path = tmp_path / "r1_config.yaml"
    cfg = materialize_run_config(sweep, sweep.runs[0], out_path)

    assert cfg["name"] == "r1"  # name overridden
    assert cfg["training"]["learning_rate"] == 5.0e-5  # run override wins
    assert cfg["training"]["num_train_epochs"] == 0.5  # sweep override applied
    assert cfg["training"]["warmup_ratio"] == 0.05  # base preserved
    assert cfg["lora"]["r"] == 16  # base preserved
    assert out_path.exists()
    with out_path.open() as f:
        from_disk = yaml.safe_load(f)
    assert from_disk == cfg


def test_materialize_run_config_run_overrides_beat_sweep(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, {"training": {"learning_rate": 1e-3}})
    sweep = SweepConfig(
        sweep_name="s",
        base_config_path=str(base_path),
        sweep_overrides={"training": {"learning_rate": 1e-4}},
        runs=[SweepRun(
            name="r1",
            overrides={"training": {"learning_rate": 5e-5}},
        )],
    )
    out_path = tmp_path / "r1.yaml"
    cfg = materialize_run_config(sweep, sweep.runs[0], out_path)
    assert cfg["training"]["learning_rate"] == 5e-5


def test_materialize_run_config_missing_base_raises(tmp_path: Path) -> None:
    sweep = SweepConfig(
        sweep_name="s",
        base_config_path=str(tmp_path / "does_not_exist.yaml"),
        runs=[SweepRun(name="r1", overrides={})],
    )
    with pytest.raises(FileNotFoundError):
        materialize_run_config(sweep, sweep.runs[0], tmp_path / "out.yaml")


# ---------------------------------------------------------------------------
# extract_run_summary
# ---------------------------------------------------------------------------


def test_extract_run_summary_missing_results(tmp_path: Path) -> None:
    cfg = {"training": {"learning_rate": 1e-4}, "lora": {"r": 16}}
    s = extract_run_summary("r1", cfg, tmp_path / "missing.json")
    assert s["name"] == "r1"
    assert s["status"] == "failed"
    assert "error" in s
    assert s["learning_rate"] == 1e-4  # config-derived fields still populated
    assert s["rank"] == 16


def test_extract_run_summary_flat_layout(tmp_path: Path) -> None:
    results_path = tmp_path / "r1_results.json"
    with results_path.open("w") as f:
        json.dump({
            "exact_match": 0.0,
            "contains_match": 0.02,
            "token_f1": 0.23,
            "mean_output_tokens": 429.62,
            "perplexity": 9.64,
            "judge_aggregate": 3.77,
            "judge_conclusion_correctness": 3.12,
        }, f)
    cfg = {
        "training": {"learning_rate": 1e-4, "warmup_ratio": 0.05},
        "lora": {"r": 16, "lora_alpha": 16, "target_modules": ["q_proj"]},
    }
    s = extract_run_summary("r1", cfg, results_path)
    assert s["status"] == "ok"
    assert s["token_f1"] == pytest.approx(0.23)
    assert s["judge_aggregate"] == pytest.approx(3.77)
    assert s["judge_correctness"] == pytest.approx(3.12)
    assert s["perplexity"] == pytest.approx(9.64)


def test_extract_run_summary_nested_layout(tmp_path: Path) -> None:
    results_path = tmp_path / "r1_results.json"
    with results_path.open("w") as f:
        json.dump({
            "task": {
                "exact_match": 0.0, "token_f1": 0.5,
                "contains_match": 0.1, "mean_output_tokens": 100,
            },
            "perplexity": {"perplexity": 8.0, "mean_loss": 2.08},
            "judge": {
                "aggregate": 4.0,
                "conclusion_correctness": 3.5,
                "reasoning_validity": 4.2,
                "no_fabrication": 4.8,
            },
        }, f)
    cfg = {"training": {"learning_rate": 1e-4}, "lora": {"r": 16}}
    s = extract_run_summary("r1", cfg, results_path)
    assert s["status"] == "ok"
    assert s["token_f1"] == pytest.approx(0.5)
    assert s["judge_aggregate"] == pytest.approx(4.0)
    assert s["perplexity"] == pytest.approx(8.0)
    assert s["judge_correctness"] == pytest.approx(3.5)


def test_extract_run_summary_malformed_json(tmp_path: Path) -> None:
    results_path = tmp_path / "r1_results.json"
    results_path.write_text("{not valid json")
    cfg = {"training": {"learning_rate": 1e-4}, "lora": {"r": 16}}
    s = extract_run_summary("r1", cfg, results_path)
    assert s["status"] == "failed"
    assert "error" in s


# ---------------------------------------------------------------------------
# write_comparison_table
# ---------------------------------------------------------------------------


def test_write_comparison_table_creates_file(tmp_path: Path) -> None:
    summaries = [
        {
            "name": "r1", "learning_rate": 1e-4, "rank": 16, "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "warmup_ratio": 0.05, "token_f1": 0.23, "perplexity": 9.64,
            "judge_aggregate": 3.77, "status": "ok",
        },
        {
            "name": "r2", "learning_rate": 5e-5, "rank": 8, "lora_alpha": 8,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "warmup_ratio": 0.05, "token_f1": 0.25, "perplexity": 10.0,
            "judge_aggregate": 3.9, "status": "ok",
        },
    ]
    out = tmp_path / "table.md"
    write_comparison_table(summaries, out, "test-sweep")
    assert out.exists()
    text = out.read_text()
    assert "test-sweep" in text
    assert "r1" in text and "r2" in text
    assert "attn" in text  # 4-element targets abbreviated
    assert "all" in text   # 7-element targets abbreviated
    # Best lines present (r2 has higher F1, higher judge, higher PPL)
    assert "Best Token F1" in text
    assert "r2" in text


def test_write_comparison_table_handles_failed_runs(tmp_path: Path) -> None:
    summaries = [
        {"name": "ok_run", "learning_rate": 1e-4, "rank": 16,
         "token_f1": 0.3, "perplexity": 8.0, "judge_aggregate": 4.0,
         "status": "ok"},
        {"name": "bad_run", "learning_rate": 1e-3, "rank": 16,
         "status": "failed", "error": "OOM"},
    ]
    out = tmp_path / "table.md"
    write_comparison_table(summaries, out, "mixed-sweep")
    text = out.read_text()
    assert "ok_run" in text
    assert "bad_run" in text
    assert "failed" in text
    assert "ok: 1" in text and "failed: 1" in text