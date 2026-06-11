"""Tests for metric functions and judge output parsing. No GPU required."""
from __future__ import annotations

from gemma_medical.evaluate import (
    compute_task_metrics,
    contains_match,
    exact_match,
    normalize_text,
    token_f1,
)
from gemma_medical.judge import parse_judge_output


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_lowercases() -> None:
    assert normalize_text("Hello WORLD") == "hello world"


def test_normalize_drops_articles() -> None:
    assert normalize_text("The cat sat on a mat") == "cat sat on mat"


def test_normalize_strips_punctuation() -> None:
    assert normalize_text("Hello, world!") == "hello world"


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------


def test_exact_match_true_on_normalization() -> None:
    assert exact_match("The Answer.", "answer")
    assert exact_match("hypertension", "Hypertension!")


def test_exact_match_false_on_diff() -> None:
    assert not exact_match("hypertension", "hypotension")


def test_contains_match() -> None:
    assert contains_match("The answer is hypertension and diabetes.", "hypertension")
    assert not contains_match("The answer is hypotension.", "hypertension")


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------


def test_token_f1_identical() -> None:
    assert token_f1("acute pancreatitis", "acute pancreatitis") == 1.0


def test_token_f1_disjoint() -> None:
    assert token_f1("apple", "banana") == 0.0


def test_token_f1_partial() -> None:
    # 1 common token out of 2 each → P=0.5, R=0.5, F1=0.5
    assert abs(token_f1("acute pancreatitis", "acute appendicitis") - 0.5) < 1e-6


def test_token_f1_empty() -> None:
    assert token_f1("", "something") == 0.0
    assert token_f1("something", "") == 0.0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def test_compute_task_metrics() -> None:
    preds = ["acute pancreatitis", "diabetes mellitus", "wrong answer"]
    refs = ["acute pancreatitis", "diabetes", "hypertension"]
    m = compute_task_metrics(preds, refs)
    assert m.n == 3
    assert m.exact_match == 1 / 3  # one exact
    assert m.contains_match == 2 / 3  # exact + diabetes-in-diabetes-mellitus


# ---------------------------------------------------------------------------
# Judge parsing
# ---------------------------------------------------------------------------


def test_parse_judge_clean_json() -> None:
    raw = '{"conclusion_correctness": 4, "reasoning_validity": 3, "no_fabrication": 5, "justification": "good"}'
    r = parse_judge_output(raw)
    assert r.parse_ok
    assert r.conclusion_correctness == 4.0
    assert r.aggregate == 4.0


def test_parse_judge_with_code_fence() -> None:
    raw = '```json\n{"conclusion_correctness": 5, "reasoning_validity": 5, "no_fabrication": 5, "justification": "ok"}\n```'
    r = parse_judge_output(raw)
    assert r.parse_ok
    assert r.aggregate == 5.0


def test_parse_judge_with_preamble() -> None:
    raw = 'Here is my evaluation:\n{"conclusion_correctness": 2, "reasoning_validity": 2, "no_fabrication": 3, "justification": "weak"}'
    r = parse_judge_output(raw)
    assert r.parse_ok
    assert r.conclusion_correctness == 2.0


def test_parse_judge_clamps_out_of_range() -> None:
    raw = '{"conclusion_correctness": 10, "reasoning_validity": -5, "no_fabrication": 3, "justification": "x"}'
    r = parse_judge_output(raw)
    assert r.parse_ok
    assert r.conclusion_correctness == 5.0  # clamped
    assert r.reasoning_validity == 1.0       # clamped


def test_parse_judge_garbage() -> None:
    r = parse_judge_output("This is not JSON at all")
    assert not r.parse_ok
    assert r.conclusion_correctness == 0.0