"""Tests for data formatting (no actual dataset download — uses fixtures)."""
from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset

from gemma_medical.config import DataConfig
from gemma_medical.data import (
    Splits,
    build_messages,
    format_example,
    make_splits,
)


class FakeTokenizer:
    """Minimal tokenizer stub for testing chat-template formatting."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts: list[str] = []
        for msg in conversation:
            parts.append(f"<{msg['role']}>{msg['content']}</{msg['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


def test_build_messages_structure() -> None:
    msgs = build_messages("Q?", "thinking step 1", "Answer: 42")
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "Q?"}
    assert msgs[1]["role"] == "assistant"
    assert "<reasoning>" in msgs[1]["content"]
    assert "Answer: 42" in msgs[1]["content"]


def test_format_example_returns_text_field() -> None:
    tok = FakeTokenizer()
    example: dict[str, Any] = {
        "Question": "What is hypertension?",
        "Complex_CoT": "Step 1: define. Step 2: explain.",
        "Response": "Hypertension is high blood pressure.",
    }
    out = format_example(example, tok)
    assert set(out.keys()) == {"text", "prompt_text"}
    assert "What is hypertension?" in out["text"]
    assert "Step 1" in out["text"]


def test_make_splits_sizes_correct() -> None:
    ds = Dataset.from_dict({
        "Question": [f"q{i}" for i in range(2000)],
        "Complex_CoT": [f"c{i}" for i in range(2000)],
        "Response": [f"r{i}" for i in range(2000)],
    })
    cfg = DataConfig(test_size=500, val_size=500)
    splits = make_splits(ds, cfg)
    assert splits.summary() == {"train": 1000, "val": 500, "test": 500}


def test_make_splits_test_is_last_n() -> None:
    """Test slice must be the LAST 500 examples (M1 spec)."""
    ds = Dataset.from_dict({
        "Question": [f"q{i}" for i in range(2000)],
        "Complex_CoT": [f"c{i}" for i in range(2000)],
        "Response": [f"r{i}" for i in range(2000)],
    })
    cfg = DataConfig(test_size=500, val_size=500)
    splits = make_splits(ds, cfg)
    assert splits.test[0]["Question"] == "q1500"
    assert splits.test[-1]["Question"] == "q1999"


def test_make_splits_rejects_tiny_dataset() -> None:
    ds = Dataset.from_dict({
        "Question": ["q0", "q1"],
        "Complex_CoT": ["c0", "c1"],
        "Response": ["r0", "r1"],
    })
    cfg = DataConfig(test_size=500, val_size=500)
    with pytest.raises(ValueError):
        make_splits(ds, cfg)