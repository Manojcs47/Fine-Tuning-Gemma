"""Dataset loading, chat-templated formatting, and train/val/test splitting.

Dataset fields (per the assignment, §5):
  - Question      : the medical question
  - Complex_CoT   : chain-of-thought reasoning trace
  - Response      : final answer

Training target is `Complex_CoT` concatenated with `Response` under the
assistant turn of Gemma 4's chat template. The held-out test slice is the
LAST 500 examples; the validation slice is the 500 before that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from datasets import Dataset, load_dataset

from gemma_medical.config import DataConfig
from gemma_medical.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class _TokenizerLike(Protocol):
    """Minimal interface for a HF tokenizer that supports apply_chat_template."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
    ) -> str: ...


@dataclass(frozen=True)
class Splits:
    """The three slices we use downstream."""

    train: Dataset
    val: Dataset
    test: Dataset

    def summary(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_dataset(config: DataConfig) -> Dataset:
    """Fetch the medical-o1 dataset from the Hub.

    Note: as of mid-2026 this dataset requires a config-name argument
    (`en`, `zh`, `en_mix`, `zh_mix`). We pass `config.dataset_config`,
    which defaults to `en`. Setting it to None falls back to the old
    no-config behavior for other datasets.
    """
    log.info(
        "loading_dataset",
        name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.split,
    )
    if config.dataset_config is None:
        ds = load_dataset(config.dataset_name, split=config.split)
    else:
        ds = load_dataset(
            config.dataset_name, config.dataset_config, split=config.split
        )
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected Dataset, got {type(ds).__name__}")
    log.info("dataset_loaded", n_examples=len(ds), columns=ds.column_names)
    _validate_columns(ds)
    return ds


def _validate_columns(ds: Dataset) -> None:
    """Assert the dataset has the three fields we expect."""
    required = {"Question", "Complex_CoT", "Response"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Found: {ds.column_names}"
        )


# ---------------------------------------------------------------------------
# Formatting — chat template
# ---------------------------------------------------------------------------


def build_messages(question: str, cot: str, response: str) -> list[dict[str, str]]:
    """Return a two-turn chat: user question, assistant CoT + final answer.

    The assistant turn concatenates the reasoning trace and the final answer
    so the model learns to produce both in one go. The CoT is wrapped in
    <reasoning>...</reasoning> tags so it can be parsed back out at eval time
    if desired.
    """
    assistant_text = (
        f"<reasoning>\n{cot.strip()}\n</reasoning>\n\n"
        f"{response.strip()}"
    )
    return [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": assistant_text},
    ]


def format_example(example: dict[str, Any], tokenizer: _TokenizerLike) -> dict[str, str]:
    """Render one example to a single string via Gemma's chat template."""
    messages = build_messages(
        example["Question"], example["Complex_CoT"], example["Response"]
    )
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def format_for_inference(question: str, tokenizer: _TokenizerLike) -> str:
    """Render a question for inference (no assistant turn, add generation prompt)."""
    messages = [{"role": "user", "content": question.strip()}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def apply_chat_template_to_dataset(
    ds: Dataset, tokenizer: _TokenizerLike, num_proc: int = 2
) -> Dataset:
    """Map format_example across the dataset, keeping only the 'text' column."""
    log.info("applying_chat_template", n=len(ds))
    formatted = ds.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="chat-template",
    )
    return formatted


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def make_splits(ds: Dataset, config: DataConfig) -> Splits:
    """Carve train / val / test.

    Test = LAST `test_size` examples (held out exactly as in M1).
    Val  = `val_size` examples immediately before test.
    Train = everything before val.
    """
    n = len(ds)
    if n <= config.test_size + config.val_size:
        raise ValueError(
            f"Dataset too small for requested splits: {n} examples, "
            f"need >{config.test_size + config.val_size}"
        )

    test_start = n - config.test_size
    val_start = test_start - config.val_size

    train = ds.select(range(0, val_start))
    val = ds.select(range(val_start, test_start))
    test = ds.select(range(test_start, n))

    splits = Splits(train=train, val=val, test=test)
    log.info("splits_built", **splits.summary())
    return splits


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def prepare_datasets(
    config: DataConfig, tokenizer: _TokenizerLike
) -> Splits:
    """End-to-end: load → split → apply chat template to train+val only.

    The test split keeps its raw fields so eval can compare generations to
    the gold Response field directly.
    """
    raw = load_raw_dataset(config)
    raw_splits = make_splits(raw, config)

    train_fmt = apply_chat_template_to_dataset(raw_splits.train, tokenizer)
    val_fmt = apply_chat_template_to_dataset(raw_splits.val, tokenizer)

    return Splits(train=train_fmt, val=val_fmt, test=raw_splits.test)