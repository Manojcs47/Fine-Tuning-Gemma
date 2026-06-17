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
    """Render one example via Gemma's chat template.

    Returns both the full text (prompt + assistant turn) and the prompt-only
    text (user turn + generation prompt). The prompt is used by
    pre_tokenize_for_training to mask prompt tokens out of the labels so loss
    is computed on the assistant response only (completion-only SFT). Without
    that masking the model is trained to predict the question too, whose loss
    is largely irreducible — the training loss then plateaus and gradients
    vanish once the response format is learned.
    """
    messages = build_messages(
        example["Question"], example["Complex_CoT"], example["Response"]
    )
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Prompt prefix = user turn + the "<start_of_turn>model" generation prompt.
    # This is exactly the prefix of `text` that precedes the assistant content.
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["Question"].strip()}],
        tokenize=False, add_generation_prompt=True,
    )
    return {"text": text, "prompt_text": prompt_text}


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
    
def pre_tokenize_for_training(
    ds: Dataset,
    tokenizer: Any,
    max_length: int = 2048,
) -> Dataset:
    """Tokenize a chat-templated dataset in-process so SFTTrainer skips its own
    (broken) multiprocess tokenization step.

    Why this exists
    ----------------
    Unsloth's FastModel returns a `Gemma4Processor` that has been monkey-patched
    to reference `torch._dynamo.config` internally. `dill` cannot pickle that
    config object, so any code path that tries to pickle the tokenizer dies with:
        TypeError: cannot pickle 'ConfigModuleInstance' object

    TRL's `SFTTrainer._prepare_dataset` calls
        dataset.map(tokenize_fn, num_proc=args.dataset_num_proc, ...)
    for tokenization. Even with `dataset_num_proc=1` the call chain still
    spins up a `multiprocess.Pool` in TRL 0.20.x + datasets 3.x, which forces
    the pickle attempt. Setting it to 1 or None in SFTConfig is therefore NOT
    sufficient on its own.

    The fix is to never let TRL tokenize. We do it ourselves with `num_proc=1`
    (no multiprocess at all), produce a dataset that already has `input_ids`,
    `attention_mask`, and `labels`, and pass that to SFTTrainer. TRL detects
    `is_processed=True` from the column names and skips the tokenization map
    entirely.

    Args:
        ds: dataset with a `text` column (output of apply_chat_template_to_dataset).
        tokenizer: the Gemma4Processor / HF tokenizer from Unsloth's FastModel.
        max_length: truncation length matching cfg.model.max_seq_length.

    Returns:
        Dataset with `input_ids`, `attention_mask`, and `labels` columns.
        Original columns are removed.
    """
    log.info("pre_tokenizing", n=len(ds), max_length=max_length)

    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        # Pass text= as keyword: Gemma4Processor.__call__ binds positional args
        # to images=, not text=. add_special_tokens=False because Gemma's chat
        # template already emits <bos> at the start.
        enc = tokenizer(
            text=batch["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        # Completion-only labels: copy input_ids, then mask the prompt prefix to
        # -100 so loss is computed only on the assistant response. `prompt_text`
        # (from format_example) is exactly the prefix of `text`, so tokenizing
        # it with the same options gives the prompt length to mask.
        if "prompt_text" in batch:
            prompt_enc = tokenizer(
                text=batch["prompt_text"],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            labels: list[list[int]] = []
            for ids, p_ids in zip(enc["input_ids"], prompt_enc["input_ids"]):
                lab = list(ids)
                p_len = len(p_ids)
                # Only mask when there is at least one response token left after
                # the prompt. If truncation left no completion (p_len >= len),
                # keep full labels so the row still contributes a loss and we
                # never emit an all-(-100) row (which would make the batch loss
                # NaN).
                if 0 < p_len < len(lab):
                    for i in range(p_len):
                        lab[i] = -100
                labels.append(lab)
            enc["labels"] = labels
        else:
            # Fallback (e.g. a dataset without prompt_text): train on all tokens.
            enc["labels"] = [list(ids) for ids in enc["input_ids"]]
        return enc

    return ds.map(
        tokenize_fn,
        batched=True,
        batch_size=64,
        num_proc=1,            # MUST be 1 — see docstring
        remove_columns=ds.column_names,
        desc="pre-tokenizing",
    )