"""Quantitative metrics for medical reasoning predictions.

Three layers:
  - Exact-match accuracy: strict normalized match on the final answer.
  - Token F1: bag-of-tokens overlap. Smoother signal than exact-match.
  - Perplexity: exp(eval_loss) on a held-out slice. Training-dynamics signal.
"""
from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass
from typing import Any

from gemma_medical.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(s: str) -> str:
    """Standard QA normalization: lowercase, strip punct, drop articles, collapse ws."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Per-example metrics
# ---------------------------------------------------------------------------


def exact_match(prediction: str, reference: str) -> bool:
    """Strict normalized exact match."""
    return normalize_text(prediction) == normalize_text(reference)


def contains_match(prediction: str, reference: str) -> bool:
    """Looser: does normalized prediction contain normalized reference?"""
    return normalize_text(reference) in normalize_text(prediction)


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 between prediction and reference."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t, c in pred_counts.items():
        if t in ref_counts:
            common[t] = min(c, ref_counts[t])

    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskMetrics:
    """Aggregate task-level metrics over a prediction set."""

    n: int
    exact_match: float       # fraction in [0, 1]
    contains_match: float    # fraction in [0, 1]
    token_f1: float          # mean
    mean_output_tokens: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_task_metrics(
    predictions: list[str],
    references: list[str],
    output_token_counts: list[int] | None = None,
) -> TaskMetrics:
    """Compute exact-match, contains, and token-F1 over a prediction set."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) != references ({len(references)})"
        )

    n = len(predictions)
    if n == 0:
        return TaskMetrics(n=0, exact_match=0.0, contains_match=0.0,
                            token_f1=0.0, mean_output_tokens=0.0)

    em = sum(exact_match(p, r) for p, r in zip(predictions, references, strict=True))
    cm = sum(contains_match(p, r) for p, r in zip(predictions, references, strict=True))
    f1 = sum(token_f1(p, r) for p, r in zip(predictions, references, strict=True))

    mean_out = (
        sum(output_token_counts) / len(output_token_counts)
        if output_token_counts
        else 0.0
    )

    metrics = TaskMetrics(
        n=n,
        exact_match=em / n,
        contains_match=cm / n,
        token_f1=f1 / n,
        mean_output_tokens=mean_out,
    )
    log.info("task_metrics_computed", **metrics.to_dict())
    return metrics


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 2048,
) -> float:
    """Compute perplexity = exp(mean cross-entropy) over a list of formatted texts.

    Each text should already be chat-templated (i.e. what the model would see
    during training). We forward-pass with labels=input_ids and average the
    per-example losses.
    """
    import torch  # type: ignore[import-not-found]

    if not texts:
        return float("nan")

    model.eval()
    losses: list[float] = []
    for i, text in enumerate(texts):
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(model.device)
        input_ids = enc["input_ids"]

        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)

        losses.append(float(out.loss.item()))

        if (i + 1) % 25 == 0:
            log.info("perplexity_progress", done=i + 1, total=len(texts))

    mean_loss = sum(losses) / len(losses)
    ppl = float(torch.exp(torch.tensor(mean_loss)).item())
    log.info("perplexity_computed", mean_loss=mean_loss, perplexity=ppl, n=len(texts))
    return ppl