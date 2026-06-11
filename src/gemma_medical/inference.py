"""Model loading for inference + batched generation + response parsing.

This module is GPU-only — it imports unsloth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from gemma_medical.config import EvaluationConfig, ModelConfig
from gemma_medical.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_model_for_inference(
    config: ModelConfig, adapter_path: str | None = None
) -> tuple[Any, Any]:
    """Load a model for inference (no LoRA attachment).

    Args:
        config: ModelConfig describing the base model.
        adapter_path: optional path to a trained LoRA adapter to load on top.

    Returns:
        (model, tokenizer)
    """
    from unsloth import FastModel  # type: ignore[import-not-found]

    log.info(
        "loading_for_inference",
        base_model=config.base_model,
        load_in_4bit=config.load_in_4bit,
        adapter_path=adapter_path,
    )

    model, tokenizer = FastModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        full_finetuning=False,
    )

    if adapter_path:
        log.info("loading_adapter", path=adapter_path)
        model.load_adapter(adapter_path)

    # Switch to inference mode (Unsloth: 2x faster)
    FastModel.for_inference(model)

    # Generation needs left padding for decoder-only models
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationOutput:
    """One generation result."""

    question: str
    raw_output: str        # full decoded model output
    reasoning: str         # extracted reasoning trace (may be empty)
    answer: str            # extracted final answer
    n_tokens_out: int      # number of generated tokens


def _build_inference_prompt(question: str, tokenizer: Any) -> str:
    """Render a question through the chat template with the assistant turn open."""
    messages = [{"role": "user", "content": question.strip()}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def _generate_one_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    config: EvaluationConfig,
) -> list[str]:
    """Generate completions for a batch of prompts. Returns the new text only."""
    import torch  # type: ignore[import-not-found]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.do_sample:
        gen_kwargs.update(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[:, input_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def generate_predictions(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    config: EvaluationConfig,
) -> list[GenerationOutput]:
    """Generate predictions for a list of questions, batched."""
    log.info(
        "generating_predictions",
        n=len(questions),
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
    )

    results: list[GenerationOutput] = []
    for i in range(0, len(questions), config.batch_size):
        batch_qs = questions[i : i + config.batch_size]
        prompts = [_build_inference_prompt(q, tokenizer) for q in batch_qs]

        try:
            outputs = _generate_one_batch(model, tokenizer, prompts, config)
        except Exception as e:
            log.error("batch_generation_failed", batch_start=i, error=str(e))
            outputs = ["" for _ in batch_qs]

        for q, raw in zip(batch_qs, outputs, strict=True):
            reasoning, answer = parse_response(raw)
            n_tokens = len(tokenizer.encode(raw, add_special_tokens=False))
            results.append(
                GenerationOutput(
                    question=q,
                    raw_output=raw,
                    reasoning=reasoning,
                    answer=answer,
                    n_tokens_out=n_tokens,
                )
            )

        if (i // config.batch_size) % 10 == 0:
            log.info("generation_progress", done=i + len(batch_qs), total=len(questions))

    log.info("generation_complete", n=len(results))
    return results


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_REASONING_PATTERN = re.compile(
    r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE
)


def parse_response(raw_output: str) -> tuple[str, str]:
    """Extract (reasoning, answer) from a model's raw output.

    The fine-tuned model is trained to emit <reasoning>...</reasoning> followed
    by the final answer. The BASE model has not seen this format and will
    typically emit prose. Fallback: treat the whole output as the answer.
    """
    if not raw_output:
        return "", ""

    match = _REASONING_PATTERN.search(raw_output)
    if match:
        reasoning = match.group(1).strip()
        # Answer is everything after the closing tag
        answer = raw_output[match.end():].strip()
        return reasoning, answer

    # No tags — base model. Heuristic: if there's a clear "Answer:" or final
    # paragraph, use that; otherwise the whole text is the answer.
    answer_match = re.search(
        r"(?:final answer|answer)\s*[:\-]\s*(.+?)$",
        raw_output, re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        return raw_output[: answer_match.start()].strip(), answer_match.group(1).strip()

    return "", raw_output.strip()