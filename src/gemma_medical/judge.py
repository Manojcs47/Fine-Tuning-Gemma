"""LLM-as-judge for reasoning quality.

The judge scores a (question, gold, prediction) triple on three dimensions:
  - conclusion_correctness: does the prediction's final answer match the gold?
  - reasoning_validity: is the reasoning coherent and medically sound?
  - no_fabrication: are there fabricated drug names, dosages, or facts?

Each dimension scored 1-5. Mean reported as aggregate.

Default backend: a Gemma model (typically the same base model used for
generation, to keep T4 memory feasible). Using the same model as predictor
AND judge introduces some self-bias risk, but for relative comparisons
between base / LoRA / QLoRA the bias cancels.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from gemma_medical.config import EvaluationConfig
from gemma_medical.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeResult:
    """One judge score on one (question, gold, prediction) triple."""

    conclusion_correctness: float  # 1-5
    reasoning_validity: float       # 1-5
    no_fabrication: float           # 1-5
    justification: str
    parse_ok: bool                  # False if judge output couldn't be parsed

    @property
    def aggregate(self) -> float:
        return (
            self.conclusion_correctness
            + self.reasoning_validity
            + self.no_fabrication
        ) / 3.0


class Judge(Protocol):
    """Anything that can score a (question, gold, prediction) triple."""

    def score(self, question: str, gold: str, prediction: str) -> JudgeResult: ...


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM = """You are an expert medical reasoning evaluator. You score model \
predictions against gold-standard medical answers. You are strict, fair, and \
return ONLY a JSON object — no preamble, no explanation outside the JSON."""


_JUDGE_USER_TEMPLATE = """Score the model prediction below on three dimensions, each from 1 to 5.

QUESTION:
{question}

GOLD ANSWER:
{gold}

MODEL PREDICTION:
{prediction}

SCORING RUBRIC (1=worst, 5=best):
1. conclusion_correctness — Does the prediction's final answer agree with the gold?
   1: contradicts gold. 3: partially correct. 5: matches gold's conclusion.
2. reasoning_validity — Is the reasoning coherent and medically sound?
   1: incoherent or missing. 3: some valid steps, some flaws. 5: rigorous, step-by-step.
3. no_fabrication — Absence of made-up drug names, dosages, anatomical terms, or facts.
   1: multiple fabricated entities. 3: minor inaccuracies. 5: no fabrication.

Return ONLY this JSON, nothing else:
{{"conclusion_correctness": <1-5>, "reasoning_validity": <1-5>, "no_fabrication": <1-5>, "justification": "<one sentence>"}}"""


# ---------------------------------------------------------------------------
# Gemma-backed judge
# ---------------------------------------------------------------------------


class GemmaJudge:
    """Use a loaded Gemma model + tokenizer as judge."""

    def __init__(self, model: Any, tokenizer: Any, config: EvaluationConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def score(self, question: str, gold: str, prediction: str) -> JudgeResult:
        prompt = self._build_prompt(question, gold, prediction)
        raw = self._generate(prompt)
        return self._parse(raw)

    def score_batch(
        self,
        triples: list[tuple[str, str, str]],
    ) -> list[JudgeResult]:
        """Score multiple triples. Sequential — judge calls are small."""
        results: list[JudgeResult] = []
        for i, (q, g, p) in enumerate(triples):
            try:
                r = self.score(q, g, p)
            except Exception as e:
                log.warning("judge_call_failed", idx=i, error=str(e))
                r = JudgeResult(
                    conclusion_correctness=0.0,
                    reasoning_validity=0.0,
                    no_fabrication=0.0,
                    justification=f"<judge error: {e}>",
                    parse_ok=False,
                )
            results.append(r)
            if (i + 1) % 10 == 0:
                log.info("judge_progress", done=i + 1, total=len(triples))
        return results

    # -- internals -----------------------------------------------------------

    def _build_prompt(self, question: str, gold: str, prediction: str) -> str:
        user = _JUDGE_USER_TEMPLATE.format(
            question=question.strip(),
            gold=gold.strip(),
            prediction=prediction.strip(),
        )
        messages = [
            {"role": "user", "content": _JUDGE_SYSTEM + "\n\n" + user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, prompt: str) -> str:
        import torch  # type: ignore[import-not-found]

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.judge_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

    def _parse(self, raw: str) -> JudgeResult:
        return parse_judge_output(raw)


# ---------------------------------------------------------------------------
# Parsing (exposed for testing)
# ---------------------------------------------------------------------------


_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)


def parse_judge_output(raw: str) -> JudgeResult:
    """Robustly parse the judge's JSON output."""
    if not raw:
        return _empty_result("empty judge output")

    # Strip code fences if present
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    match = _JSON_BLOCK.search(cleaned)
    if not match:
        return _empty_result(f"no JSON found in: {raw[:100]}")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return _empty_result(f"JSON decode error: {e}")

    try:
        return JudgeResult(
            conclusion_correctness=_clamp(data.get("conclusion_correctness", 0)),
            reasoning_validity=_clamp(data.get("reasoning_validity", 0)),
            no_fabrication=_clamp(data.get("no_fabrication", 0)),
            justification=str(data.get("justification", ""))[:500],
            parse_ok=True,
        )
    except (TypeError, ValueError) as e:
        return _empty_result(f"field error: {e}")


def _clamp(v: Any) -> float:
    """Clamp to [1, 5]. Returns 0 if invalid (which fails parse_ok)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(1.0, min(5.0, f))


def _empty_result(reason: str) -> JudgeResult:
    return JudgeResult(
        conclusion_correctness=0.0,
        reasoning_validity=0.0,
        no_fabrication=0.0,
        justification=f"<parse failed: {reason}>",
        parse_ok=False,
    )