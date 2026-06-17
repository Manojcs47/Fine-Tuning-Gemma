"""Custom training callbacks: sample printer, memory monitor, structured logger.

These are the operator-side signals referenced in docs/manual-kill-criteria.md.
By the time loss curves tell you something is wrong, the generations have
already started looking bad — the sample printer is what surfaces that.

GPU-only — imports torch, calls generate.
"""
from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import (  # type: ignore[import-not-found]
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from gemma_medical.data import format_for_inference
from gemma_medical.inference import parse_response
from gemma_medical.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer/Processor compatibility helper
# ---------------------------------------------------------------------------
#
# Gemma 4 is multimodal, so Unsloth's FastModel.from_pretrained returns a
# `Gemma4Processor` rather than a plain `PreTrainedTokenizer`. The processor:
#   1. Binds positional args to `images=`, not `text=` — see `inference.py`.
#   2. May not expose `pad_token_id` / `eos_token_id` directly on every
#      version; they live on `tokenizer.tokenizer` for sure.
# This helper mirrors `inference._text_tokenizer` so the sample printer is
# robust to whichever object we're handed.


def _text_tokenizer(tokenizer: Any) -> Any:
    """Return the underlying text tokenizer, whether given a tokenizer or processor."""
    if hasattr(tokenizer, "encode"):
        return tokenizer
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "encode"):
        return inner
    return tokenizer  # last-resort fallback


# ---------------------------------------------------------------------------
# Sample printer
# ---------------------------------------------------------------------------


class SamplePrinterCallback(TrainerCallback):
    """Every N steps, print one validation generation to stdout.

    This is the single most useful operator-side signal. Loss curves lag;
    generation collapse is visible immediately.
    """

    def __init__(
        self,
        tokenizer: Any,
        val_dataset_raw: Dataset,
        every_n_steps: int = 100,
        max_new_tokens: int = 300,
        n_examples_to_rotate: int = 5,
    ) -> None:
        self.tokenizer = tokenizer
        self.val_raw = val_dataset_raw
        self.every_n = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.n_rotate = min(n_examples_to_rotate, len(val_dataset_raw))
        self._call_count = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if state.global_step <= 0:
            return control
        if state.global_step % self.every_n != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        try:
            self._print_one_sample(model, state.global_step)
        except Exception as e:
            log.warning(
                "sample_printer_failed",
                step=state.global_step,
                error_type=type(e).__name__,
                error=str(e),
            )

        return control

    def _print_one_sample(self, model: Any, step: int) -> None:
        """Generate one validation sample and print it.

        Memory hygiene: KV cache and intermediate tensors from generate() can
        linger in PyTorch's caching allocator and starve the next training
        step. We explicitly del every tensor created here and call
        empty_cache() before returning, so the next step gets a clean slate.
        """
        import gc
        import torch  # type: ignore[import-not-found]

        idx = self._call_count % self.n_rotate
        self._call_count += 1
        ex = self.val_raw[idx]
        question = ex["Question"]
        gold = ex["Response"]

        prompt = format_for_inference(question, self.tokenizer)

        # IMPORTANT: pass text as a keyword arg. Unsloth Zoo patches
        # Gemma4Processor.__call__ to `(images=None, text=None, videos=None,
        # **kwargs)`, so a positional `tokenizer(prompt, ...)` binds `prompt`
        # to `images`.
        inputs = self.tokenizer(
            text=prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        ).to(model.device)

        inner = _text_tokenizer(self.tokenizer)
        pad_id = getattr(inner, "pad_token_id", None) or getattr(inner, "eos_token_id", None)
        eos_id = getattr(inner, "eos_token_id", None)

        # CRITICAL memory hygiene — the reason the smoke test OOM'd at step 11.
        #
        # generate() on an Unsloth model allocates a static inference cache
        # (KV cache + workspace), several GB for a 5B model. Plain
        # `model.eval()` / `model.train()` only flips the nn.Module training
        # flag; it does NOT release that cache. So after the sample printed at
        # step 10 the cache stayed *live* (not merely reserved — empty_cache()
        # can't touch it), leaving ~18 MB free, and the step-11 forward OOM'd.
        #
        # `FastModel.for_inference(model)` is the supported way to enter
        # generation mode, and `FastModel.for_training(model)` tears the
        # inference cache back down and restores the training state (gradient
        # checkpointing, use_cache=False, requires_grad on the LoRA params).
        # inference.py uses the same for_inference path. This is what lets a
        # 16-bit-LoRA 5B model keep printing samples on a 16 GB T4.
        from unsloth import FastModel  # type: ignore[import-not-found]

        was_training = model.training
        outputs = None
        new_tokens = None
        raw = ""

        try:
            FastModel.for_inference(model)
        except Exception:
            model.eval()

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
            new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        finally:
            # Order matters: drop our own references first, then for_training()
            # (which frees Unsloth's inference cache), then empty_cache() to
            # return the now-free blocks to the allocator before the next step.
            try:
                del inputs
                if outputs is not None:
                    del outputs
                if new_tokens is not None:
                    del new_tokens
            except NameError:
                pass
            try:
                FastModel.for_training(model)
            except Exception:
                if was_training:
                    model.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        reasoning, answer = parse_response(raw)

        print("\n" + "─" * 60)
        print(f"SAMPLE @ step {step}  (val idx {idx})")
        print("─" * 60)
        print(f"Q: {question[:200]}{'...' if len(question) > 200 else ''}")
        print(f"\nGOLD: {gold[:300]}{'...' if len(gold) > 300 else ''}")
        print(f"\nMODEL ({len(raw)} chars):")
        print(raw[:800] + ("..." if len(raw) > 800 else ""))
        if reasoning:
            print(f"\n[parsed reasoning: {len(reasoning)} chars]")
        print(f"[parsed answer: {len(answer)} chars]")
        print("─" * 60 + "\n")

# ---------------------------------------------------------------------------
# Memory monitor
# ---------------------------------------------------------------------------


class MemoryMonitorCallback(TrainerCallback):
    """Log VRAM usage at every eval. Catches the 'memory climbs every step' bug."""

    def __init__(self, log_every_n_steps: int = 100) -> None:
        self.every_n = log_every_n_steps
        self.peak_seen: float = 0.0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if state.global_step <= 0 or state.global_step % self.every_n != 0:
            return control

        try:
            import torch  # type: ignore[import-not-found]
            if not torch.cuda.is_available():
                return control

            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            free, total = torch.cuda.mem_get_info()
            free_gb = free / 1e9

            self.peak_seen = max(self.peak_seen, peak)

            log.info(
                "memory_snapshot",
                step=state.global_step,
                allocated_gb=round(allocated, 2),
                reserved_gb=round(reserved, 2),
                peak_gb=round(peak, 2),
                free_gb=round(free_gb, 2),
            )
        except Exception as e:
            log.warning("memory_monitor_failed", error=str(e))

        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        log.info("training_peak_vram_gb", peak=round(self.peak_seen, 2))
        return control


# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------


class StructuredLoggerCallback(TrainerCallback):
    """Mirror trainer.log() metrics through structlog so they appear in JSON logs."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if logs is None:
            return control
        # Avoid logging huge things; keep numeric fields only.
        numeric = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if numeric:
            log.info("trainer_metrics", step=state.global_step, **numeric)
        return control