"""Inspect the raw dataset locally (no GPU, no tokenizer required).

Run: python scripts/inspect_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemma_medical.config import DataConfig  # noqa: E402
from gemma_medical.data import load_raw_dataset, make_splits  # noqa: E402
from gemma_medical.logging_setup import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    configure_logging(level="INFO")
    cfg = DataConfig()

    ds = load_raw_dataset(cfg)
    print(f"\nTotal examples: {len(ds)}")
    print(f"Columns: {ds.column_names}\n")

    print("=" * 60)
    print("FIRST EXAMPLE")
    print("=" * 60)
    ex = ds[0]
    print(f"\nQuestion:\n{ex['Question']}\n")
    print(f"Complex_CoT (first 500 chars):\n{ex['Complex_CoT'][:500]}...\n")
    print(f"Response:\n{ex['Response']}\n")

    splits = make_splits(ds, cfg)
    print("=" * 60)
    print("SPLITS")
    print("=" * 60)
    for name, count in splits.summary().items():
        print(f"  {name}: {count}")

    print("\n(Skipping chat-template step — requires the model tokenizer, "
          "which only loads on GPU.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())