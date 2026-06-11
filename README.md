# Fine-Tuning Gemma 4 E2B for Medical Reasoning

Fine-tunes `unsloth/gemma-4-E2B-it` on `FreedomIntelligence/medical-o1-reasoning-SFT`
using LoRA (and QLoRA for comparison) on a free Kaggle T4 GPU.

**Status:** WIP — see milestone progress in `docs/design-notes.md`.
**Disclaimer:** Not for clinical use. Learning exercise only.

## Quick start

### 1. Local development (laptop, no GPU)

```bash
git clone https://github.com/Manojcs47/Fine-Tuning-Gemma.git
cd Fine-Tuning-Gemma
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
cp .env.example .env               # then edit .env with your tokens
pytest                             # smoke-test the codebase
```

### 2. Training on Kaggle

Prerequisites: Kaggle account with phone-verified GPU access, HF account
with Gemma 4 license accepted, W&B account.

1. Create a Kaggle notebook with GPU T4 ×2, Internet On, Persistence "Files only".
2. Add Kaggle Secrets `HF_TOKEN` and `WANDB_API_KEY`.
3. Follow the cells in `notebooks/KAGGLE_RUNNER.md`.

## Repo layout
src/gemma_medical/   # all importable logic
scripts/             # thin CLI entrypoints
notebooks/           # Kaggle runner docs
configs/             # YAML experiment configs
tests/               # unit tests (CPU-only)
docs/adr/            # architecture decision records
report/              # technical report (M6)
## Engineering standards

- Strict mypy on `src/`.
- pydantic-settings for runtime config; pydantic models for experiment configs.
- Structured logging via structlog.
- All hyperparameters live in `configs/*.yaml` — no magic numbers in code.
- Every W&B run is tagged with its full config and the git commit SHA.

## License

Apache-2.0 (matches Gemma 4 and the dataset).