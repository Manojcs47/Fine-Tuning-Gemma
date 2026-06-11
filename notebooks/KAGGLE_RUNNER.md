# Kaggle Runner Notebook — Cell-by-Cell

This is the canonical structure of the Kaggle notebook that drives training.
Recreate these cells on Kaggle. The notebook itself stays on Kaggle; all
real logic lives in `src/gemma_medical/`.

## Session settings (right sidebar)

- Accelerator: **GPU T4 x2** (or P100)
- Persistence: **Files only**
- Internet: **On**

## Cell 1 — Clone the repo

```python
!git clone https://github.com/Manojcs47/Fine-Tuning-Gemma.git
%cd Fine-Tuning-Gemma
!git pull  # safety: get latest commits
```

## Cell 2 — Install Unsloth + project deps

```python
!pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
!pip install -r requirements-gpu.txt
!pip install -e .
```

After this cell completes, **Run → Restart & Run All from here** (Unsloth
requires a restart before its imports work cleanly).

## Cell 3 — Load secrets from Kaggle Secrets

In Kaggle: Add-ons → Secrets → enable two secrets named `HF_TOKEN` and
`WANDB_API_KEY`. Then:

```python
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
```

## Cell 4 — Verify GPU

```python
!python scripts/verify_gpu.py
```

## Cell 5 — Run a training script (placeholder until Part 6)

```python
# Example for later milestones — these scripts don't exist yet in Parts 1–4
# !python scripts/run_baseline.py
# !python scripts/run_train.py --config configs/lora_default.yaml
```

## Cell 6 — Persist outputs

Save adapters and logs to `/kaggle/working/` so they appear in the right-sidebar
Output tab and survive session end.