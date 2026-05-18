# DA6401 — Assignment 3: Transformer for Neural Machine Translation

A from-scratch PyTorch implementation of the Transformer architecture from
*"Attention Is All You Need"* (Vaswani et al., 2017), trained on the
**Multi30k** German → English dataset.

- **GitHub repo:** `https://github.com/Soumya-1110/DA6401_ASSIGNMENT_3`
- **W&B report (public):** `https://wandb.ai/ee23b140-iit-madras/da6401-a3/reports/DA6401-Assignment-03--VmlldzoxNjkyMTAwNg?accessToken=627286g449sfjg36knzyv9u1qlneoqaoi0zufflw2m3a0o2ap7sehlxd90eguf0d`

---

## Project structure

```text
da6401_assignment_3/
├── model.py           # Transformer, MultiHeadAttention, PositionalEncoding, masks
├── lr_scheduler.py    # Noam learning-rate scheduler
├── dataset.py         # Multi30k loading + spaCy tokenization + vocab building
├── train.py           # Training loop, label smoothing, greedy decode, BLEU
├── requirements.txt
└── README.md
```

Built entirely on `torch.nn.Linear`, `nn.Embedding`, `nn.LayerNorm` and
`F.softmax` — `nn.MultiheadAttention` is **not** used (per assignment rules).

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

`spaCy` models are only required for **training** (tokenization). At inference
time `model.py` falls back to a regex tokenizer if spaCy isn't available, so
the autograder works on bare containers.

---

## How to run

### Train

```bash
python train.py
```

Defaults (see `DEFAULT_CONFIG` in `train.py`): `d_model=256, N=3, num_heads=8,
d_ff=1024, dropout=0.1, label_smoothing=0.1, warmup_steps=4000, num_epochs=20`.
Best-validation checkpoint is saved to `checkpoints/<run_tag>.pt`.

Override any hyperparameter via the `config` dict, e.g.:

```python
from train import run_training_experiment
run_training_experiment({"use_noam": False, "fixed_lr": 1e-4,
                         "wandb_run_name": "fixed-lr"})
```

## Results summary (test BLEU on Multi30k)

| Experiment                          | Test BLEU |
|-------------------------------------|-----------|
| Noam scheduler + sinusoidal PE + LS=0.1 (default) | **~39.5** |
| Learned positional embeddings       | ~36.2     |
| Fixed LR = 1e-4 (no warmup)         | lower train loss plateau, lower final BLEU |
| No `1/√d_k` scaling                 | unstable Q/K gradients, divergent training |
| Label smoothing ε=0.0               | higher confidence but worse calibration   |

Full plots, attention heatmaps, gradient-norm traces and analysis are in the
**W&B report** linked above.

---

## Implementation notes

- **Pre-LayerNorm vs Post-LayerNorm:** Post-LN (as in the original paper),
  with `Add → LayerNorm` ordering. Justified in the report.
- **Tokenization:** spaCy at training; a `\w+|[^\w\s]` regex fallback at
  inference (matches spaCy's German tokenizer byte-for-byte on Multi30k).
- **Checkpoint hosting:** the trained weights file is too large for the
  Gradescope upload limit, so `Transformer.__init__` calls `gdown` to fetch
  it from Google Drive on first construction.

---

## Permitted libraries

`torch, numpy, matplotlib, scikit-learn, wandb, datasets, spacy, sacrebleu,
tqdm, gdown`.
