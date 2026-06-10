# O&M in Wind Turbines with AI Methods — Hands-On Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asmaletale/wind-turbine-anomaly-tutorial/blob/main/TOB_2026.ipynb)

> A single, self-contained ~90-minute hands-on session: build an **autoencoder**
> for **anomaly detection** on real **wind-turbine SCADA data** (EDP open
> dataset). Theory and code live in one Colab notebook — open it and run.

`TOB_2026.ipynb` interleaves short concept sections (what ML is, supervised vs.
unsupervised, why anomaly detection) with a complete, runnable pipeline. To stay
inside the time budget it **loads pre-trained autoencoder weights** instead of
training live, so the whole notebook runs end-to-end in a few minutes on a free
Colab CPU.

## Quick start

1. Click the **Open in Colab** badge above (after replacing
   `YOUR-GITHUB-USERNAME` with your handle).
2. `Runtime → Run all`. The first cell clones this repo and installs deps.

## Repository layout

```
TOB_2026.ipynb              the tutorial (theory + hands-on)
tutorial_helpers/           importable helpers — keeps the notebook short
  data_utils.py             loading, labelling, sequencing, Isolation Forest
  plotting.py               all visualisations (was one 423-line cell)
  models.py                 the dense autoencoder
  feature_groups.json       named SCADA feature groups
assets/                     figures used in the theory sections (from the deck)
data/                       the SCADA dataset (parquet, ~15 MB)
models/                     pre-trained weights + fitted scaler
scripts/train_and_save.py   one-time offline script that produced models/
requirements.txt
```

## Regenerating the pre-trained model

The committed weights were produced once with:

```bash
python scripts/train_and_save.py
```

This mirrors the notebook pipeline exactly (TRANSFORMER feature group,
`RobustScaler`, 144-step daily windows, dense AE, 100 epochs) and writes
`models/dense_ae_pretrained.weights.h5` and `models/scaler.pkl`.

## Data & attribution

SCADA and fault-log data derive from the open **EDP (Energias de Portugal)**
wind-turbine dataset. Please retain attribution to EDP when reusing this
material.
