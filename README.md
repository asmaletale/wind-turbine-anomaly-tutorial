# O&M in Wind Turbines with AI Methods — Hands-On Tutorial

PyTorch: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asmaletale/wind-turbine-anomaly-tutorial/blob/main/TOB_2026.ipynb)
&nbsp;&nbsp;Keras: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asmaletale/wind-turbine-anomaly-tutorial/blob/main/TOB_2026_keras.ipynb)

> A single, self-contained ~90-minute hands-on session: build an **autoencoder**
> for **anomaly detection** on real **wind-turbine SCADA data** (EDP open
> dataset). Theory and code live in one Colab notebook — open it and run.

`TOB_2026.ipynb` interleaves short concept sections (what ML is, supervised vs.
unsupervised, why anomaly detection) with a complete, runnable pipeline. To stay
inside the time budget it **loads pre-trained autoencoder weights** instead of
training live, so the whole notebook runs end-to-end in a few minutes on a free
Colab CPU.

### Two framework editions

The same tutorial is available in two interchangeable versions — identical data,
preprocessing, plots and narrative; only the model framework differs. Pick one:

| Notebook | Framework | Runner | Train + save weights |
|---|---|---|---|
| `TOB_2026.ipynb` | PyTorch | `run_tutorial_pytorch.py` | `scripts/train_and_save.py` |
| `TOB_2026_keras.ipynb` | Keras / TensorFlow | `run_tutorial_keras.py` | `scripts/train_and_save_keras.py` |

Both share the `tutorial_helpers` package; the model code is split into
`models.py` (PyTorch) and `models_keras.py` (Keras) so neither import pulls in
the other framework.

## Quick start

1. Click the **Open in Colab** badge above.
2. `Runtime → Run all`. The first cell clones this repo and installs deps.

## Repository layout

```
TOB_2026.ipynb                  the tutorial — PyTorch (theory + hands-on)
TOB_2026_keras.ipynb            the tutorial — Keras / TensorFlow
run_tutorial_pytorch.py         runnable .py version (PyTorch)
run_tutorial_keras.py           runnable .py version (Keras)
tutorial_helpers/               importable helpers — keeps the notebook short
  data_utils.py                 loading, labelling, sequencing, Isolation Forest
  plotting.py                   all visualisations (was one 423-line cell)
  models.py                     the dense autoencoder (PyTorch)
  models_keras.py               the dense autoencoder (Keras)
  feature_groups.json           named SCADA feature groups
assets/                         figures used in the theory sections (from the deck)
data/                           the SCADA dataset (parquet, ~15 MB)
models/                         pre-trained weights + fitted scaler
scripts/train_and_save.py       one-time offline script (PyTorch) -> models/
scripts/train_and_save_keras.py one-time offline script (Keras)   -> models/
requirements.txt                PyTorch deps
requirements_keras.txt          Keras / TensorFlow deps
```

## Regenerating the pre-trained model

The committed weights are produced once with the offline training script for
your chosen framework. Both mirror the notebook pipeline exactly (TRANSFORMER
feature group, `RobustScaler`, 144-step daily windows, dense AE, 100 epochs) and
also write the fitted `models/scaler.pkl`:

```bash
# PyTorch -> models/dense_ae_pretrained.pt
python scripts/train_and_save.py

# Keras   -> models/dense_ae_pretrained.weights.h5
python scripts/train_and_save_keras.py
```

The two weight files are framework-specific and not interchangeable; each
notebook loads its own.

## Data & attribution

SCADA and fault-log data derive from the open **EDP (Energias de Portugal)**
wind-turbine dataset. Please retain attribution to EDP when reusing this
material.
