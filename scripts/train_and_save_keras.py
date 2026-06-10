"""
train_and_save_keras.py
=======================
ONE-TIME, OFFLINE script — the **Keras / TensorFlow** counterpart of
``train_and_save.py``. Run it once to produce the pre-trained artifacts the
Keras notebook (``TOB_2026_keras.ipynb``) loads:

    models/dense_ae_pretrained.weights.h5   (Keras weights)
    models/scaler.pkl                       (fitted RobustScaler)

Run from the repo root:

    python scripts/train_and_save_keras.py

The pipeline mirrors the Keras notebook exactly (same feature group, same
RobustScaler, same architecture and hyper-parameters), so the saved weights load
straight back into the model built in the notebook.
"""

import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from tutorial_helpers import (  # noqa: E402
    add_anomaly_label_simple,
    anomaly_array_filtering,
    create_sequences,
    get_feature_group,
)
from tutorial_helpers.models_keras import create_dense_autoencoder_keras  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration — keep in sync with TOB_2026_keras.ipynb                       #
# (HIDDEN_DIM / LATENT_DIM must match so the weights load into the notebook    #
#  model — the notebook builds hidden_dim=32, latent_dim=4.)                   #
# --------------------------------------------------------------------------- #
SEED = 42
TEST_WT = "T07"
FEATURE_GROUP = "TRANSFORMER"
SEQUENCE_LENGTH = 144
STRIDE = SEQUENCE_LENGTH // 4
BATCH_SIZE = 64
EPOCHS = 100
HIDDEN_DIM = 32
LATENT_DIM = 4
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2

DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "dense_ae_pretrained.weights.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seeds()
    os.makedirs(MODELS_DIR, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Using device: {'GPU' if gpus else 'CPU'}")

    # --- 1. Load data -------------------------------------------------------
    df_full = pd.read_parquet(os.path.join(DATA_DIR, "df_full_edp.prqt.gzip"))
    df_log = pd.read_parquet(os.path.join(DATA_DIR, "edp_log.prqt.gzip"))

    wtgs = df_full["WTG"].unique()
    train_wtgs = wtgs[wtgs != TEST_WT]
    group_features = get_feature_group(FEATURE_GROUP)
    print(f"Train turbines: {list(train_wtgs)} | test turbine: {TEST_WT}")
    print(f"Features: {group_features}")

    # --- 2. Label anomalies -------------------------------------------------
    labelled = []
    for wtg in wtgs:
        df_t = df_full[df_full["WTG"] == wtg].copy()
        log_t = df_log[df_log["WTG"] == wtg].copy()
        labelled.append(add_anomaly_label_simple(df_t, log_t, verbose=False))
    df_concat = pd.concat(labelled, axis=0)

    # --- 3. Select features + clean -----------------------------------------
    cols = group_features + ["WTG", "is_anomaly"]
    df_cleaned = df_concat[cols].copy()
    df_cleaned = df_cleaned.dropna(axis=1, how="all")
    df_cleaned[group_features] = df_cleaned[group_features].fillna(
        df_cleaned[group_features].mean()
    )

    # --- 4. Scale (fit on train turbines only) -------------------------------
    scaler = RobustScaler()
    train_df = df_cleaned[df_cleaned["WTG"] != TEST_WT].copy()
    train_df[group_features] = scaler.fit_transform(train_df[group_features])

    # --- 5. Build clean training sequences ----------------------------------
    seqs = []
    for wtg in train_wtgs:
        t = train_df[train_df["WTG"] == wtg].drop(columns=["WTG"])
        arr = anomaly_array_filtering(create_sequences(t, SEQUENCE_LENGTH, STRIDE))
        if arr is not None:
            seqs.append(arr)
    sequences = np.concatenate(seqs, axis=0)
    print(f"Training sequences: {sequences.shape}")

    # --- 6. Build + train the autoencoder -----------------------------------
    model = create_dense_autoencoder_keras(
        input_dim=len(group_features),
        sequence_length=SEQUENCE_LENGTH,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    )
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    model.fit(
        sequences, sequences,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
    )

    # --- 7. Save artifacts --------------------------------------------------
    model.save_weights(WEIGHTS_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nSaved weights -> {WEIGHTS_PATH}")
    print(f"Saved scaler  -> {SCALER_PATH}")


if __name__ == "__main__":
    main()
