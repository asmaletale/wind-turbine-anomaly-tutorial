"""
train_and_save.py
=================
ONE-TIME, OFFLINE script. Run it once to produce the pre-trained artifacts:

    models/dense_ae_pretrained.pt   (PyTorch state-dict)
    models/scaler.pkl               (fitted RobustScaler)

Run from the repo root:

    python scripts/train_and_save.py

The pipeline mirrors the notebook exactly (same feature group, same RobustScaler,
same architecture and hyper-parameters).
"""

import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from tutorial_helpers import (  # noqa: E402
    add_anomaly_label_simple,
    anomaly_array_filtering,
    create_dense_autoencoder,
    create_sequences,
    get_feature_group,
)

# --------------------------------------------------------------------------- #
# Configuration — keep in sync with the notebook                               #
# --------------------------------------------------------------------------- #
SEED = 42
TEST_WT = "T07"
FEATURE_GROUP = "TRANSFORMER"
SEQUENCE_LENGTH = 144
STRIDE = SEQUENCE_LENGTH // 4
BATCH_SIZE = 64
EPOCHS = 100
HIDDEN_DIM = 128
LATENT_DIM = 8
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2

DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "dense_ae_pretrained.pt")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seeds()
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # --- 6. Build PyTorch dataset -------------------------------------------
    n_val = int(len(sequences) * VAL_SPLIT)
    idx = np.random.permutation(len(sequences))
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    X_train = torch.tensor(sequences[train_idx], dtype=torch.float32)
    X_val = torch.tensor(sequences[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, X_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, X_val),
                            batch_size=BATCH_SIZE)

    # --- 7. Build + train the autoencoder -----------------------------------
    model = create_dense_autoencoder(
        input_dim=len(group_features),
        sequence_length=SEQUENCE_LENGTH,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={np.mean(train_losses):.4f}  "
                  f"val={np.mean(val_losses):.4f}")

    # --- 8. Save artifacts --------------------------------------------------
    torch.save(model.state_dict(), WEIGHTS_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nSaved weights -> {WEIGHTS_PATH}")
    print(f"Saved scaler  -> {SCALER_PATH}")


if __name__ == "__main__":
    main()
