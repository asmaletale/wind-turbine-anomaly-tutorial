"""
run_tutorial_pytorch.py
=======================
Runnable .py version of the TOB_2026 notebook, rewritten for PyTorch.

Usage:
    python run_tutorial_pytorch.py              # load pretrained weights if available
    python run_tutorial_pytorch.py --train 5    # train for 5 epochs instead
    python run_tutorial_pytorch.py --train 100  # full training run

Plots are saved to results/ (non-interactive, no display needed).
Pre-trained weights are expected at models/dense_ae_pretrained.pt.
If they are missing the script trains from scratch for --train epochs (default 5).
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")   # write to file; no display required

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

from tutorial_helpers import (
    add_anomaly_label_simple,
    anomaly_array_filtering,
    create_dense_autoencoder,
    create_sequences,
    get_feature_group,
    plot_autoencoder_architecture,
    plot_latent_space,
    plot_training_history,
    plot_latent_space_projections,
    plot_loss_distribution,
    plot_mahalanobis_distance,
    plot_reconstruction_comparison,
    plot_scaling_effect,
)

# --------------------------------------------------------------------------- #
# Config (mirrors the notebook)                                                #
# --------------------------------------------------------------------------- #
SEED = 42
TEST_WT = "T07"
FEATURE_GROUP = "TRANSFORMER"
SEQUENCE_LENGTH = 144
STRIDE = SEQUENCE_LENGTH // 4
BATCH_SIZE = 64
HIDDEN_DIM = 32
LATENT_DIM = 4
LEARNING_RATE = 1e-3
RESULTS_DIR = "results/pytorch"
WEIGHTS_PATH = "models/dense_ae_pretrained.pt"


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Training helper                                                              #
# --------------------------------------------------------------------------- #
def train(model, sequences, epochs, device):
    n_val = int(len(sequences) * 0.2)
    idx = np.random.permutation(len(sequences))
    X_train = torch.tensor(sequences[idx[n_val:]], dtype=torch.float32)
    X_val = torch.tensor(sequences[idx[:n_val]], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, X_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, X_val), batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        train_losses, val_losses = [], []
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                train_losses.append(loss_fn(model(xb), yb).item())
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())

        history["train"].append(np.mean(train_losses))
        history["val"].append(np.mean(val_losses))
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train={history['train'][-1]:.4f}  val={history['val'][-1]:.4f}")
    return history


# --------------------------------------------------------------------------- #
# Main pipeline                                                                #
# --------------------------------------------------------------------------- #
def main(train_epochs=None, save_weights=False):
    set_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Load data -------------------------------------------------------
    print("\n=== 1. Load data ===")
    df_full = pd.read_parquet("data/df_full_edp.prqt.gzip")
    df_log = pd.read_parquet("data/edp_log.prqt.gzip")
    print(f"SCADA: {df_full.shape[0]:,} rows x {df_full.shape[1]} cols")
    print(f"Fault log: {df_log.shape[0]} events")

    # --- 2. Select turbines and features ------------------------------------
    print("\n=== 2. Setup ===")
    wtgs = df_full["WTG"].unique()
    train_wtgs = wtgs[wtgs != TEST_WT]
    group_features = get_feature_group(FEATURE_GROUP)
    test_log = df_log[df_log["WTG"] == TEST_WT].copy()
    print(f"Train turbines: {list(train_wtgs)}  |  Test turbine: {TEST_WT}")
    print(f"Features: {group_features}")

    # --- 3. Label anomalies -------------------------------------------------
    print("\n=== 3. Label anomalies ===")
    labelled = [
        add_anomaly_label_simple(df_full[df_full["WTG"] == w].copy(),
                                 df_log[df_log["WTG"] == w].copy(), verbose=False)
        for w in wtgs
    ]
    df_labelled = pd.concat(labelled)

    # --- 4. Preprocess and scale --------------------------------------------
    print("\n=== 4. Scale ===")
    cols = group_features + ["WTG", "is_anomaly"]
    df_clean = df_labelled[cols].dropna(axis=1, how="all").copy()
    df_clean[group_features] = df_clean[group_features].fillna(df_clean[group_features].mean())

    scaler = RobustScaler()
    train_df = df_clean[df_clean["WTG"] != TEST_WT].copy()
    test_df = df_clean[df_clean["WTG"] == TEST_WT].copy()
    train_df[group_features] = scaler.fit_transform(train_df[group_features])
    test_df[group_features] = scaler.transform(test_df[group_features])
    print(f"train: {train_df.shape}  |  test: {test_df.shape}")

    plot_scaling_effect(df_clean[df_clean["WTG"] != TEST_WT],
                        train_df, group_features[0],
                        results_path=RESULTS_DIR)

    # --- 5. Build sequences -------------------------------------------------
    print("\n=== 5. Sequences ===")
    seqs = []
    for w in train_wtgs:
        t = train_df[train_df["WTG"] == w].drop(columns=["WTG"])
        arr = anomaly_array_filtering(create_sequences(t, SEQUENCE_LENGTH, STRIDE))
        if arr is not None:
            seqs.append(arr)
    sequences = np.concatenate(seqs, axis=0)
    print(f"Clean training sequences: {sequences.shape}")   # (n, 144, 3)

    # --- 6. Build model -----------------------------------------------------
    print("\n=== 6. Model ===")
    model = create_dense_autoencoder(
        input_dim=len(group_features),
        sequence_length=SEQUENCE_LENGTH,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total parameters: {total_params:,}")

    plot_autoencoder_architecture(model, results_path=RESULTS_DIR)

    # Load or train weights
    history = None
    weights_exist = os.path.exists(WEIGHTS_PATH)
    if train_epochs is not None:
        print(f"\nTraining for {train_epochs} epochs...")
        history = train(model, sequences, train_epochs, device)
    elif weights_exist:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device,
                                         weights_only=True))
        print(f"Pre-trained weights loaded from {WEIGHTS_PATH}")
    else:
        default_epochs = 5
        print(f"No weights at {WEIGHTS_PATH}. Training for {default_epochs} epochs "
              f"(run 'python scripts/train_and_save.py' for full training).")
        history = train(model, sequences, default_epochs, device)

    if save_weights:
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"Saved weights -> {WEIGHTS_PATH}")

    # --- 7. Detect anomalies ------------------------------------------------
    print("\n=== 7. Detect anomalies ===")
    test_seq, idx = create_sequences(test_df[group_features], SEQUENCE_LENGTH,
                                     stride=SEQUENCE_LENGTH, return_indexes=True)
    recon = model.predict(test_seq)
    latent = model.encode(test_seq)

    loss = np.mean(np.square(test_seq - recon), axis=(1, 2))
    loss_df = pd.DataFrame({"loss": loss}, index=idx)
    print(f"Reconstruction errors: {loss_df.shape}")
    print(f"Loss — mean: {loss.mean():.4f}  max: {loss.max():.4f}")

    # --- 8. Plots -----------------------------------------------------------
    print("\n=== 8. Save plots ===")
    if history is not None:
        plot_training_history(history, results_path=RESULTS_DIR)

    plot_reconstruction_comparison(
        test_seq, recon,
        show_features_index=list(range(len(group_features))),
        seq_timestamps=idx, log_df=test_log,
        feature_names=group_features,
        results_path=RESULTS_DIR)

    plot_latent_space(latent, idx, log_df=test_log, results_path=RESULTS_DIR)

    plot_latent_space_projections(latent, idx, log_df=test_log,
                                  results_path=RESULTS_DIR)

    plot_mahalanobis_distance(test_seq, recon, idx, log_df=test_log,
                              feature_names=group_features,
                              results_path=RESULTS_DIR)

    plot_loss_distribution(test_df, group_features, loss_df, test_log,
                           results_path=RESULTS_DIR, percentile=0.96)

    print(f"\nAll plots saved to {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=100, metavar="EPOCHS",
                        help="Train for EPOCHS epochs instead of loading weights")
    parser.add_argument("--save", action="store_true",
                        help="Save trained weights to models/dense_ae_pretrained.pt")
    args = parser.parse_args()
    main(train_epochs=args.train, save_weights=args.save)
