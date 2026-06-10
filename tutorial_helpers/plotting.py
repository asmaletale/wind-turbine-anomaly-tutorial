"""
plotting.py
===========
Visualisation helpers for the wind-turbine anomaly-detection tutorial.

Every ``results_path`` argument is optional: pass a folder to also save the
figure to disk, or leave it as ``None`` (the default) to just display it.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import torch.nn as nn


def _save(fig_or_plt, results_path, filename, is_plotly=False):
    """Save a figure only if a results_path was provided."""
    if not results_path:
        return
    os.makedirs(results_path, exist_ok=True)
    path = os.path.join(results_path, filename)
    if is_plotly:
        fig_or_plt.write_html(path)
    else:
        fig_or_plt.savefig(path, dpi=300, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# Anomaly exploration                                                          #
# --------------------------------------------------------------------------- #
def plot_anomaly_analysis(df_cleaned, log_df=None, results_path=None,
                          plot="scatter", x="wind speed - avg",
                          y="active power - avg", feature_list=()):
    """Visualise labelled anomalies (power-curve scatter or full analysis)."""
    if plot == "scatter":
        plt.figure(figsize=(8, 6))
        normal = df_cleaned[df_cleaned["is_anomaly"] == 1]
        anomaly = df_cleaned[df_cleaned["is_anomaly"] == -1]
        plt.scatter(normal[x], normal[y], alpha=0.3, label="Normal", color="blue", s=10)
        plt.scatter(anomaly[x], anomaly[y], alpha=0.6, label="Anomaly", color="red", s=10)
        plt.title("Power curve: anomalies vs normal")
        plt.xlabel(x); plt.ylabel(y); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(plt, results_path, "02b_scatter.png")
        plt.show()

    elif plot == "full":
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        anomalies = df_cleaned[df_cleaned["is_anomaly"] == -1]
        daily = anomalies.groupby(anomalies.index.date).size()
        axes[0].plot(daily.index, daily.values, "ro-", alpha=0.7)
        axes[0].set_title("Anomalies over time")
        axes[0].set_xlabel("Date"); axes[0].set_ylabel("Daily anomalies")
        axes[0].tick_params(axis="x", rotation=45)
        for col in feature_list:
            normal_data = df_cleaned[df_cleaned["is_anomaly"] == 1][col]
            anomaly_data = df_cleaned[df_cleaned["is_anomaly"] == -1][col]
            axes[1].hist(normal_data.dropna(), bins=50, alpha=0.7, label=f"Normal {col}")
            axes[1].hist(anomaly_data.dropna(), bins=50, alpha=0.7, label=f"Anomaly {col}")
        axes[1].set_title("Feature distribution"); axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Frequency"); axes[1].legend()
        plt.tight_layout()
        _save(fig, results_path, "02_anomaly_analysis.png")
        plt.show()


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #
def plot_scaling_effect(df_before, df_after, feature_name, results_path=None):
    """Show a feature's distribution before vs after scaling."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(df_before[feature_name].dropna(), bins=50, alpha=0.7, color="blue")
    axes[0].set_title(f"{feature_name}\nBefore scaling")
    axes[0].set_xlabel("Value"); axes[0].set_ylabel("Occurrence")
    axes[1].hist(df_after[feature_name].dropna(), bins=50, alpha=0.7, color="green")
    axes[1].set_title(f"{feature_name}\nAfter scaling")
    axes[1].set_xlabel("Scaled value"); axes[1].set_ylabel("Occurrence")
    axes[2].boxplot([df_before[feature_name].dropna(), df_after[feature_name].dropna()],
                    labels=["Before", "After"])
    axes[2].set_title(f"{feature_name}\nBox plot"); axes[2].set_ylabel("Value")
    plt.tight_layout()
    _save(fig, results_path, "03_scaling_effect.png")
    plt.show()


# --------------------------------------------------------------------------- #
# Model architecture                                                           #
# --------------------------------------------------------------------------- #
def plot_training_history(history, results_path=None):
    """Plot train and validation loss curves from a training run.

    Parameters
    ----------
    history : dict with keys 'train' and 'val', each a list of per-epoch losses
    """
    epochs = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train"], label="Train loss")
    axes[0].plot(epochs, history["val"],   label="Val loss")
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_xlim(1, len(history["train"]))
    axes[0].set_ylabel("MSE loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs, history["train"], label="Train loss")
    axes[1].semilogy(epochs, history["val"],   label="Val loss")
    axes[1].set_title("Training and validation loss (log scale)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_xlim(1, len(history["train"]))
    axes[1].set_ylabel("MSE loss (log)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, results_path, "04_training_history.png")
    plt.show()


def plot_autoencoder_architecture(model, results_path=None):
    """Plot layer dimensions (bottleneck) and parameter counts for a PyTorch model."""
    # Collect leaf modules (skip top-level Sequential containers)
    leaf_modules = [
        (name, m) for name, m in model.named_modules()
        if not isinstance(m, nn.Sequential) and name != ""
    ]
    layer_names = [name for name, _ in leaf_modules]
    layer_params = [sum(p.numel() for p in m.parameters()) for _, m in leaf_modules]

    # Linear layers give the bottleneck shape
    linear_layers = [(name, m) for name, m in leaf_modules if isinstance(m, nn.Linear)]
    flat_dim = model.input_dim * model.sequence_length
    arch_dims = [flat_dim] + [m.out_features for _, m in linear_layers] + [flat_dim]
    arch_labels = (["input"]
                   + [name for name, _ in linear_layers]
                   + ["output"])

    colors = ["lightgreen" if "encoder" in n else "lightsalmon" for n in layer_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.plot(arch_dims, marker="o", linewidth=2, color="steelblue")
    ax1.set_xticks(range(len(arch_dims)))
    ax1.set_xticklabels(arch_labels, rotation=45, ha="right")
    ax1.set_ylabel("Layer dimension (units)")
    ax1.set_title("Undercomplete autoencoder"); ax1.grid(True)
    ax2.bar(range(len(layer_params)), layer_params, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, rotation=45, ha="right")
    ax2.set_ylabel("Number of parameters"); ax2.set_title("Parameters by layer")
    plt.tight_layout()
    _save(fig, results_path, "05_model_architecture.png")
    plt.show()


# --------------------------------------------------------------------------- #
# Reconstruction / loss diagnostics                                            #
# --------------------------------------------------------------------------- #
def plot_reconstruction_comparison(original_sequences, reconstructed_sequences,
                                   show_features_index, results_path=None,
                                   n_examples=2, seq_timestamps=None,
                                   log_df=None, feature_names=None):
    """Plot original vs reconstructed windows side by side."""
    def _feat_label(j):
        if feature_names and j < len(feature_names):
            return feature_names[j]
        return f"Feature {j + 1}"

    if seq_timestamps is not None and log_df is not None:
        ts_index = pd.DatetimeIndex(seq_timestamps)
        seen, selected = set(), []
        for _, row in log_df.iterrows():
            target = pd.Timestamp(row["Timestamp"]).tz_localize(None) \
                     if pd.Timestamp(row["Timestamp"]).tzinfo is None \
                     else pd.Timestamp(row["Timestamp"]).tz_convert(None)
            closest = ts_index.get_indexer([target], method="nearest")[0]
            if closest not in seen:
                seen.add(closest)
                selected.append((closest, row))
        example_indices = [s[0] for s in selected]
        row_labels = [
            f"{pd.Timestamp(row['Timestamp']).date()}  —  {row.get('Component', '')}"
            for _, row in selected
        ]
    else:
        example_indices = list(range(n_examples))
        row_labels = [f"Sequence {i + 1}" for i in example_indices]

    n_rows = len(example_indices)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for row_i, seq_i in enumerate(example_indices):
        for j in show_features_index:
            axes[row_i, 0].plot(original_sequences[seq_i][:, j],
                                label=_feat_label(j), alpha=0.7)
        axes[row_i, 0].set_title(f"Original — {row_labels[row_i]}")
        axes[row_i, 0].set_xlabel("Timestep"); axes[row_i, 0].set_ylabel("Value")
        axes[row_i, 0].legend(); axes[row_i, 0].grid(True, alpha=0.3)
        for j in show_features_index:
            axes[row_i, 1].plot(reconstructed_sequences[seq_i][:, j], "--",
                                label=_feat_label(j), alpha=0.7)
        axes[row_i, 1].set_title(f"Reconstructed — {row_labels[row_i]}")
        axes[row_i, 1].set_xlabel("Timestep"); axes[row_i, 1].set_ylabel("Value")
        axes[row_i, 1].legend(); axes[row_i, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, results_path, "07_reconstruction_comparison.png")
    plt.show()


def plot_latent_space(latent_vectors, seq_timestamps, log_df=None, results_path=None):
    """Plot each latent dimension over time, with fault events marked.

    Parameters
    ----------
    latent_vectors : np.ndarray, shape (n_sequences, latent_dim)
    seq_timestamps : array-like of datetime-like values (one per sequence)
    log_df         : fault-log DataFrame with a 'Timestamp' column (optional)
    results_path   : folder to write the PNG to (optional)
    """
    n_dims = latent_vectors.shape[1]
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    times = pd.DatetimeIndex(seq_timestamps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n_dims):
        ax = axes[i]
        ax.plot(times, latent_vectors[:, i], linewidth=0.8, alpha=0.8)
        if log_df is not None:
            for _, row in log_df.iterrows():
                ax.axvline(pd.to_datetime(row["Timestamp"]),
                           color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_title(f"Latent dim {i + 1}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    _save(fig, results_path, "08_latent_space.png")
    plt.show()


def plot_latent_space_projections(latent_vectors, seq_timestamps, log_df=None,
                                  results_path=None):
    """PCA and t-SNE 2-D projections of the latent space, coloured by date.

    Parameters
    ----------
    latent_vectors : np.ndarray, shape (n_sequences, latent_dim)
    seq_timestamps : array-like of datetime-like values (one per sequence)
    log_df         : fault-log DataFrame with a 'Timestamp' column (optional)
    results_path   : folder to write the PNG to (optional)
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    times = pd.DatetimeIndex(seq_timestamps)
    t_num = (times - times[0]).total_seconds() / 86400.0  # days since start

    fault_indices = []
    if log_df is not None:
        for _, row in log_df.iterrows():
            ts = pd.Timestamp(row["Timestamp"])
            ts = ts.tz_convert(None) if ts.tzinfo is not None else ts.tz_localize(None)
            fault_indices.append(times.get_indexer([ts], method="nearest")[0])

    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(latent_vectors)

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(latent_vectors) - 1))
    tsne_proj = tsne.fit_transform(latent_vectors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = [
        f"PCA  (variance explained: {pca.explained_variance_ratio_.sum():.1%})",
        "t-SNE",
    ]
    for ax, proj, title in zip(axes, [pca_proj, tsne_proj], titles):
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=t_num, cmap="viridis",
                        s=8, alpha=0.6)
        for n, fi in enumerate(fault_indices, start=1):
            x, y = proj[fi, 0], proj[fi, 1]
            ax.plot(x, y, "o", color="red", markersize=14, zorder=5,
                    alpha=0.85, markeredgecolor="white", markeredgewidth=0.5)
            ax.text(x, y, str(n), color="white", fontsize=6, fontweight="bold",
                    ha="center", va="center", zorder=6)
        plt.colorbar(sc, ax=ax, label="Days from start")
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, results_path, "09_latent_projections.png")
    plt.show()


def plot_mahalanobis_distance(test_seq, recon, seq_timestamps, log_df=None,
                              feature_names=None, results_path=None, percentile=0.95):
    """Mahalanobis distance of reconstruction residuals over time.

    Top panel  : multivariate Mahalanobis distance (captures cross-signal anomalies).
    Bottom panel: per-feature mean signed residual (shows which signal drives the distance).

    Parameters
    ----------
    test_seq      : np.ndarray (n, seq_len, n_feat)  — original sequences
    recon         : np.ndarray (n, seq_len, n_feat)  — reconstructed sequences
    seq_timestamps: array-like of datetime-like values (one per sequence)
    log_df        : fault-log DataFrame with a 'Timestamp' column (optional)
    feature_names : list of feature name strings (optional)
    results_path  : folder to write the PNG to (optional)
    percentile    : detection threshold percentile (default 0.95)
    """
    from sklearn.covariance import EmpiricalCovariance

    def _feat_label(j):
        if feature_names and j < len(feature_names):
            return feature_names[j]
        return f"Feature {j + 1}"

    residuals = test_seq - recon                    # (n, seq_len, n_feat)
    residual_per_feat = residuals.mean(axis=1)      # (n, n_feat) — mean over time

    cov = EmpiricalCovariance()
    cov.fit(residual_per_feat)
    mahal = np.sqrt(np.maximum(cov.mahalanobis(residual_per_feat), 0))

    times = pd.DatetimeIndex(seq_timestamps)
    n_feat = residuals.shape[2]
    threshold = np.percentile(mahal, percentile * 100)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(times, mahal, linewidth=0.8, alpha=0.8, label="Mahalanobis distance")
    axes[0].axhline(threshold, color="red", linestyle="--",
                    label=f"Threshold ({percentile * 100:.0f}th pct)")
    if log_df is not None:
        for _, row in log_df.iterrows():
            axes[0].axvline(pd.to_datetime(row["Timestamp"]),
                            color="red", alpha=0.3, linewidth=0.8)
    axes[0].set_title("Mahalanobis distance of reconstruction residuals")
    axes[0].set_ylabel("Distance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for j in range(n_feat):
        axes[1].plot(times, residual_per_feat[:, j],
                     label=_feat_label(j), linewidth=0.8, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.6, linestyle=":")
    if log_df is not None:
        for _, row in log_df.iterrows():
            axes[1].axvline(pd.to_datetime(row["Timestamp"]),
                            color="red", alpha=0.3, linewidth=0.8)
    axes[1].set_title("Per-feature mean signed residual (true − reconstructed)")
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save(fig, results_path, "10_mahalanobis.png")
    plt.show()


def plot_loss_distribution(df, features, loss_df, test_log, results_path=None,
                           percentile=0.95):
    """Show reconstruction-loss distribution and its alignment with faults."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].hist(loss_df["loss"], bins=50, alpha=0.7, color="blue")
    axes[0, 0].set_title("Reconstruction loss distribution")
    axes[0, 0].set_xlabel("Loss"); axes[0, 0].set_ylabel("Occurrence")
    axes[0, 0].axvline(loss_df["loss"].mean(), color="red", linestyle="--", label="Mean")
    axes[0, 0].axvline(loss_df["loss"].quantile(0.95), color="orange", linestyle="--",
                       label="95th pct")
    axes[0, 0].legend()

    axes[0, 1].boxplot(loss_df["loss"].dropna())
    axes[0, 1].set_title("Loss box plot"); axes[0, 1].set_ylabel("Loss")

    axes[1, 0].plot(df[features])
    for _, row in test_log.iterrows():
        axes[1, 0].axvline(pd.to_datetime(row["Timestamp"]), color="red", alpha=0.3)
    axes[1, 0].set_title("Feature group"); axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_xlabel("Date")

    axes[1, 1].plot(loss_df.index, loss_df["loss"], alpha=0.7, label="Loss")
    threshold = loss_df["loss"].quantile(percentile)
    axes[1, 1].axhline(threshold, color="red", linestyle="--",
                       label=f"Threshold {percentile * 100:.0f}%")
    for _, row in test_log.iterrows():
        axes[1, 1].axvline(pd.to_datetime(row["Timestamp"]), color="red", alpha=0.3)
    axes[1, 1].set_title("Loss over time with threshold")
    axes[1, 1].set_xlabel("Date"); axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend(); axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    _save(fig, results_path, "06_loss_analysis.png")
    plt.show()


# --------------------------------------------------------------------------- #
# Interactive Plotly overview (SCADA + losses + fault log)                    #
# --------------------------------------------------------------------------- #
def plot_df_with_log(loss_df, log_df, scada_df, results_path=None,
                     output_name="plot", show_vertical_lines=True):
    """Interactive two-panel Plotly figure: SCADA signals on top, losses below."""
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=("Original signals", "Losses"))
    for col in scada_df.columns:
        fig.add_trace(go.Scatter(x=scada_df.index, y=scada_df[col], mode="lines",
                                 name=f"(signal) {col}"), row=1, col=1)
    for col in loss_df.columns:
        fig.add_trace(go.Scatter(x=loss_df.index, y=loss_df[col], mode="lines",
                                 name=f"(loss) {col}"), row=2, col=1)

    y_min, y_max = loss_df.min().min(), loss_df.max().max()
    for _, row in log_df.iterrows():
        if show_vertical_lines:
            fig.add_trace(go.Scatter(x=[row["Timestamp"], row["Timestamp"]],
                                     y=[y_min, y_max], mode="lines",
                                     line=dict(color="red", dash="dash", width=1),
                                     showlegend=False), row=2, col=1)
        fig.add_annotation(x=row["Timestamp"], y=y_max, text=row["Component"],
                           textangle=280, showarrow=False, font=dict(color="red"),
                           xref="x2", yref="y2")
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color="red"),
                                 name=f"{row['Component']}: {row['Remarks']}"))

    wtg = log_df["WTG"].unique()[0] if "WTG" in log_df.columns else "?"
    fig.update_layout(title_text=f"SCADA signals and reconstruction loss - turbine {wtg}",
                      template="plotly_white")
    _save(fig, results_path, f"{output_name}.html", is_plotly=True)
    return fig


def plot_scada_with_faults_only(scada_df, log_df, results_path=None,
                                output_name="scada", show_vertical_lines=True):
    """Plot raw SCADA signals with fault-log events overlaid."""
    fig = go.Figure()
    numeric_cols = scada_df.select_dtypes(include=np.number).columns
    for col in scada_df.columns:
        fig.add_trace(go.Scatter(x=scada_df.index, y=scada_df[col], mode="lines",
                                 name=f"SCADA - {col}"))
    y_min = scada_df[numeric_cols].min().min()
    y_max = scada_df[numeric_cols].max().max()
    for _, row in log_df.iterrows():
        ts = pd.to_datetime(row["Timestamp"])
        if show_vertical_lines:
            fig.add_trace(go.Scatter(x=[ts, ts], y=[y_min, y_max], mode="lines",
                                     line=dict(color="red", dash="dash", width=1),
                                     showlegend=False))
            fig.add_annotation(x=ts, y=0, text=row["Component"], textangle=280,
                               showarrow=False, font=dict(color="red"),
                               xref="x", yref="y")
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color="red"),
                                 name=f"{row['Component']}: {row['Remarks']}"))
    wtg = log_df["WTG"].iloc[0] if "WTG" in log_df.columns else "Unknown"
    fig.update_layout(title=f"SCADA signals with fault logs - turbine {wtg}",
                      xaxis_title="Time", yaxis_title="SCADA value",
                      template="plotly_white")
    _save(fig, results_path, f"{output_name}.html", is_plotly=True)
    return fig
