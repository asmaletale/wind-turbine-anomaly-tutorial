"""
tutorial_helpers
================
Helper modules for the *O&M in Wind Turbines with AI Methods* hands-on tutorial.
"""

from .data_utils import (
    add_anomaly_label_simple,
    add_angle_features,
    angle_diff,
    anomaly_array_filtering,
    anomaly_array_filtering_tf,   # backward-compatible alias
    create_sequences,
    detect_anomalies_with_isolation_forest,
    get_feature_group,
    load_feature_groups,
)
from .models import create_dense_autoencoder
from .plotting import (
    plot_anomaly_analysis,
    plot_autoencoder_architecture,
    plot_df_with_log,
    plot_latent_space,
    plot_latent_space_projections,
    plot_loss_distribution,
    plot_mahalanobis_distance,
    plot_reconstruction_comparison,
    plot_scada_with_faults_only,
    plot_scaling_effect,
    plot_training_history,
)

__all__ = [
    # data_utils
    "load_feature_groups",
    "get_feature_group",
    "angle_diff",
    "add_angle_features",
    "add_anomaly_label_simple",
    "create_sequences",
    "anomaly_array_filtering",
    "anomaly_array_filtering_tf",
    "detect_anomalies_with_isolation_forest",
    # models
    "create_dense_autoencoder",
    # plotting
    "plot_anomaly_analysis",
    "plot_scaling_effect",
    "plot_autoencoder_architecture",
    "plot_reconstruction_comparison",
    "plot_latent_space",
    "plot_latent_space_projections",
    "plot_loss_distribution",
    "plot_mahalanobis_distance",
    "plot_df_with_log",
    "plot_scada_with_faults_only",
    "plot_training_history",
]
