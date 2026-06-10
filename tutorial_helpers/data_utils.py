"""
data_utils.py
=============
Data-handling helpers for the wind-turbine anomaly-detection tutorial.

Public functions
----------------
- load_feature_groups / get_feature_group : read feature_groups.json
- add_angle_features                      : turn raw angle columns into sin/cos
- angle_diff                              : signed difference between two angles
- add_anomaly_label_simple               : label rows as normal (1) / anomaly (-1)
- create_sequences                       : sliding-window sequence generation
- anomaly_array_filtering                : drop windows containing anomalies (numpy)
- detect_anomalies_with_isolation_forest : quick Isolation-Forest baseline
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FEATURE_GROUPS = os.path.join(_HERE, "feature_groups.json")


# --------------------------------------------------------------------------- #
# Feature-group configuration                                                  #
# --------------------------------------------------------------------------- #
def load_feature_groups(file_path=_DEFAULT_FEATURE_GROUPS):
    """Load the feature-group dictionary from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def get_feature_group(group_name, feature_groups=None):
    """Return the list of SCADA columns belonging to ``group_name``."""
    if feature_groups is None:
        feature_groups = load_feature_groups()
    if group_name not in feature_groups:
        raise KeyError(
            f"Unknown feature group '{group_name}'. "
            f"Available: {list(feature_groups)}"
        )
    return feature_groups[group_name]


# --------------------------------------------------------------------------- #
# Angle features                                                               #
# --------------------------------------------------------------------------- #
def angle_diff(a_rad, b_rad):
    """Signed smallest difference (in radians) between two angles."""
    return np.arctan2(np.sin(a_rad - b_rad), np.cos(a_rad - b_rad))


def add_angle_features(df):
    """Convert circular angle columns into sine/cosine components."""
    df = df.copy()
    angle_columns = [
        "pitch angle - avg",
        "wind direction - avg",
        "nacelle position - avg",
    ]
    present = [c for c in angle_columns if c in df.columns]
    for col in present:
        df[f"{col}_rad"] = np.deg2rad(df[col])
    df.drop(columns=present, inplace=True)

    if {"wind direction - avg_rad", "nacelle position - avg_rad"} <= set(df.columns):
        df["offset_angle"] = angle_diff(
            df["wind direction - avg_rad"], df["nacelle position - avg_rad"]
        )

    rad_cols = [f"{c}_rad" for c in present] + ["offset_angle"]
    for drop in ("wind direction - avg_rad", "nacelle position - avg_rad"):
        if drop in rad_cols:
            rad_cols.remove(drop)

    for col in rad_cols:
        if col in df.columns:
            df[f"{col}_sin"] = np.sin(df[col])
            df[f"{col}_cos"] = np.cos(df[col])
    return df


# --------------------------------------------------------------------------- #
# Anomaly labelling                                                            #
# --------------------------------------------------------------------------- #
def add_anomaly_label_simple(df, log_df, verbose=True):
    """Label each timestamp as normal (1) or anomalous (-1).

    Two strategies combined:
    1. Log-based: window from 20 days before to 10 days after each fault.
    2. Condition-based: active power < 50 while wind speed > 2.5.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    log_df = log_df.copy()
    log_df["Timestamp"] = pd.to_datetime(log_df["Timestamp"]).dt.tz_localize(None)

    df["is_anomaly"] = 1

    mask = pd.Series(False, index=df.index)
    for _, row in log_df.iterrows():
        start = row["Timestamp"] - pd.Timedelta(days=20)
        end = row["Timestamp"] + pd.Timedelta(days=10)
        mask |= (df.index >= start) & (df.index <= end)
    df.loc[mask, "is_anomaly"] = -1

    if {"active power - avg", "wind speed - avg"} <= set(df.columns):
        condition = (df["active power - avg"] < 50) & (df["wind speed - avg"] > 2.5)
        df.loc[condition, "is_anomaly"] = -1

    if verbose:
        n = int((df["is_anomaly"] == -1).sum())
        print(f"Anomalies labelled: {n} rows ({100 * n / len(df):.2f}%)")
    return df


# --------------------------------------------------------------------------- #
# Sequence preparation                                                         #
# --------------------------------------------------------------------------- #
def create_sequences(data, seq_length, stride=1, return_indexes=False):
    """Build overlapping fixed-length windows from a time series.

    Returns a 3-D numpy array (n_windows, seq_length, n_features).
    When return_indexes=True also returns each window's end timestamp.
    """
    sequences, indexes = [], []
    for i in range(seq_length - 1, len(data), stride):
        sequences.append(data[i - seq_length + 1 : i + 1])
        indexes.append(data.index[i])
    if return_indexes:
        return np.array(sequences), indexes
    return np.array(sequences)


def anomaly_array_filtering(array):
    """Keep only all-normal windows and strip the label column.

    Returns a float32 numpy array, or None if no windows pass.
    """
    filtered = [item[:, :-1] for item in array if not np.all(item[:, -1] == -1)]
    if filtered:
        return np.array(filtered, dtype=np.float32)
    print("No item meets the criteria.")
    return None


# Backward-compatible alias (the old name referenced TensorFlow tensors).
anomaly_array_filtering_tf = anomaly_array_filtering


# --------------------------------------------------------------------------- #
# Isolation-Forest baseline                                                    #
# --------------------------------------------------------------------------- #
def detect_anomalies_with_isolation_forest(df, features, contamination=0.05):
    """Flag anomalies with an Isolation Forest."""
    df = df.copy()
    iso = IsolationForest(contamination=contamination, random_state=42)
    pred = iso.fit_predict(df[features])
    df["is_anomaly"] = np.where(pred == -1, -1, 1)
    df["anomaly_score"] = iso.decision_function(df[features])
    n = int((df["is_anomaly"] == -1).sum())
    print(f"Anomalies detected: {n} rows ({100 * n / len(df):.2f}%)")
    return df
