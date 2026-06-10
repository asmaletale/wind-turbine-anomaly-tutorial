"""
models_keras.py
===============
Keras / TensorFlow dense autoencoder for the wind-turbine anomaly-detection
tutorial.

This is the **Keras counterpart** of ``models.py`` (which stays PyTorch). It is
intentionally kept in a separate module so importing the PyTorch path never
imports TensorFlow and vice-versa. Import it explicitly:

    from tutorial_helpers.models_keras import (
        create_dense_autoencoder_keras,
        plot_autoencoder_architecture_keras,
    )

The returned object exposes the same ``.predict()`` / ``.encode()`` /
``.input_dim`` / ``.sequence_length`` contract the notebook uses, so the rest of
the pipeline (and every plotting helper) is unchanged.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class KerasDenseAutoencoder:
    """Thin wrapper around two Keras models (autoencoder + encoder).

    Mirrors the PyTorch ``DenseAutoencoder`` API used by the tutorial:
    ``predict`` returns reconstructions shaped ``(n, sequence_length,
    input_dim)`` and ``encode`` returns latent vectors shaped ``(n,
    latent_dim)`` — both NumPy in, NumPy out.
    """

    def __init__(self, autoencoder, encoder, input_dim, sequence_length):
        self.model = autoencoder          # full input -> reconstruction model
        self.encoder = encoder            # input -> latent model (shared weights)
        self.input_dim = input_dim
        self.sequence_length = sequence_length

    # --- inference (NumPy in, NumPy out) ----------------------------------- #
    def predict(self, x):
        return self.model.predict(np.asarray(x, dtype="float32"), verbose=0)

    def encode(self, x):
        return self.encoder.predict(np.asarray(x, dtype="float32"), verbose=0)

    # --- training / persistence (delegate to the underlying Keras model) --- #
    def compile(self, *args, **kwargs):
        return self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    def save_weights(self, path):
        return self.model.save_weights(path)

    def load_weights(self, path):
        return self.model.load_weights(path)

    def __repr__(self):
        return (f"KerasDenseAutoencoder(input_dim={self.input_dim}, "
                f"sequence_length={self.sequence_length}, "
                f"params={self.model.count_params():,})")


def create_dense_autoencoder_keras(input_dim, sequence_length, hidden_dim,
                                   latent_dim, dropout_rate=0.0):
    """Build and return an untrained KerasDenseAutoencoder.

    The architecture matches the PyTorch ``DenseAutoencoder``: flatten the
    window, compress through an undercomplete ``latent_dim`` bottleneck, then
    expand back and reshape to ``(sequence_length, input_dim)``.
    """
    flat_dim = input_dim * sequence_length

    inputs = keras.Input(shape=(sequence_length, input_dim), name="input")
    x = layers.Flatten(name="flatten")(inputs)
    x = layers.Dense(hidden_dim, activation="relu", name="enc_dense_1")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="enc_dropout")(x)
    x = layers.Dense(hidden_dim, activation="relu", name="enc_dense_2")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    x = layers.Dense(hidden_dim, activation="relu", name="dec_dense_1")(latent)
    x = layers.Dense(hidden_dim, activation="relu", name="dec_dense_2")(x)
    x = layers.Dense(flat_dim, name="dec_output")(x)
    outputs = layers.Reshape((sequence_length, input_dim), name="reshape")(x)

    autoencoder = keras.Model(inputs, outputs, name="dense_autoencoder")
    # Shares the same layer objects (and therefore weights) as the autoencoder.
    encoder = keras.Model(inputs, latent, name="encoder")

    return KerasDenseAutoencoder(autoencoder, encoder, input_dim, sequence_length)


def plot_autoencoder_architecture_keras(model, results_path=None):
    """Plot layer dimensions (bottleneck) and parameter counts for a Keras model.

    Keras counterpart of ``plotting.plot_autoencoder_architecture`` — accepts
    either a ``KerasDenseAutoencoder`` wrapper or a raw ``keras.Model``.
    """
    import os

    import matplotlib.pyplot as plt

    keras_model = getattr(model, "model", model)
    input_dim = getattr(model, "input_dim", None)
    sequence_length = getattr(model, "sequence_length", None)

    # Layers that actually carry weights (Dense); skip Flatten/Dropout/Reshape.
    weighted = [l for l in keras_model.layers if l.count_params() > 0]
    layer_names = [l.name for l in weighted]
    layer_params = [l.count_params() for l in weighted]
    units = [l.units for l in weighted]

    flat_dim = (input_dim * sequence_length
                if input_dim is not None else units[-1])
    arch_dims = [flat_dim] + units
    arch_labels = ["input"] + layer_names

    colors = ["lightgreen" if ("enc" in n or "latent" in n) else "lightsalmon"
              for n in layer_names]

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

    if results_path:
        os.makedirs(results_path, exist_ok=True)
        fig.savefig(os.path.join(results_path, "05_model_architecture.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
