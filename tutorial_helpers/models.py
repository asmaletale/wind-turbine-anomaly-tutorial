"""
models.py
=========
PyTorch dense autoencoder for the wind-turbine anomaly-detection tutorial.
"""

import numpy as np
import torch
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    """Fully-connected autoencoder for sequence data.

    The input (batch, sequence_length, input_dim) is flattened, compressed
    through an undercomplete bottleneck of size latent_dim, then expanded back
    and reshaped to the original sequence shape.
    """

    def __init__(self, input_dim, sequence_length, hidden_dim, latent_dim,
                 dropout_rate=0.0):
        super().__init__()
        flat_dim = input_dim * sequence_length
        self.input_dim = input_dim
        self.sequence_length = sequence_length

        enc = [
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim), nn.ReLU(),
        ]
        if dropout_rate > 0:
            enc.append(nn.Dropout(dropout_rate))
        enc += [
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  nn.ReLU(),
        ]
        self.encoder = nn.Sequential(*enc)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  nn.ReLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z).view(-1, self.sequence_length, self.input_dim)

    def predict(self, x):
        """NumPy array in → NumPy array out (mirrors Keras model.predict)."""
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            t = torch.tensor(np.asarray(x), dtype=torch.float32).to(device)
            return self(t).cpu().numpy()

    def encode(self, x):
        """NumPy array in → latent vectors out, shape (n, latent_dim)."""
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            t = torch.tensor(np.asarray(x), dtype=torch.float32).to(device)
            return self.encoder(t).cpu().numpy()


def create_dense_autoencoder(input_dim, sequence_length, hidden_dim, latent_dim,
                              dropout_rate=0.0):
    """Build and return an untrained DenseAutoencoder."""
    return DenseAutoencoder(input_dim, sequence_length, hidden_dim, latent_dim,
                            dropout_rate)
