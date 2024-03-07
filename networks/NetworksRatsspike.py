import numpy as np

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, latent_dim, original_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim # default 2
        self.original_dim = original_dim # vary from 46*10 to 104*10, e.g. mitt 104*10
        self.encoder = nn.Sequential(
            nn.Linear(self.original_dim, self.original_dim),
            nn.LeakyReLU(),
            nn.Linear(self.original_dim, self.original_dim),
            nn.LeakyReLU(),
            nn.Linear(self.original_dim, self.latent_dim),
            nn.LeakyReLU(),
        )
        # latent representation
        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return (
            self.mu(h),
            self.logvar(h),
        )


class Decoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, latent_dim, original_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.original_dim = original_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.original_dim),
            nn.LeakyReLU(),
            nn.Linear(self.original_dim, self.original_dim),
            nn.LeakyReLU(),
            nn.Linear(self.original_dim, self.original_dim),
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat, torch.tensor(0.75).to(
            z.device
        )  # NOTE: consider learning scale param, too
