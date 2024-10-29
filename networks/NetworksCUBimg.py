import numpy as np

import torch
import torch.nn as nn

imgChans = 3
fBase = 64

class Encoder(nn.Module):
    """ Generate latent parameters for CUB image data. """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        modules = [
            # input size: 3 x 128 x 128
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # input size: 1 x 64 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 16 x 16
            nn.Conv2d(fBase * 4, fBase * 8, 4, 2, 1, bias=True),
            nn.ReLU(True)]
        # size: (fBase * 8) x 4 x 4

        self.enc = nn.Sequential(*modules)
        self.c1 = nn.Conv2d(fBase * 8, latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 8, latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        mu, logvar = self.c1(e).squeeze(), self.c2(e).squeeze()
        return mu, logvar

class Decoder(nn.Module):
    """ Generate an image given a sample from the latent space. """

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        modules = [nn.ConvTranspose2d(latent_dim, fBase * 8, 4, 1, 0, bias=True),
                   nn.ReLU(True), ]

        modules.extend([
            nn.ConvTranspose2d(fBase * 8, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 16 x 16
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 32
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 64 x 64
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 128 x 128
        ])
        self.dec = nn.Sequential(*modules)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        return out, torch.tensor(0.75).to(
            z.device
        ) 
