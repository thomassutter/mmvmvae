import numpy as np

import torch
import torch.nn as nn


def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)


class Encoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 7, 7)
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 4, 4)
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            Flatten(),  # -> (2048)
            nn.Linear(2048, self.latent_dim),
            nn.ReLU(),
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

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),  # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),  # -> (128, 4, 4)
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (3, 28, 28)
            # nn.BatchNorm2d(3),
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat, torch.tensor(0.75).to(
            z.device
        )  # NOTE: consider learning scale param, too


class EncoderDist(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, latent_dim, n_groups):
        super(EncoderDist, self).__init__()

        self.latent_dim = latent_dim
        self.n_groups = n_groups
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 7, 7)
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 4, 4)
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            Flatten(),  # -> (2048)
            nn.Linear(2048, latent_dim + n_groups),
            nn.ReLU(),
        )
        # latent representation
        self.log_omega = nn.Linear(latent_dim + n_groups, n_groups)
        self.log_scores = nn.Linear(latent_dim + n_groups, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return (
            self.log_omega(h),
            self.log_scores(h),
        )


class ResnetEncoderDist(nn.Module):
    def __init__(self, cfg):  # , z_dim, size, nfilter=64, nfilter_max=1024, **kwargs):
        super().__init__()
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28
        n_groups = cfg.model.n_groups
        latent_dim = cfg.model.latent_dim

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        # latent representation
        self.log_omega = nn.Linear(self.nf0 * s0 * s0, n_groups)
        self.log_scores = nn.Linear(self.nf0 * s0 * s0, latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        return self.log_omega(out), self.log_scores(out)


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias
        )
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class ResnetEncoder(nn.Module):
    def __init__(self, cfg):  # , z_dim, size, nfilter=64, nfilter_max=1024, **kwargs):
        super().__init__()
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc_mu = nn.Linear(self.nf0 * s0 * s0, cfg.model.latent_dim)
        self.fc_lv = nn.Linear(self.nf0 * s0 * s0, cfg.model.latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        # out = self.fc(actvn(out))
        return self.fc_mu(out), self.fc_lv(out)


class ResnetDecoder(nn.Module):
    def __init__(self, cfg):  # , z_dim, size, nfilter=64, nfilter_max=512, **kwargs):
        super().__init__()
        self.latent_dim = cfg.model.latent_dim

        # NOTE: I've modified/set below variables according to Kieran's suggestions
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(self.latent_dim, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [ResnetBlock(nf0, nf1), nn.Upsample(scale_factor=2)]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    # def forward(self, z):
    def forward(self, z):
        batch_size = z.size(0)
        out = self.fc(z)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        return out, torch.tensor(0.75).to(
            z.device
        )  # NOTE: consider learning scale param, too
        # return torch.sigmoid(out)  # torch.tanh(out), torch.sigmoid(out)
