import numpy as np

import torch
import torch.nn as nn

maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1868

class Encoder(nn.Module):
    """ Generate latent parameters for sentence data. """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
        self.enc = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            nn.Conv2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=False)
        self.c2 = nn.Conv2d(fBase * 4, latent_dim, 4, 1, 0, bias=False)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(self.embedding(x.long()).unsqueeze(1))
        mu, logvar = self.c1(e).squeeze(), self.c2(e).squeeze()
        # return mu, F.softplus(logvar) + 1e-6
        return mu, logvar

class Decoder(nn.Module):
    """ Generate a sentence given a sample from the latent space. """

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fBase * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=False),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )
        # inverts the 'embedding' module upto one-hotness
        self.toVocabSize = nn.Linear(embeddingDim, vocabSize)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:])).view(-1, embeddingDim)

        x_hat = self.toVocabSize(out).view(*z.size()[:-3], maxSentLen, vocabSize)
        return x_hat, torch.tensor(0.75).to(
            z.device
        ) 
