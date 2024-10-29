import numpy as np
import torch
import torch.nn as nn

maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1868
num_attributes = 6

# class ClfText(nn.Module):
#     def __init__(self): 
#         super(ClfText, self).__init__()
#         self.blocks = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, num_attributes),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         out = self.blocks(x)
#         return out

class ClfText(nn.Module):
    def __init__(self):
        super(ClfText, self).__init__()
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
        self.linear = nn.Conv2d(fBase * 4, 128, 4, 1, 0, bias=False)
        self.linearfinal = nn.Linear(128, num_attributes)
        # linear size: num_attributes x 1 x 1
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.enc(self.embedding(x.long()).unsqueeze(1))
        h = self.linear(h).squeeze()
        h = self.relu(h)
        h = self.linearfinal(h)
        out = self.sigmoid(h)
        return out
