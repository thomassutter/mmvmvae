import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

imgChans = 3
fBase = 64
num_attributes = 6


# class ClfImg(nn.Module):
#     def __init__(self):
#         super(ClfImg, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(64 * 2 * 2, 128)
#         self.fc2 = nn.Linear(128, num_attributes)
        
#         # Pooling layer
#         self.pool = nn.MaxPool2d(2, 2)
        
#         # Resize transform
#         self.resize = transforms.Resize((16, 16))

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Resize the input
#         x = self.resize(x)
        
#         # Convolutional layers with ReLU and pooling
#         x = self.pool(F.relu(self.conv1(x)))  # Output: [128, 16, 8, 8]
#         x = self.pool(F.relu(self.conv2(x)))  # Output: [128, 32, 4, 4]
#         x = self.pool(F.relu(self.conv3(x)))  # Output: [128, 64, 2, 2]
        
#         # Flatten the tensor
#         x = x.view(-1, 64 * 2 * 2)
        
#         # Fully connected layers with ReLU
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         out = self.sigmoid(x)
#         return out

class ClfImg(nn.Module):
    def __init__(self):
        super(ClfImg, self).__init__()
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
        self.linear = nn.Conv2d(fBase * 8, 128, 4, 1, 0, bias=True)
        self.linearfinal = nn.Linear(128, num_attributes)
        # linear size: num_attributes x 1 x 1

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.enc(x)
        h = self.linear(h).squeeze()
        h = self.relu(h)
        h = self.linearfinal(h)
        out = self.sigmoid(h)
        return out
        
