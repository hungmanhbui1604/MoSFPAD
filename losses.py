import torch
import torch.nn as nn
import torch.nn.functional as F


class SVCHingeLoss(nn.Module):
    def __init__(self, C=1.0):
        super().__init__()
        self.C = C

    def forward(self, outputs, targets):
        # Convert labels from {0, 1} to {-1, 1} if needed
        if targets.min() >= 0:
            targets = targets * 2.0 - 1.0

        losses = F.relu(1 - targets * outputs)
        return self.C * losses.mean()