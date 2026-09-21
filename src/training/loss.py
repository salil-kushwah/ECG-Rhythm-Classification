import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.config import DEVICE

def effective_num_weights(y, n_classes, beta=0.9995):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective, 1e-12)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE), counts

class ClassBalancedCE(nn.Module):
    def __init__(self, weights, smoothing=0.04, gamma=2.0):
        super().__init__()
        self.register_buffer("weights", weights)
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=self.weights,
            label_smoothing=self.smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
