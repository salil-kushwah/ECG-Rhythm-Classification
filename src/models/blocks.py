import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))

class MultiScaleResBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.08):
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        self.mix = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.se = SEBlock(channels)

    def forward(self, x):
        y = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        y = self.mix(y)
        y = self.se(y)
        return x + y

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.net(x)
