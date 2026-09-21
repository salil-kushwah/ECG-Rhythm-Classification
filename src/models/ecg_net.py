import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.blocks import MultiScaleResBlock, DownBlock

class ECGRhythmNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.stage1 = nn.Sequential(
            MultiScaleResBlock(64, 1),
            MultiScaleResBlock(64, 1),
            MultiScaleResBlock(64, 2)
        )
        self.down1 = DownBlock(64, 96)
        
        self.stage2 = nn.Sequential(
            MultiScaleResBlock(96, 1),
            MultiScaleResBlock(96, 2),
            MultiScaleResBlock(96, 3)
        )
        self.down2 = DownBlock(96, 160)
        
        self.stage3 = nn.Sequential(
            MultiScaleResBlock(160, 1),
            MultiScaleResBlock(160, 2),
            MultiScaleResBlock(160, 4)
        )
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=160,
            nhead=8,
            dim_feedforward=320,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=3)
        self.positional_embedding = nn.Parameter(torch.zeros(1, 64, 160))
        self.token_score = nn.Sequential(
            nn.LayerNorm(160),
            nn.Linear(160, 1)
        )
        
        self.timing_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 64),
            nn.GELU()
        )
        
        self.stats_mlp = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU()
        )
        
        self.timing_to_signal = nn.Linear(64, 160)
        self.gate = nn.Sequential(
            nn.Linear(224, 160),
            nn.GELU(),
            nn.Linear(160, 160),
            nn.Sigmoid()
        )
        
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, n_classes)
        )

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def signal_statistics(x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        maximum = x.amax(dim=-1)
        minimum = x.amin(dim=-1)
        energy = x.square().mean(dim=-1).sqrt()
        diff = x[..., 1:] - x[..., :-1]
        mean_abs_diff = diff.abs().mean(dim=-1)
        max_abs = x.abs().amax(dim=-1)
        zero_crossings = (x[..., 1:] * x[..., :-1] < 0).float().mean(dim=-1)
        
        return torch.cat(
            [mean, std, maximum, minimum, energy, mean_abs_diff, max_abs, zero_crossings],
            dim=1
        )

    def forward(self, batch):
        signal = batch["signal"]
        derivative = signal[..., 1:] - signal[..., :-1]
        derivative = F.pad(derivative, (1, 0))
        x = torch.cat([signal, derivative], dim=1)
        raw_signal = signal

        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)

        x = x.transpose(1, 2)
        x = x + self.positional_embedding[:, :x.shape[1]]
        x = self.transformer(x)

        scores = self.token_score(x).squeeze(-1)
        attention = torch.softmax(scores, dim=1)
        signal_embedding = (x * attention.unsqueeze(-1)).sum(dim=1)
        
        timing_embedding = self.timing_mlp(batch["timing"])
        
        gate_input = torch.cat([signal_embedding, timing_embedding], dim=1)
        gate = self.gate(gate_input)
        
        signal_embedding = signal_embedding + gate * self.timing_to_signal(timing_embedding)
        stats_embedding = self.stats_mlp(self.signal_statistics(raw_signal))
        
        features = torch.cat([signal_embedding, timing_embedding, stats_embedding], dim=1)
        return self.head(features)
