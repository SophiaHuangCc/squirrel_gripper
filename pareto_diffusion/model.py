"""Diffusion-time-conditioned antisymmetric pairwise preference classifier."""

import torch
from torch import nn
from generator.diffusion_utils import SinusoidalPosEmb


class PreferenceClassifier(nn.Module):
    def __init__(self, design_dim=16, width=256, time_dim=64, dropout=0.1):
        super().__init__()
        self.design_dim, self.width, self.time_dim = int(design_dim), int(width), int(time_dim)
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        # Subtracting swapped-order outputs makes C(a,b,t) = -C(b,a,t) exactly.
        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * design_dim + width, width), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, 1),
        )

    def _ordered_score(self, a, b, time):
        features = torch.cat((a, b, a - b, (a - b).abs(), time), dim=-1)
        return self.pair_encoder(features).squeeze(-1)

    def forward(self, design_a, design_b, timestep):
        if design_a.shape != design_b.shape or design_a.ndim != 2:
            raise ValueError("design_a and design_b must both have shape (B, design_dim)")
        if timestep.ndim == 0:
            timestep = timestep.expand(design_a.shape[0])
        timestep = timestep.reshape(-1).expand(design_a.shape[0])
        time = self.time_encoder(timestep)
        return self._ordered_score(design_a, design_b, time) - self._ordered_score(design_b, design_a, time)
