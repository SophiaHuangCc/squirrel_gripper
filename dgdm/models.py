"""Task-agnostic interaction dynamics surrogate."""

import math
import torch
from torch import nn


class InteractionProfileModel(nn.Module):
    """Predicts an entire interaction trajectory from design and initial condition."""

    def __init__(self, design_dim=16, scenario_dim=8, channels=18, profile_steps=32, width=256):
        super().__init__()
        self.channels = int(channels)
        self.profile_steps = int(profile_steps)
        self.encoder = nn.Sequential(
            nn.Linear(design_dim + scenario_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.time = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.diffusion_time = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.decoder = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, channels))

    def forward(self, design: torch.Tensor, scenario: torch.Tensor,
                diffusion_timestep: torch.Tensor | None = None) -> torch.Tensor:
        latent = self.encoder(torch.cat((design, scenario), dim=-1))
        if diffusion_timestep is None:
            diffusion_timestep = torch.zeros(design.shape[0], device=design.device, dtype=design.dtype)
        tau = diffusion_timestep.to(design).reshape(-1)
        tau_features = torch.stack(
            (tau, torch.sin(2 * math.pi * tau), torch.cos(2 * math.pi * tau)), dim=-1
        )
        latent = latent + self.diffusion_time(tau_features)
        t = torch.linspace(0.0, 1.0, self.profile_steps, device=design.device, dtype=design.dtype)
        time_features = torch.stack((t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)), dim=-1)
        hidden = latent[:, None, :] + self.time(time_features)[None, :, :]
        return self.decoder(hidden)


def masked_profile_loss(prediction, target, mask):
    error = (prediction - target).square() * mask
    return error.sum() / mask.sum().clamp_min(1.0)
