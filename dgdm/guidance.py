"""Inference-only task specifications and multi-scenario aggregation."""

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch

from .data import PROFILE_CHANNELS


@dataclass(frozen=True)
class ScenarioBatch:
    values: torch.Tensor  # [S, 7]
    weights: torch.Tensor | None = None

    def normalized_weights(self, device, dtype):
        if self.weights is None:
            return torch.full((self.values.shape[0],), 1.0 / self.values.shape[0], device=device, dtype=dtype)
        weights = self.weights.to(device=device, dtype=dtype).clamp_min(0)
        return weights / weights.sum().clamp_min(1e-12)


@dataclass(frozen=True)
class ProfileTarget:
    target: torch.Tensor  # [T, C]
    mask: torch.Tensor
    weights: torch.Tensor
    loss: str = "mse"

    @classmethod
    def from_dict(cls, spec: Dict[str, Any], steps: int, device=None):
        channels = len(PROFILE_CHANNELS)
        target = torch.zeros(steps, channels, device=device)
        mask = torch.zeros_like(target)
        weights = torch.ones_like(target)
        for name, value in spec.get("channels", {}).items():
            if name not in PROFILE_CHANNELS:
                raise ValueError(f"Unknown profile channel {name!r}; choices: {PROFILE_CHANNELS}")
            idx = PROFILE_CHANNELS.index(name)
            entry = value if isinstance(value, dict) else {"target": value}
            raw = torch.as_tensor(entry["target"], dtype=torch.float32, device=device).flatten()
            if raw.numel() == 1:
                raw = raw.repeat(steps)
            if raw.numel() != steps:
                raise ValueError(f"Target {name!r} has {raw.numel()} values, expected 1 or {steps}")
            target[:, idx] = raw
            mask[:, idx] = torch.as_tensor(entry.get("mask", 1.0), device=device)
            weights[:, idx] = float(entry.get("weight", 1.0))
        if not mask.any():
            raise ValueError("A profile target must enable at least one channel")
        return cls(target, mask, weights, spec.get("loss", "mse"))

    def score(self, prediction: torch.Tensor) -> torch.Tensor:
        delta = prediction - self.target.to(prediction)
        if self.loss == "mse":
            penalty = delta.square()
        elif self.loss == "l1":
            penalty = delta.abs()
        else:
            raise ValueError(f"Unsupported target loss: {self.loss}")
        weighted = penalty * self.mask.to(prediction) * self.weights.to(prediction)
        denom = (self.mask * self.weights).sum().to(prediction).clamp_min(1e-12)
        return -weighted.sum(dim=(-2, -1)) / denom


def aggregate_profile_score(model, designs, scenarios: ScenarioBatch, target: ProfileTarget):
    """Evaluate every design under every initial condition and take weighted expectation."""
    batch, scenario_count = designs.shape[0], scenarios.values.shape[0]
    design_grid = designs[:, None, :].expand(batch, scenario_count, -1).reshape(batch * scenario_count, -1)
    scenario_grid = scenarios.values.to(designs)[None, :, :].expand(batch, -1, -1).reshape(batch * scenario_count, -1)
    profiles = model(design_grid, scenario_grid).reshape(batch, scenario_count, model.profile_steps, model.channels)
    per_scenario = target.score(profiles)
    return (per_scenario * scenarios.normalized_weights(designs.device, designs.dtype)[None, :]).sum(dim=1)
