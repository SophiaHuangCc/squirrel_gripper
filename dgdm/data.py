"""Data adapters for an unconditional geometry prior and reusable interaction profiles."""

import glob
import os
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dynamics.dataloader import DynamicsDataset
from generator.dataloader import DesignBounds, model_norm_to_physical, physical_to_diffusion


PROFILE_CHANNELS = (
    "contact_fraction",
    "tip_x",
    "tip_y",
    "tip_z",
    "ankle_angle",
    "tendon_tension",
    "drag_left_contact_fraction",
    "drag_left_force_x",
    "drag_left_force_y",
    "drag_left_force_z",
    "drag_right_contact_fraction",
    "drag_right_force_x",
    "drag_right_force_y",
    "drag_right_force_z",
    "drag_down_contact_fraction",
    "drag_down_force_x",
    "drag_down_force_y",
    "drag_down_force_z",
)


def _scalar(z, names: Iterable[str], default=0.0) -> float:
    for name in names:
        if name in z:
            value = np.asarray(z[name]).reshape(-1)
            if value.size:
                return float(value[0])
    return float(default)


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.shape[0] == length:
        return values
    if values.shape[0] == 1:
        return np.repeat(values, length, axis=0)
    old = np.linspace(0.0, 1.0, values.shape[0])
    new = np.linspace(0.0, 1.0, length)
    flat = values.reshape(values.shape[0], -1)
    out = np.stack([np.interp(new, old, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return out.reshape((length,) + values.shape[1:]).astype(np.float32)


def extract_profile(z, profile_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract a normalized trajectory; mask denotes genuinely observed channels."""
    profile = np.zeros((profile_steps, len(PROFILE_CHANNELS)), dtype=np.float32)
    mask = np.zeros_like(profile)
    n_elements = max(_scalar(z, ("n_elements", "arg_n_elements"), 100.0), 1.0)

    if "contact_counts" in z:
        profile[:, 0] = _resample(z["contact_counts"], profile_steps) / n_elements
        mask[:, 0] = 1.0
    if "position" in z:
        position = np.asarray(z["position"], dtype=np.float32)
        tip = _resample(position[:, :, -1], profile_steps)
        scale = max(_scalar(z, ("base_length", "arg_base_len"), 0.1), 1e-6)
        profile[:, 1:4] = tip / scale
        mask[:, 1:4] = 1.0
    for key, channel, scale in (("ankle_angle", 4, np.pi), ("current_tension", 5, 10.0)):
        if key in z:
            profile[:, channel] = _resample(z[key], profile_steps) / scale
            mask[:, channel] = 1.0

    for direction, start in (("left", 6), ("right", 10), ("down", 14)):
        contacts = f"disturbance_drag_{direction}_contacts_history"
        forces = f"disturbance_drag_{direction}_force_history"
        if contacts in z:
            profile[:, start] = _resample(z[contacts], profile_steps) / n_elements
            mask[:, start] = 1.0
        if forces in z:
            force = _resample(z[forces], profile_steps)
            force_scale = max(_scalar(z, ("arg_disturbance_force_mag",), 1.0), 1e-6)
            profile[:, start + 1 : start + 4] = force / force_scale
            mask[:, start + 1 : start + 4] = 1.0
    return profile, mask


def extract_scenario(z) -> np.ndarray:
    """Task-independent initial condition/object description used by dynamics only."""
    return np.asarray(
        [
            _scalar(z, ("arg_approach_deg",)) / 90.0,
            _scalar(z, ("arg_landing_approach_deg",)) / 90.0,
            _scalar(z, ("cyl_radius", "arg_cyl_rad"), 0.015) / 0.05,
            _scalar(z, ("arg_landing_height",)) / 0.10,
            _scalar(z, ("arg_landing_speed",)),
            _scalar(z, ("arg_initial_x_gap",)) / 0.10,
            _scalar(z, ("mu_contact", "arg_mu_contact"), 0.5),
            _scalar(z, ("body_mass", "arg_body_mass"), 1.0),
        ],
        dtype=np.float32,
    )


class UnconditionalDesignDataset(Dataset):
    """The valid-geometry prior. No task, metric, or simulation outcome is returned."""

    def __init__(self, dataset_dir: str, bounds: DesignBounds | None = None):
        self.base = DynamicsDataset(dataset_dir=dataset_dir)
        self.bounds = bounds or DesignBounds.defaults()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        physical = model_norm_to_physical(self.base[index]["design_params"].float())
        unit = physical_to_diffusion(physical, self.bounds).clamp(-1.0, 1.0)
        return {"design_unit": unit.reshape(-1, 1)}


class InteractionProfileDataset(Dataset):
    """One full simulator rollout per item for training reusable dynamics."""

    def __init__(self, dataset_dir: str, profile_steps: int = 32):
        candidates = sorted(glob.glob(os.path.join(os.path.abspath(dataset_dir), "**", "*.npz"), recursive=True))
        self.files = []
        for path in candidates:
            try:
                with np.load(path, allow_pickle=True) as z:
                    if "position" in z or "contact_counts" in z:
                        self.files.append(path)
            except (OSError, ValueError):
                continue
        if not self.files:
            raise ValueError(f"No trajectory archives found below {dataset_dir!r}")
        self.profile_steps = int(profile_steps)
        self.designs = DynamicsDataset(dataset_dir=dataset_dir)
        self._design_index = {os.path.abspath(p): i for i, p in enumerate(self.designs.files)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.files[index]
        with np.load(path, allow_pickle=True) as z:
            profile, mask = extract_profile(z, self.profile_steps)
            scenario = extract_scenario(z)
        design = self.designs[self._design_index[os.path.abspath(path)]]["design_params"].float()
        return {
            "design_norm": design,
            "scenario": torch.from_numpy(scenario),
            "profile": torch.from_numpy(profile),
            "profile_mask": torch.from_numpy(mask),
        }
