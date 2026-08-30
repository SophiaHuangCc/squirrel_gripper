import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dynamics.dataloader import DynamicsDataset


DESIGN_NAMES = [
    "joint_softness_0",
    "joint_softness_1",
    "joint_softness_2",
    "link_length_0",
    "link_length_1",
    "link_length_2",
    "link_length_3",
    "joint_length_0",
    "joint_length_1",
    "joint_length_2",
    "base_radius",
    "base_length",
    "tension",
    "ankle_wrap_radius",
    "ankle_stiffness",
]

DESIGN_MODEL_SCALES = torch.tensor(
    [0.001, 0.001, 0.001, 0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.02, 0.2, 10.0, 0.025, 1000.0],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class DesignBounds:
    """Physical design bounds used to map squirrel designs to diffusion [-1, 1]."""

    lo: torch.Tensor
    hi: torch.Tensor

    @classmethod
    def defaults(cls) -> "DesignBounds":
        return cls(
            lo=torch.tensor(
                [0.0005, 0.0005, 0.0005, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.01025, 0.15, 1.0, 0.015, 300.0],
                dtype=torch.float32,
            ),
            hi=torch.tensor(
                [0.05, 0.05, 0.05, 0.13, 0.10, 0.10, 0.10, 0.03, 0.03, 0.03, 0.013, 0.30, 30.0, 0.035, 700.0],
                dtype=torch.float32,
            ),
        )

    @classmethod
    def from_npz(cls, path: str) -> "DesignBounds":
        data = np.load(path)
        bounds = cls(
            lo=torch.from_numpy(data["design_lo"].astype(np.float32)),
            hi=torch.from_numpy(data["design_hi"].astype(np.float32)),
        )
        if bounds.lo.numel() != len(DESIGN_NAMES):
            raise ValueError(
                f"{path} contains {bounds.lo.numel()}D legacy bounds; expected "
                f"{len(DESIGN_NAMES)}D From Links bounds. Retrain the generator."
            )
        return bounds

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            design_lo=self.lo.cpu().numpy(),
            design_hi=self.hi.cpu().numpy(),
            design_names=np.asarray(DESIGN_NAMES),
            design_model_scales=DESIGN_MODEL_SCALES.cpu().numpy(),
        )


def model_norm_to_physical(design_norm: torch.Tensor) -> torch.Tensor:
    """Convert DynamicsDataset's model-normalized design vector to physical units."""
    scales = DESIGN_MODEL_SCALES.to(design_norm.device)
    return design_norm * scales


def physical_to_model_norm(design_physical: torch.Tensor) -> torch.Tensor:
    """Convert physical design vector to the dynamics model normalization."""
    scales = DESIGN_MODEL_SCALES.to(design_physical.device)
    return design_physical / scales


def physical_to_diffusion(design_physical: torch.Tensor, bounds: DesignBounds) -> torch.Tensor:
    lo = bounds.lo.to(design_physical.device)
    hi = bounds.hi.to(design_physical.device)
    return 2.0 * (design_physical - lo) / (hi - lo).clamp_min(1e-12) - 1.0


def diffusion_to_physical(design_unit: torch.Tensor, bounds: DesignBounds) -> torch.Tensor:
    lo = bounds.lo.to(design_unit.device)
    hi = bounds.hi.to(design_unit.device)
    return lo + 0.5 * (design_unit + 1.0) * (hi - lo)


def project_physical_design(design_physical: torch.Tensor, bounds: DesignBounds) -> torch.Tensor:
    """
    Clamp designs and make free link lengths plus joint lengths sum to base_length.

    Your simulator expects physically consistent vertebra locations. The dataset
    free links plus finite joints always sum to base_length; raw diffusion
    samples may not, so this projection restores a valid From Links geometry.
    """
    lo = bounds.lo.to(design_physical.device)
    hi = bounds.hi.to(design_physical.device)
    design = torch.clamp(design_physical, lo, hi)

    links = design[..., 3:7]
    joints = design[..., 7:10]
    base_length = design[..., 11:12]
    link_lo = lo[3:7]
    link_hi = hi[3:7]
    links = torch.clamp(links, link_lo, link_hi)
    available_link_length = (base_length - joints.sum(dim=-1, keepdim=True)).clamp_min(1e-6)
    links = links / links.sum(dim=-1, keepdim=True).clamp_min(1e-12) * available_link_length
    design = torch.cat([design[..., :3], links, design[..., 7:]], dim=-1)
    return design


def build_condition(
    task_params: torch.Tensor,
    init_config: torch.Tensor,
    target_metrics: torch.Tensor,
    metric_mask: Optional[Iterable[float]] = None,
) -> torch.Tensor:
    """
    Create the global condition vector.

    The condition is intentionally small and matches your trained dynamics model:
    task_params(2), init_config(3), desired_metrics(3).

    metric_mask can zero out unknown metric targets. By default all target
    metrics are used.
    """
    if metric_mask is not None:
        mask = torch.tensor(list(metric_mask), dtype=target_metrics.dtype, device=target_metrics.device)
        target_metrics = target_metrics * mask
    return torch.cat([task_params, init_config, target_metrics], dim=-1)


class SquirrelDiffusionDataset(Dataset):
    """
    Dataset for diffusion training.

    It reuses DynamicsDataset so diffusion and the dynamics model see the exact
    same parsing conventions. Returned sample:
      design_unit: (15, 1), normalized to [-1, 1]
      cond:        (8,), task/init/target metric condition
    """

    def __init__(
        self,
        dataset_dir: str,
        bounds: Optional[DesignBounds] = None,
        metric_mask: Optional[Iterable[float]] = None,
    ):
        self.base = DynamicsDataset(dataset_dir=dataset_dir)
        self.bounds = bounds or DesignBounds.defaults()
        self.metric_mask = metric_mask

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        design_physical = model_norm_to_physical(item["design_params"].float())
        design_unit = physical_to_diffusion(design_physical, self.bounds).clamp(-1.0, 1.0)
        cond = build_condition(
            item["task_params"].float(),
            item["init_config"].float(),
            item["target_metrics"].float(),
            metric_mask=self.metric_mask,
        )
        return {
            "design_unit": design_unit.reshape(-1, 1),
            "cond": cond,
            "task_params": item["task_params"].float(),
            "init_config": item["init_config"].float(),
            "target_metrics": item["target_metrics"].float(),
        }
