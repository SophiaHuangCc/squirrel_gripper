"""Objective tables and preference-pair datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dynamics.dataloader import DynamicsDataset
from generator.dataloader import DesignBounds, model_norm_to_physical, physical_to_diffusion
from .core import OBJECTIVE_NAMES, build_preference_pairs, crowding_distance, non_dominated_sort


def design_id(design, decimals=7):
    rounded = np.round(np.asarray(design, dtype=np.float64), decimals)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]


def split_for_id(identifier, val_fraction=0.1, test_fraction=0.1, seed=0):
    digest = hashlib.sha1(f"{seed}:{identifier}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


@dataclass
class ObjectiveTable:
    designs: np.ndarray
    objectives: np.ndarray
    feasible: np.ndarray
    violation: np.ndarray
    design_ids: np.ndarray
    splits: np.ndarray
    scenario_counts: np.ndarray
    metadata: dict

    def validate(self):
        n = len(self.designs)
        if self.designs.ndim != 2 or self.objectives.shape != (n, 3):
            raise ValueError("Invalid design/objective table shapes")
        for values in (self.feasible, self.violation, self.design_ids, self.splits, self.scenario_counts):
            if len(values) != n:
                raise ValueError("Objective table columns have inconsistent lengths")
        if not np.all(np.isfinite(self.designs)) or not np.all(np.isfinite(self.objectives)):
            raise ValueError("Objective table contains non-finite values")
        return self


def save_table(path, table: ObjectiveTable):
    table.validate(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, designs=table.designs.astype(np.float32), objectives=table.objectives.astype(np.float32),
        feasible=table.feasible.astype(bool), violation=table.violation.astype(np.float32),
        design_ids=table.design_ids.astype(str), splits=table.splits.astype(str),
        scenario_counts=table.scenario_counts.astype(np.int32), objective_names=np.asarray(OBJECTIVE_NAMES),
        metadata_json=np.asarray([json.dumps(table.metadata, sort_keys=True)]))
    return path


def load_table(path):
    with np.load(path, allow_pickle=False) as z:
        names = tuple(np.asarray(z["objective_names"], dtype=str))
        if names != OBJECTIVE_NAMES:
            raise ValueError(f"Objective order {names} does not match {OBJECTIVE_NAMES}")
        table = ObjectiveTable(z["designs"], z["objectives"], z["feasible"], z["violation"],
            z["design_ids"].astype(str), z["splits"].astype(str), z["scenario_counts"],
            json.loads(str(z["metadata_json"][0])))
    return table.validate()


def build_table(dataset_dir, val_fraction=0.1, test_fraction=0.1, split_seed=0,
                min_scenarios=1, max_failure_rate=0.0):
    """Aggregate the three simulator outcomes by design; curl metrics are intentionally absent."""
    dataset = DynamicsDataset(dataset_dir=str(dataset_dir))
    groups = {}
    for index, path in enumerate(dataset.files):
        try:
            item = dataset[index]
            physical = model_norm_to_physical(item["design_params"]).numpy()
            identifier = design_id(physical)
            with np.load(path, allow_pickle=True) as z:
                required = ("disturbance_resistance_score", "num_contacts", "angular_span")
                ok = all(key in z and np.asarray(z[key]).size for key in required)
                if ok:
                    n_elements = max(float(np.asarray(z.get("n_elements", [100])).reshape(-1)[0]), 1.0)
                    objective = np.asarray([
                        float(np.asarray(z["disturbance_resistance_score"]).reshape(-1)[0]),
                        np.log1p(float(np.asarray(z["num_contacts"]).reshape(-1)[0])) / np.log1p(n_elements),
                        np.clip(float(np.asarray(z["angular_span"]).reshape(-1)[0]) / 360.0, 0.0, 1.0),
                    ], dtype=np.float64)
                    ok = np.all(np.isfinite(objective))
                group = groups.setdefault(identifier, {"design": physical, "objectives": [], "failures": 0, "total": 0})
                group["total"] += 1
                if ok:
                    group["objectives"].append(np.clip(objective, 0.0, 1.0))
                else:
                    group["failures"] += 1
        except (KeyError, ValueError, OSError, IndexError):
            continue
    rows = []
    for identifier, group in groups.items():
        count = len(group["objectives"]); failure_rate = group["failures"] / max(group["total"], 1)
        if count < min_scenarios:
            continue
        rows.append((identifier, group["design"], np.mean(group["objectives"], axis=0),
                     failure_rate <= max_failure_rate, failure_rate, count))
    if len(rows) < 2:
        raise ValueError("Need at least two designs with complete three-objective outcomes")
    return ObjectiveTable(
        designs=np.stack([r[1] for r in rows]), objectives=np.stack([r[2] for r in rows]),
        feasible=np.asarray([r[3] for r in rows]), violation=np.asarray([r[4] for r in rows]),
        design_ids=np.asarray([r[0] for r in rows]),
        splits=np.asarray([split_for_id(r[0], val_fraction, test_fraction, split_seed) for r in rows]),
        scenario_counts=np.asarray([r[5] for r in rows]),
        metadata={"dataset_dir": str(Path(dataset_dir).resolve()), "aggregation": "mean",
                  "split_seed": split_seed, "val_fraction": val_fraction, "test_fraction": test_fraction,
                  "min_scenarios": min_scenarios, "max_failure_rate": max_failure_rate},
    ).validate()


class PreferencePairDataset(Dataset):
    def __init__(self, table: ObjectiveTable, split="train", bounds=None, max_pairs=200000, seed=0):
        ids = np.flatnonzero(table.splits == split)
        if len(ids) < 2:
            raise ValueError(f"Split {split!r} needs at least two designs")
        objectives, feasible, violation = table.objectives[ids], table.feasible[ids], table.violation[ids]
        ranks = non_dominated_sort(objectives, feasible, violation)
        crowding = crowding_distance(objectives, ranks)
        local_pairs = build_preference_pairs(ranks, crowding, max_pairs=max_pairs, seed=seed)
        self.pairs = local_pairs
        self.designs = torch.from_numpy(table.designs[ids].astype(np.float32))
        self.bounds = bounds or DesignBounds.defaults()
        self.design_units = physical_to_diffusion(self.designs, self.bounds).clamp(-1, 1)
        self.ranks, self.crowding = ranks, crowding

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a, b, label = self.pairs[index]
        return {"design_a": self.design_units[int(a)], "design_b": self.design_units[int(b)],
                "label": torch.tensor(label, dtype=torch.float32)}
