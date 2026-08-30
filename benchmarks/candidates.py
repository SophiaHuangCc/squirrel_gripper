"""Common candidate format and adapters for benchmark design methods."""

import json
from pathlib import Path

import numpy as np


DESIGN_DIM = 16


def _text_scalar(value, default=""):
    if value is None:
        return default
    array = np.asarray(value).reshape(-1)
    return default if array.size == 0 else str(array[0])


def validate_designs(designs):
    designs = np.asarray(designs, dtype=np.float32)
    if designs.ndim == 1:
        designs = designs.reshape(1, -1)
    if designs.ndim != 2 or designs.shape[1] != DESIGN_DIM:
        raise ValueError(f"Expected candidate designs with shape (N, {DESIGN_DIM}), got {designs.shape}")
    if not np.all(np.isfinite(designs)):
        raise ValueError("Candidate designs contain non-finite values")
    geometry_total = designs[:, 3:7].sum(axis=1) + designs[:, 7:10].sum(axis=1)
    if not np.allclose(geometry_total, designs[:, 12], rtol=0.0, atol=1e-5):
        raise ValueError("Every candidate must satisfy sum(links) + sum(joints) = base_length")
    return designs


def save_candidates(path, designs, method, seed=0, candidate_ids=None, scores=None, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    designs = validate_designs(designs)
    if candidate_ids is None:
        candidate_ids = [f"{method}_s{seed}_{index:03d}" for index in range(len(designs))]
    candidate_ids = np.asarray(candidate_ids, dtype=str)
    if len(candidate_ids) != len(designs) or len(set(candidate_ids.tolist())) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique and match the number of designs")
    score_array = np.asarray([], dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32)
    np.savez_compressed(
        path,
        design_params=designs,
        candidate_ids=candidate_ids,
        method=np.asarray([method]),
        seed=np.asarray([int(seed)]),
        selection_scores=score_array,
        metadata_json=np.asarray([json.dumps(metadata or {}, sort_keys=True)]),
    )
    return path


def load_candidates(path, method=None, seed=None, top_k=None):
    """Load the common schema or adapt existing generator/optimizer NPZ files."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        if "design_params" not in data:
            raise KeyError(f"{path} does not contain design_params")
        designs = validate_designs(data["design_params"])
        loaded_method = _text_scalar(data.get("method"), path.stem)
        loaded_seed = int(float(_text_scalar(data.get("seed"), "0")))
        ids = (
            np.asarray(data["candidate_ids"], dtype=str)
            if "candidate_ids" in data
            else np.asarray([f"{loaded_method}_s{loaded_seed}_{i:03d}" for i in range(len(designs))])
        )
        scores = None
        metadata = json.loads(_text_scalar(data.get("metadata_json"), "{}"))
        for key in ("selection_scores", "scores"):
            if key in data and np.asarray(data[key]).size == len(designs):
                scores = np.asarray(data[key], dtype=np.float32)
                break
        if "top_ids" in data:
            order = np.asarray(data["top_ids"], dtype=int)
        elif scores is not None:
            order = np.argsort(scores)[::-1]
        else:
            order = np.arange(len(designs))

    if top_k is not None:
        order = order[: min(int(top_k), len(order))]
    return {
        "design_params": designs[order],
        "candidate_ids": ids[order],
        "method": method or loaded_method,
        "seed": loaded_seed if seed is None else int(seed),
        "selection_scores": None if scores is None else scores[order],
        "metadata": metadata,
        "source_path": str(path.resolve()),
    }
