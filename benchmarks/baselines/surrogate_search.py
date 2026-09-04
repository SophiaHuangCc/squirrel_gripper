"""Adam and CMA-ES proposal baselines using the frozen three-output surrogate."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from benchmarks.protocol import expand_core_scenarios, expand_physical_conditions
from dynamics.profile_forward_2d import ProfileForward2DModel
from generator.dataloader import DesignBounds, physical_to_model_norm, project_physical_design


@dataclass
class SearchResult:
    designs: np.ndarray
    scores: np.ndarray
    model_evaluations: int
    target_scenario_ids: list[str]


class MetricScaleWrapper(torch.nn.Module):
    """Expose checkpoint predictions in benchmark C/D/A coordinates."""

    def __init__(self, model, angular_mode="none", angular_mean=0.0, angular_std=1.0):
        super().__init__()
        self.model = model
        self.angular_mode = angular_mode
        self.angular_mean = float(angular_mean)
        self.angular_std = float(angular_std)

    def forward(self, task_params, design_params, init_config, timesteps):
        prediction = self.model(task_params, design_params, init_config, timesteps)
        if self.angular_mode != "zscore":
            return prediction
        return torch.cat(
            (
                prediction[..., :2],
                prediction[..., 2:3] * self.angular_std + self.angular_mean,
            ),
            dim=-1,
        )


def select_target_cells(config, scenario_id=None, family=None, generalist=False):
    cells = expand_core_scenarios(config)
    selectors = sum(value is not None and value is not False for value in (scenario_id, family, generalist))
    if selectors > 1:
        raise ValueError("Choose only one of scenario_id, family, or generalist")
    if scenario_id:
        cells = [cell for cell in cells if cell["scenario_id"] == scenario_id]
    elif family:
        cells = [cell for cell in cells if cell["family"] == family]
    elif not generalist:
        default_id = config.get("default_target_scenario_id")
        if default_id:
            cells = [cell for cell in cells if cell["scenario_id"] == default_id]
        else:
            cells = [cell for cell in cells if cell["family"] == "nominal"]
    if not cells:
        raise ValueError("The surrogate-search target selected zero scenarios")
    return expand_physical_conditions(cells, config)


def load_surrogate(checkpoint_path, device="cpu", hidden_dim=256, expected_noise_conditioned=False):
    state = torch.load(checkpoint_path, map_location=device)
    metadata = state if isinstance(state, dict) and "model" in state else {}
    checkpoint_args = metadata.get("args", {}) if metadata else {}
    architecture = metadata.get(
        "model_architecture", checkpoint_args.get("model_architecture", "legacy")
    )
    model_width = int(metadata.get(
        "hidden_dim", checkpoint_args.get("hidden_dim", hidden_dim)
    ))
    num_hidden_layers = int(metadata.get(
        "num_hidden_layers", checkpoint_args.get("num_hidden_layers", 3)
    ))
    model = ProfileForward2DModel(
        W=model_width, task_ch=3, design_ch=16, init_ch=3, output_ch=3,
        architecture=architecture, num_hidden_layers=num_hidden_layers,
    ).to(device)
    noise_conditioned = bool(metadata.get("noise_conditioned", False))
    if noise_conditioned != bool(expected_noise_conditioned):
        wanted = "noise-conditioned DGDM" if expected_noise_conditioned else "clean-design"
        found = "noise-conditioned DGDM" if noise_conditioned else "clean-design/legacy"
        raise ValueError(f"Expected a {wanted} dynamics checkpoint, but {checkpoint_path} is {found}.")
    if metadata:
        state = metadata["model"]
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(
            "The dynamics checkpoint is not compatible with the current "
            "three-output model. Retrain dynamics with --output_dim 3."
        ) from exc
    angular_mode = metadata.get("angular_target_normalization", "none")
    if angular_mode == "zscore":
        model = MetricScaleWrapper(
            model,
            angular_mode=angular_mode,
            angular_mean=metadata.get("angular_target_mean", 0.0),
            angular_std=metadata.get("angular_target_std", 1.0),
        ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.noise_conditioned = noise_conditioned
    model.num_train_timesteps = int(metadata.get("num_train_timesteps", 100))
    model.design_coordinates = metadata.get(
        "design_coordinates", "diffusion_unit" if noise_conditioned else "model_norm"
    )
    return model


def _scenario_tensors(cells, device):
    task = torch.tensor(
        [[cell["params"]["approach_deg"] / 90.0,
          cell["params"]["landing_approach_deg"] / 90.0,
          cell["params"]["cyl_rad"] / 0.05] for cell in cells],
        dtype=torch.float32,
        device=device,
    )
    init = torch.tensor(
        [[
            cell["params"]["landing_height"] / 0.10,
            cell["params"]["landing_speed"] / 1.0,
            cell["params"]["initial_x_gap"] / 0.10,
        ] for cell in cells],
        dtype=torch.float32,
        device=device,
    )
    return task, init


def _raw_to_design(raw, bounds):
    lo = bounds.lo.to(raw.device)
    hi = bounds.hi.to(raw.device)
    physical = lo + torch.sigmoid(raw) * (hi - lo)
    return project_physical_design(physical, bounds)


def _candidate_scores(model, designs, task, init, weights):
    candidate_count, scenario_count = designs.shape[0], task.shape[0]
    design_batch = physical_to_model_norm(designs).repeat_interleave(scenario_count, dim=0)
    task_batch = task.repeat(candidate_count, 1)
    init_batch = init.repeat(candidate_count, 1)
    timesteps = torch.zeros(candidate_count * scenario_count, device=designs.device)
    pred = model(task_batch, design_batch, init_batch, timesteps).reshape(candidate_count, scenario_count, 3)
    # C, D, and A are normalized benchmark quantities.  The regression model
    # has a linear output head, so modest extrapolation outside [0, 1] is
    # possible, especially after gradient-based design search.  Score the
    # prediction under the same bounded contract used by the simulator rather
    # than allowing an impossible value (for example, contact coverage > 1) to
    # dominate candidate selection.
    pred = pred.clamp(0.0, 1.0)
    per_cell = (
        float(weights["contact_coverage_norm"]) * pred[..., 0]
        + float(weights["disturbance_resistance_score"]) * pred[..., 1]
        + float(weights["angular_span_norm"]) * pred[..., 2]
    )
    return per_cell.mean(dim=1)


def rank_designs(
    model, designs, config, scenario_id=None, family=None, generalist=False,
    device="cpu",
):
    """Rank a pre-generated physical design pool without test simulation."""
    cells = select_target_cells(config, scenario_id, family, generalist)
    task, init = _scenario_tensors(cells, device)
    design_tensor = torch.as_tensor(designs, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = _candidate_scores(
            model, design_tensor, task, init, config["evaluation"]["utility_weights"]
        )
    order = torch.argsort(scores, descending=True)
    return SearchResult(
        designs=design_tensor[order].cpu().numpy().astype(np.float32),
        scores=scores[order].cpu().numpy().astype(np.float32),
        model_evaluations=len(designs) * len(cells),
        target_scenario_ids=[cell["scenario_id"] for cell in cells],
    )


def adam_search(
    model, config, num_candidates, seed, num_steps=300, learning_rate=0.03,
    scenario_id=None, family=None, generalist=False, device="cpu",
):
    cells = select_target_cells(config, scenario_id, family, generalist)
    task, init = _scenario_tensors(cells, device)
    bounds = DesignBounds.defaults()
    generator = torch.Generator(device=device).manual_seed(seed)
    raw = torch.nn.Parameter(torch.randn(num_candidates, 16, generator=generator, device=device))
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    weights = config["evaluation"]["utility_weights"]
    best_scores = torch.full((num_candidates,), -torch.inf, device=device)
    best_designs = None
    for _ in range(num_steps):
        designs = _raw_to_design(raw, bounds)
        scores = _candidate_scores(model, designs, task, init, weights)
        with torch.no_grad():
            improved = scores > best_scores
            if best_designs is None:
                best_designs = designs.detach().clone()
            best_designs[improved] = designs.detach()[improved]
            best_scores[improved] = scores.detach()[improved]
        optimizer.zero_grad()
        (-scores.mean()).backward()
        optimizer.step()
    order = torch.argsort(best_scores, descending=True)
    return SearchResult(
        designs=best_designs[order].cpu().numpy().astype(np.float32),
        scores=best_scores[order].cpu().numpy().astype(np.float32),
        model_evaluations=num_candidates * len(cells) * num_steps,
        target_scenario_ids=[cell["scenario_id"] for cell in cells],
    )


def cma_es_search(
    model, config, num_candidates, seed, num_generations=100, popsize=32,
    sigma=0.5, scenario_id=None, family=None, generalist=False, device="cpu",
):
    try:
        import cma
    except ImportError as exc:
        raise ImportError("CMA-ES requires the optional 'cma' package") from exc
    cells = select_target_cells(config, scenario_id, family, generalist)
    task, init = _scenario_tensors(cells, device)
    bounds = DesignBounds.defaults()
    weights = config["evaluation"]["utility_weights"]
    strategy = cma.CMAEvolutionStrategy(
        np.zeros(16), sigma,
        {"seed": seed, "popsize": max(popsize, num_candidates), "bounds": [-5.0, 5.0], "verbose": -9},
    )
    archive = {}
    evaluations = 0
    for _ in range(num_generations):
        solutions = np.asarray(strategy.ask(), dtype=np.float32)
        raw = torch.from_numpy(solutions).to(device)
        with torch.no_grad():
            designs = _raw_to_design(raw, bounds)
            scores = _candidate_scores(model, designs, task, init, weights)
        score_np = scores.cpu().numpy()
        strategy.tell(solutions.tolist(), (-score_np).tolist())
        evaluations += len(solutions) * len(cells)
        for design, score in zip(designs.cpu().numpy(), score_np):
            key = tuple(np.round(design, 7))
            if key not in archive or score > archive[key][1]:
                archive[key] = (design.astype(np.float32), float(score))
    ranked = sorted(archive.values(), key=lambda row: row[1], reverse=True)[:num_candidates]
    return SearchResult(
        designs=np.asarray([row[0] for row in ranked], dtype=np.float32),
        scores=np.asarray([row[1] for row in ranked], dtype=np.float32),
        model_evaluations=evaluations,
        target_scenario_ids=[cell["scenario_id"] for cell in cells],
    )
