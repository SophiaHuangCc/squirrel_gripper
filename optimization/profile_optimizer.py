import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def sigmoid_to_range(x, lo, hi):
    """Map unconstrained raw parameter to physical range [lo, hi]."""
    return lo + (hi - lo) * torch.sigmoid(x)


def inverse_sigmoid_from_range(x, lo, hi):
    """Initialize raw parameter from a physical value in [lo, hi]."""
    x = np.clip(x, lo + 1e-8, hi - 1e-8)
    y = (x - lo) / (hi - lo)
    return np.log(y / (1.0 - y))


def make_initial_state(
    batch_size,
    seed,
    device,
    init_joint_softness=(0.003, 0.003, 0.003),
    init_link_lengths=(0.06, 0.056, 0.044, 0.04),
    init_base_radius=0.01025,
    init_base_length=0.20,
    init_tension=3.0,
    init_ankle_wrap_radius=0.02,
    init_ankle_stiffness=500.0,
    joint_soft_min=0.0005,
    joint_soft_max=0.005,
    link_min=0.02,
    link_max=0.10,
    base_radius_min=0.01025,
    base_radius_max=0.013,
    base_length_min=0.15,
    base_length_max=0.25,
    tension_min=1.0,
    tension_max=6.0,
    ankle_wrap_min=0.015,
    ankle_wrap_max=0.025,
    ankle_stiff_min=300.0,
    ankle_stiff_max=700.0,
    init_approach_deg=45.0,
    approach_deg_min=0.0,
    approach_deg_max=90.0
):
    """
    Create batch_size copies of a raw 13D optimizer state.

    This mirrors the reference file's random state initialization, but starts from
    a known physical finger design instead of random gripper points.
    """
    raw = []

    for x in init_joint_softness:
        raw.append(inverse_sigmoid_from_range(x, joint_soft_min, joint_soft_max))

    for x in init_link_lengths:
        raw.append(inverse_sigmoid_from_range(x, link_min, link_max))

    raw.append(inverse_sigmoid_from_range(init_base_radius, base_radius_min, base_radius_max))
    raw.append(inverse_sigmoid_from_range(init_base_length, base_length_min, base_length_max))
    raw.append(inverse_sigmoid_from_range(init_tension, tension_min, tension_max))
    raw.append(inverse_sigmoid_from_range(init_ankle_wrap_radius, ankle_wrap_min, ankle_wrap_max))
    raw.append(inverse_sigmoid_from_range(init_ankle_stiffness, ankle_stiff_min, ankle_stiff_max))
    raw.append(inverse_sigmoid_from_range(init_approach_deg, approach_deg_min, approach_deg_max))

    raw = np.asarray(raw, dtype=np.float32)

    rs = np.random.RandomState(seed)
    noise = 0.01 * rs.randn(batch_size, raw.shape[0]).astype(np.float32)
    init_state = raw.reshape(1, -1).repeat(batch_size, axis=0) + noise

    return torch.from_numpy(init_state).float().to(device)


def link_lengths_to_v_list(link_lengths, base_length, n_elements=100):
    """Convert 4 link lengths into 3 vertebra node indices."""
    link_lengths = np.asarray(link_lengths, dtype=np.float32).reshape(-1)
    cum = np.cumsum(link_lengths[:-1])
    joints = np.round(cum / float(base_length) * n_elements).astype(int)
    joints = np.clip(joints, 1, n_elements - 1)
    return joints.tolist()


def design_to_dict(design_params, n_elements=100):
    """Convert one optimized physical design vector into finger.py-friendly values."""
    d = design_params.detach().cpu().numpy().reshape(-1)

    joint_softness = d[0:3]
    link_lengths = d[3:7]
    base_radius = float(d[7])
    base_length = float(d[8])
    tension = float(d[9])
    ankle_wrap_radius = float(d[10])
    ankle_stiffness = float(d[11])

    v_list = link_lengths_to_v_list(link_lengths, base_length, n_elements=n_elements)

    return {
        "joint_softness": joint_softness.tolist(),
        "joint_softness_str": ",".join([f"{x:.6f}" for x in joint_softness]),
        "link_lengths": link_lengths.tolist(),
        "v_list": v_list,
        "v_list_str": ",".join([str(x) for x in v_list]),
        "base_rad": base_radius,
        "base_len": base_length,
        "tension": tension,
        "ankle_wrap_radius": ankle_wrap_radius,
        "ankle_stiffness": ankle_stiffness,
    }


# -----------------------------------------------------------------------------
# Main model wrapper: minimal analog of ProfileOptimizerModel
# -----------------------------------------------------------------------------

class ProfileOptimizerModel(nn.Module):
    def __init__(
        self,
        profile_model: nn.Module,
        batch_size: int = 16,
        num_epochs: int = 1000,
        learning_rate: float = 1e-4,
        opt_obj: str = "disturbance",
        input_dim: int = 13, # 12 for the design parameters + 1 for the task parameter
        num_points: int = 1,
        grid_size: int = 1,
        seed: int = 0,
        device: torch.device = torch.device("cuda:0"),
        # fixed task params
        approach_deg: float = 45.0,
        cyl_radius: float = 0.03,
        landing_height: float = 0.04,
        landing_speed: float = 0.0,
        initial_x_gap: float = 0.06,
        # objective weights
        contact_weight: float = 0.1,
        disturbance_weight: float = 1.0,
        reg_weight: float = 0.0,
        # physical bounds
        joint_soft_min: float = 0.0005,
        joint_soft_max: float = 0.005,
        link_min: float = 0.02,
        link_max: float = 0.10,
        base_radius_min: float = 0.01025,
        base_radius_max: float = 0.013,
        base_length_min: float = 0.15,
        base_length_max: float = 0.25,
        tension_min: float = 1.0,
        tension_max: float = 6.0,
        ankle_wrap_min: float = 0.015,
        ankle_wrap_max: float = 0.025,
        ankle_stiff_min: float = 300.0,
        ankle_stiff_max: float = 700.0,
        # initialization
        init_joint_softness=(0.003, 0.003, 0.003),
        init_link_lengths=(0.06, 0.056, 0.044, 0.04),
        init_base_radius: float = 0.01025,
        init_base_length: float = 0.20,
        init_tension: float = 3.0,
        init_ankle_wrap_radius: float = 0.02,
        init_ankle_stiffness: float = 500.0,
        approach_deg_min: float = 0.0,
        approach_deg_max: float = 90.0,
    ):
        super(ProfileOptimizerModel, self).__init__()

        self.profile_model = profile_model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.opt_obj = opt_obj
        self.input_dim = input_dim
        self.num_points = num_points
        self.grid_size = grid_size
        self.seed = seed
        self.device = device

        self.contact_weight = contact_weight
        self.disturbance_weight = disturbance_weight
        self.reg_weight = reg_weight

        self.joint_soft_min = joint_soft_min
        self.joint_soft_max = joint_soft_max
        self.link_min = link_min
        self.link_max = link_max
        self.base_radius_min = base_radius_min
        self.base_radius_max = base_radius_max
        self.base_length_min = base_length_min
        self.base_length_max = base_length_max
        self.tension_min = tension_min
        self.tension_max = tension_max
        self.ankle_wrap_min = ankle_wrap_min
        self.ankle_wrap_max = ankle_wrap_max
        self.ankle_stiff_min = ankle_stiff_min
        self.ankle_stiff_max = ankle_stiff_max
        self.approach_deg_min = approach_deg_min
        self.approach_deg_max = approach_deg_max

        self.task_params = torch.tensor(
            [[approach_deg, cyl_radius]],
            dtype=torch.float32,
            device=self.device,
        )

        self.init_config = torch.tensor(
            [[landing_height, landing_speed, initial_x_gap]],
            dtype=torch.float32,
            device=self.device,
        )

        # variables: same role as self.state in the reference file
        init_state = make_initial_state(
            batch_size=batch_size,
            seed=seed,
            device=device,
            init_joint_softness=init_joint_softness,
            init_link_lengths=init_link_lengths,
            init_base_radius=init_base_radius,
            init_base_length=init_base_length,
            init_tension=init_tension,
            init_ankle_wrap_radius=init_ankle_wrap_radius,
            init_ankle_stiffness=init_ankle_stiffness,
            joint_soft_min=joint_soft_min,
            joint_soft_max=joint_soft_max,
            link_min=link_min,
            link_max=link_max,
            base_radius_min=base_radius_min,
            base_radius_max=base_radius_max,
            base_length_min=base_length_min,
            base_length_max=base_length_max,
            tension_min=tension_min,
            tension_max=tension_max,
            ankle_wrap_min=ankle_wrap_min,
            ankle_wrap_max=ankle_wrap_max,
            ankle_stiff_min=ankle_stiff_min,
            ankle_stiff_max=ankle_stiff_max,
            approach_deg_min=approach_deg_min,
            approach_deg_max=approach_deg_max
        )
        self.state = nn.Parameter(init_state)
        self.state.requires_grad = True

    def state_to_design(self):
        """
        Convert raw optimizer state into physical design params.
        This is the squirrel analog of converting point state to gripper profile.
        """
        x = self.state

        joint_softness = sigmoid_to_range(x[:, 0:3], self.joint_soft_min, self.joint_soft_max)

        raw_links = sigmoid_to_range(x[:, 3:7], self.link_min, self.link_max)
        base_length = sigmoid_to_range(x[:, 8:9], self.base_length_min, self.base_length_max)
        link_lengths = raw_links / torch.sum(raw_links, dim=-1, keepdim=True) * base_length

        base_radius = sigmoid_to_range(x[:, 7:8], self.base_radius_min, self.base_radius_max)
        tension = sigmoid_to_range(x[:, 9:10], self.tension_min, self.tension_max)
        ankle_wrap_radius = sigmoid_to_range(x[:, 10:11], self.ankle_wrap_min, self.ankle_wrap_max)
        ankle_stiffness = sigmoid_to_range(x[:, 11:12], self.ankle_stiff_min, self.ankle_stiff_max)

        approach_deg = sigmoid_to_range(x[:, 12:13], self.approach_deg_min, self.approach_deg_max)

        design_params = torch.cat([
            joint_softness,
            link_lengths,
            base_radius,
            base_length,
            tension,
            ankle_wrap_radius,
            ankle_stiffness,
        ], dim=-1)

        cyl_radius = self.task_params[:, 1:2].repeat(x.shape[0], 1)

        task_params = torch.cat([approach_deg, cyl_radius,], dim=-1)

        return design_params, task_params

    def logits2loss(self, logits):
        """
        Same method name as reference, but logits are model predictions here.

        logits[:, 0] = normalized contacts
        logits[:, 1] = disturbance resistance score
        """
        pred_contacts = logits[:, 0]
        pred_disturbance = logits[:, 1]

        if self.opt_obj == "disturbance":
            objective = self.disturbance_weight * pred_disturbance
        elif self.opt_obj == "disturbance_contact":
            objective = (
                self.disturbance_weight * pred_disturbance
                + self.contact_weight * pred_contacts
            )
        elif self.opt_obj == "contact":
            objective = pred_contacts
        else:
            raise ValueError("opt obj not supported")

        loss = -torch.mean(objective)

        if self.reg_weight > 0.0:
            loss = loss + self.reg_weight * torch.mean(self.state ** 2)

        return loss

    def forward(self, object_vertices=None):
        """
        Keep object_vertices argument for structural compatibility with the
        reference profile_optimizer.py, but it is unused for squirrel finger.
        """
        design_params, task_params = self.state_to_design()
        batch_size = design_params.shape[0]

        init_config = self.init_config.repeat(batch_size, 1)
        timesteps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        logits = self.profile_model(
            task_params=task_params,
            design_params=design_params,
            init_config=init_config,
            timesteps=timesteps,
        )

        loss = self.logits2loss(logits)
        return loss


# -----------------------------------------------------------------------------
# Adam optimizer wrapper: minimal analog of ProfileOptimizer
# -----------------------------------------------------------------------------

class ProfileOptimizer():
    def __init__(
        self,
        profile_model: nn.Module,
        batch_size: int = 16,
        num_epochs: int = 1000,
        learning_rate: float = 1e-4,
        opt_obj: str = "disturbance",
        input_dim: int = 13,
        num_points: int = 1,
        grid_size: int = 1,
        object_vertices: Optional[torch.Tensor] = None,
        seed: int = 0,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super(ProfileOptimizer, self).__init__()
        self.model = ProfileOptimizerModel(
            profile_model=profile_model,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            opt_obj=opt_obj,
            input_dim=input_dim,
            num_points=num_points,
            grid_size=grid_size,
            seed=seed,
            device=device,
            **kwargs,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.learning_rate)
        self.object_vertices = object_vertices
        self.opt_obj = opt_obj

    def solve(self):
        for _ in tqdm(range(self.model.num_epochs)):
            self.optimizer.zero_grad()
            loss = self.model(object_vertices=self.object_vertices)
            loss.backward()
            self.optimizer.step()
        design_params, task_params = self.model.state_to_design()
        return design_params.detach().cpu().numpy(), task_params.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Optional CMA-ES optimizer wrapper: minimal analog of ProfileOptimizerES
# -----------------------------------------------------------------------------

import cma
class ProfileOptimizerES():
    def __init__(
        self,
        profile_model: nn.Module,
        batch_size: int = 16,
        num_epochs: int = 1000,
        learning_rate: float = 1e-4,
        opt_obj: str = "disturbance",
        input_dim: int = 13,
        num_points: int = 1,
        grid_size: int = 1,
        object_vertices: Optional[torch.Tensor] = None,
        seed: int = 0,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super(ProfileOptimizerES, self).__init__()
        if cma is None:
            raise ImportError("cma is not installed. Install cma or use ProfileOptimizer instead.")

        self.model = ProfileOptimizerModel(
            profile_model=profile_model,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            opt_obj=opt_obj,
            input_dim=input_dim,
            num_points=num_points,
            grid_size=grid_size,
            seed=seed,
            device=device,
            **kwargs,
        )

        self.optimizer = cma.CMAEvolutionStrategy(
            x0=self.model.state.detach().view(-1).cpu().numpy(),
            sigma0=0.2,
            inopts={
                "popsize": 16,
            },
        )
        self.object_vertices = object_vertices
        self.opt_obj = opt_obj
        self.batch_size = batch_size
        self.input_dim = input_dim

    def solve(self):
        with torch.no_grad():
            for _ in tqdm(range(self.model.num_epochs)):
                solutions = self.optimizer.ask()
                losses = []
                for solution in solutions:
                    self.model.state = nn.Parameter(
                        torch.from_numpy(solution.reshape(self.batch_size, self.input_dim))
                        .float()
                        .to(self.model.device)
                    )
                    loss = self.model(object_vertices=self.object_vertices)
                    losses.append(loss.item())
                losses = np.asarray(losses)
                self.optimizer.tell(solutions, losses)
                print("loss", losses.mean())

        soln = np.asarray(self.optimizer.best.x).reshape(self.batch_size, self.input_dim)
        self.model.state = nn.Parameter(torch.from_numpy(soln).float().to(self.model.device))
        design_params, task_params = self.model.state_to_design()
        return design_params.detach().cpu().numpy(), task_params.detach().cpu().numpy()
