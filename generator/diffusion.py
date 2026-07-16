from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from generator.dataloader import (
    DesignBounds,
    build_condition,
    diffusion_to_physical,
    physical_to_diffusion,
    physical_to_model_norm,
    project_physical_design,
)


class SimpleEMA:
    """Small version-stable EMA helper for model parameters."""

    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in module.named_parameters()
            if param.requires_grad
        }

    def step(self, module: nn.Module) -> None:
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name not in self.shadow:
                    continue
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, module: nn.Module) -> None:
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name in self.shadow:
                    param.copy_(self.shadow[name].to(param.device))

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor], device: torch.device) -> None:
        self.shadow = {name: value.detach().clone().to(device) for name, value in state.items()}


class SquirrelDesignDiffusion(nn.Module):
    """
    Squirrel-finger adaptation of DGDM's diffusion generator.

    Training learns p(design | task, init, desired_metrics).
    Sampling starts from Gaussian noise and denoises into a 12D physical design.
    """

    def __init__(
        self,
        noise_pred_net: nn.Module,
        noise_scheduler: DDIMScheduler,
        bounds: Optional[DesignBounds] = None,
        learning_rate: float = 1e-4,
        ema_power: float = 0.75,
        num_inference_steps: int = 20,
    ):
        super().__init__()
        self.noise_pred_net = noise_pred_net
        self.noise_scheduler = noise_scheduler
        self.learning_rate = learning_rate
        self.bounds = bounds or DesignBounds.defaults()
        self.num_inference_steps = num_inference_steps
        # DGDM uses EMA weights for sampling. Use an internal helper so this
        # code does not depend on the exact diffusers EMA API version.
        ema_decay = 1.0 - (1.0 - float(ema_power)) * 0.01
        self.ema = SimpleEMA(self.noise_pred_net, decay=ema_decay)

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        clean = batch["design_unit"]
        cond = batch["cond"]
        noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (clean.shape[0],),
            device=clean.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(clean, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy, timesteps, global_cond=cond)
        return F.mse_loss(noise_pred, noise)

    def optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.noise_pred_net.parameters(), lr=self.learning_rate)

    def update_ema(self) -> None:
        self.ema.step(self.noise_pred_net)

    def copy_ema_to_model(self) -> None:
        self.ema.copy_to(self.noise_pred_net)

    def _score_for_guidance(
        self,
        design_unit: torch.Tensor,
        cond: torch.Tensor,
        dynamics_model: nn.Module,
        objective: str,
    ) -> torch.Tensor:
        design_physical = project_physical_design(
            diffusion_to_physical(design_unit.squeeze(-1), self.bounds),
            self.bounds,
        )
        design_norm = physical_to_model_norm(design_physical)
        task_params = cond[:, 0:2]
        init_config = cond[:, 2:5]
        timesteps = torch.zeros(design_unit.shape[0], dtype=torch.float32, device=design_unit.device)
        pred = dynamics_model(task_params, design_norm, init_config, timesteps)

        contacts = pred[:, 0]
        disturbance = pred[:, 1]
        angular_span = pred[:, 2]
        curl_speed = pred[:, 3] if pred.shape[-1] > 3 else torch.zeros_like(contacts)

        if objective == "disturbance":
            score = disturbance
        elif objective == "contact":
            score = contacts
        elif objective == "angular_span":
            score = angular_span
        elif objective == "curl_speed":
            score = curl_speed
        elif objective == "disturbance_contact_span":
            score = disturbance + 0.1 * contacts + 0.5 * angular_span
        elif objective == "disturbance_contact_span_speed":
            gate = torch.sigmoid((contacts - 0.3) / 0.05)
            score = disturbance + 0.1 * contacts + 0.5 * angular_span + 0.1 * curl_speed * gate
        else:
            raise ValueError(f"Unknown guidance objective: {objective}")
        return score

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        dynamics_model: Optional[nn.Module] = None,
        guidance_scale: float = 0.0,
        guidance_objective: str = "disturbance_contact_span_speed",
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate designs.

        If guidance_scale=0, this is ordinary conditional DDIM sampling.
        If guidance_scale>0 and dynamics_model is provided, classifier-style
        guidance nudges the denoising process toward high predicted objective.
        """
        device = cond.device
        sample = torch.randn(
            cond.shape[0],
            self.bounds.lo.numel(),
            1,
            device=device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=device)

        for t in self.noise_scheduler.timesteps:
            timesteps = t.expand(cond.shape[0])
            noise_pred = self.noise_pred_net(sample, timesteps, global_cond=cond)

            if dynamics_model is not None and guidance_scale > 0.0:
                with torch.enable_grad():
                    guided_sample = sample.detach().requires_grad_(True)
                    score = self._score_for_guidance(
                        guided_sample,
                        cond,
                        dynamics_model=dynamics_model,
                        objective=guidance_objective,
                    ).sum()
                    grad = torch.autograd.grad(score, guided_sample)[0]
                alpha_term = (1.0 - self.noise_scheduler.alphas_cumprod[t]).sqrt()
                noise_pred = noise_pred - alpha_term * guidance_scale * grad

            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            sample = sample.clamp(-1.5, 1.5)

        design_physical = project_physical_design(
            diffusion_to_physical(sample.squeeze(-1).clamp(-1.0, 1.0), self.bounds),
            self.bounds,
        )
        design_unit = physical_to_diffusion(design_physical, self.bounds).clamp(-1.0, 1.0)
        design_norm = physical_to_model_norm(design_physical)
        return {
            "design_unit": design_unit,
            "design_physical": design_physical,
            "design_norm": design_norm,
        }


def make_condition_batch(
    batch_size: int,
    approach_deg: float = 45.0,
    cyl_radius: float = 0.03,
    landing_height: float = 0.04,
    landing_speed: float = 0.0,
    initial_x_gap: float = 0.06,
    target_contacts: float = 0.6,
    target_disturbance: float = 0.8,
    target_angular_span: float = 0.8,
    target_curl_speed: float = 0.8,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    task = torch.tensor(
        [[approach_deg / 90.0, cyl_radius / 0.05]],
        dtype=torch.float32,
        device=device,
    ).repeat(batch_size, 1)
    init = torch.tensor(
        [[landing_height / 0.10, landing_speed / 1.0, initial_x_gap / 0.10]],
        dtype=torch.float32,
        device=device,
    ).repeat(batch_size, 1)
    metrics = torch.tensor(
        [[target_contacts, target_disturbance, target_angular_span, target_curl_speed]],
        dtype=torch.float32,
        device=device,
    ).repeat(batch_size, 1)
    return build_condition(task, init, metrics)
