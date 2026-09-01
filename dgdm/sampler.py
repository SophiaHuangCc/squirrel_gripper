"""Unconditional DDIM prior with interaction-profile gradient guidance."""

from typing import Optional
import torch
from torch import nn

from generator.dataloader import DesignBounds, diffusion_to_physical, physical_to_diffusion, physical_to_model_norm, project_physical_design
from .guidance import ProfileTarget, ScenarioBatch, aggregate_profile_score


class SimpleEMA:
    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {name: value.detach().clone() for name, value in module.named_parameters()}

    def step(self, module: nn.Module):
        with torch.no_grad():
            for name, value in module.named_parameters():
                self.shadow[name] = self.shadow[name].to(value).mul(self.decay).add(value.detach(), alpha=1-self.decay)

    def copy_to(self, module: nn.Module):
        with torch.no_grad():
            for name, value in module.named_parameters():
                value.copy_(self.shadow[name].to(value))

    def state_dict(self):
        return {name: value.detach().cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, state, device):
        self.shadow = {name: value.detach().clone().to(device) for name, value in state.items()}


class DGDMDesignDiffusion(nn.Module):
    def __init__(self, noise_pred_net, noise_scheduler, bounds=None, num_inference_steps=20, ema_decay=0.999):
        super().__init__()
        self.noise_pred_net = noise_pred_net
        self.noise_scheduler = noise_scheduler
        self.bounds = bounds or DesignBounds.defaults()
        self.num_inference_steps = int(num_inference_steps)
        self.ema = SimpleEMA(noise_pred_net, decay=ema_decay)

    def training_loss(self, batch):
        clean = batch["design_unit"]
        noise = torch.randn_like(clean)
        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (clean.shape[0],), device=clean.device)
        noisy = self.noise_scheduler.add_noise(clean, noise, timestep)
        return torch.nn.functional.mse_loss(self.noise_pred_net(noisy, timestep, global_cond=None), noise)

    @torch.no_grad()
    def sample(self, batch_size: int, dynamics_model=None, scenarios: Optional[ScenarioBatch]=None,
               target: Optional[ProfileTarget]=None, guidance_scale=0.0, generator=None, device=None):
        device = device or next(self.parameters()).device
        sample = torch.randn(batch_size, self.bounds.lo.numel(), 1, device=device, generator=generator)
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=device)
        guided = guidance_scale != 0.0
        if guided and (dynamics_model is None or scenarios is None or target is None):
            raise ValueError("Guided sampling requires dynamics_model, scenarios, and target")
        for t in self.noise_scheduler.timesteps:
            timestep = t.expand(batch_size)
            noise = self.noise_pred_net(sample, timestep, global_cond=None)
            if guided:
                with torch.enable_grad():
                    x = sample.detach().requires_grad_(True)
                    # The profile model was trained on x_t in diffusion [-1, 1]
                    # coordinates and on the matching normalized noise level.
                    tau = t.float() / float(self.noise_scheduler.config.num_train_timesteps)
                    score = aggregate_profile_score(
                        dynamics_model, x.squeeze(-1), scenarios, target, tau
                    ).sum()
                    grad = torch.autograd.grad(score, x)[0]
                    grad = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1)
                sigma = (1.0 - self.noise_scheduler.alphas_cumprod[t]).sqrt()
                noise = noise - float(guidance_scale) * sigma * grad
            sample = self.noise_scheduler.step(noise, t, sample).prev_sample.clamp(-1.5, 1.5)
        physical = project_physical_design(diffusion_to_physical(sample.squeeze(-1).clamp(-1, 1), self.bounds), self.bounds)
        return {"design_physical": physical, "design_norm": physical_to_model_norm(physical),
                "design_unit": physical_to_diffusion(physical, self.bounds).clamp(-1, 1)}
