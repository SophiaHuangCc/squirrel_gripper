"""Preference classifier guidance applied to the shared unconditional DDIM prior."""

from __future__ import annotations

import torch

from generator.dataloader import diffusion_to_physical, physical_to_diffusion, physical_to_model_norm, project_physical_design


def preference_scores(classifier, candidates, references, timestep):
    """Mean log-probability that each candidate is preferred to its reference set."""
    batch, refs = candidates.shape[0], references.shape[0]
    a = candidates[:, None, :].expand(batch, refs, -1).reshape(batch * refs, -1)
    b = references[None, :, :].expand(batch, refs, -1).reshape(batch * refs, -1)
    t = torch.as_tensor(timestep, device=candidates.device).reshape(1).expand(batch * refs)
    logits = classifier(a, b, t).reshape(batch, refs)
    return torch.nn.functional.logsigmoid(logits).mean(dim=1)


@torch.no_grad()
def sample_pareto_guided(prior, classifier, reference_units, batch_size, guidance_scale=0.1,
                         references_per_sample=16, generator=None, device=None):
    device = device or next(prior.parameters()).device
    reference_units = reference_units.to(device)
    if reference_units.ndim != 2 or reference_units.shape[1] != prior.bounds.lo.numel():
        raise ValueError("reference_units must have shape (N, design_dim)")
    if not len(reference_units):
        raise ValueError("At least one Pareto reference design is required")
    sample = torch.randn(batch_size, prior.bounds.lo.numel(), 1, device=device, generator=generator)
    prior.noise_scheduler.set_timesteps(prior.num_inference_steps, device=device)
    for t in prior.noise_scheduler.timesteps:
        timestep = t.expand(batch_size)
        noise_pred = prior.noise_pred_net(sample, timestep, global_cond=None)
        if guidance_scale != 0:
            count = min(int(references_per_sample), len(reference_units))
            selected = torch.randperm(len(reference_units), device=device, generator=generator)[:count]
            clean_refs = reference_units[selected]
            ref_noise = torch.randn(clean_refs.shape, device=device, dtype=clean_refs.dtype, generator=generator)
            ref_t = t.expand(count)
            noisy_refs = prior.noise_scheduler.add_noise(clean_refs, ref_noise, ref_t)
            with torch.enable_grad():
                x = sample.detach().squeeze(-1).requires_grad_(True)
                score = preference_scores(classifier, x, noisy_refs, t).sum()
                grad = torch.autograd.grad(score, x)[0].unsqueeze(-1)
                norm = grad.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1)
                grad = grad / norm
            sigma = (1.0 - prior.noise_scheduler.alphas_cumprod[t]).sqrt()
            noise_pred = noise_pred - float(guidance_scale) * sigma * grad
        sample = prior.noise_scheduler.step(noise_pred, t, sample).prev_sample.clamp(-1.5, 1.5)
    physical = project_physical_design(diffusion_to_physical(sample.squeeze(-1).clamp(-1, 1), prior.bounds), prior.bounds)
    return {"design_physical": physical, "design_norm": physical_to_model_norm(physical),
            "design_unit": physical_to_diffusion(physical, prior.bounds).clamp(-1, 1)}
