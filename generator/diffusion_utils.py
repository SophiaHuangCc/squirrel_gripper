import math
from typing import List, Optional

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """Diffusion timestep positional encoding, matching the DGDM generator style."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        n_groups = min(n_groups, out_channels)
        while out_channels % n_groups != 0:
            n_groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    1D residual block with FiLM conditioning.

    This is the same architectural idea as DGDM/diffusion-policy:
    the noisy design vector is processed by Conv1d blocks, while timestep and
    optional global condition modulate features through scale/bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0]
        bias = embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """
    DGDM-style 1D noise predictor for design vectors.

    Input shape:  (B, horizon, input_dim)
    Output shape: (B, horizon, input_dim)

    For this project, horizon=15 design parameters and input_dim=1.
    The condition is usually:
      [task_params(2), init_config(3), desired_metrics(4)] -> 9D
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        down_dims: List[int],
        diffusion_step_embed_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ]
        )

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_horizon = sample.shape[-2]
        sample = sample.moveaxis(-1, -2)
        if timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        cond = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            cond = torch.cat([cond, global_cond], dim=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, cond)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)
        # Strided down/up-sampling rounds odd horizons up (15 -> 16). The
        # denoiser must predict exactly one value for every input design field.
        x = x[..., :input_horizon]
        return x.moveaxis(-1, -2)
