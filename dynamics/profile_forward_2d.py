import math
import torch
import torch.nn as nn


class ApproachAngleEmbedder:
    def __init__(self, multires=4):
        self.multires = multires
        self.freq_bands = 2.0 ** torch.linspace(0.0, multires - 1, steps=multires)

    def embed(self, angle):
        freq_bands = self.freq_bands.to(angle.device)

        embeds = [angle]
        for freq in freq_bands:
            embeds.append(torch.sin(angle * freq))
            embeds.append(torch.cos(angle * freq))

        return torch.cat(embeds, dim=-1)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )

    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])],
            dim=-1,
        )

    return embedding


class ProfileForward2DModel(nn.Module):
    def __init__(
        self,
        W=256,
        task_ch=2,
        design_ch=12,
        init_ch=3,
        output_ch=3,
    ):
        super().__init__()

        self.W = W

        self.angle_embedder = ApproachAngleEmbedder(multires=4)
        angle_embed_dim = 1 + 2 * 4  # 9

        task_input_dim = angle_embed_dim + 1  # embedded angle + cyl_radius

        self.time_embed_dim = W
        self.time_encoder = nn.Sequential(
            nn.Linear(W // 2, W),
            nn.SiLU(),
            nn.Linear(W, W),
        )

        input_dim = task_input_dim + design_ch + init_ch + self.time_embed_dim

        self.linears = nn.Sequential(
            nn.Linear(input_dim, W),
            nn.ReLU(),

            nn.Linear(W, W),
            nn.ReLU(),

            nn.Linear(W, W),
            nn.ReLU(),

            nn.Linear(W, output_ch),
        )

    def forward(self, task_params, design_params, init_config, timesteps):
        approach_angle = task_params[:, 0:1]
        cyl_radius = task_params[:, 1:2]

        approach_embed = self.angle_embedder.embed(approach_angle)
        x_task = torch.cat([approach_embed, cyl_radius], dim=-1)

        time_emb = self.time_encoder(
            timestep_embedding(timesteps, self.W // 2)
        )

        x = torch.cat(
            [x_task, design_params, init_config, time_emb],
            dim=-1,
        )

        return self.linears(x)