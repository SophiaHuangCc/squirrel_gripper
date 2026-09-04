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
        task_ch=3,
        design_ch=16,
        init_ch=3,
        output_ch=3,
        architecture="legacy",
        num_hidden_layers=3,
    ):
        super().__init__()

        self.W = W
        self.architecture = architecture
        self.num_hidden_layers = int(num_hidden_layers)
        self.task_ch = task_ch
        self.design_ch = design_ch
        self.init_ch = init_ch
        self.output_ch = output_ch
        if architecture not in {"legacy", "dgdm"}:
            raise ValueError("architecture must be 'legacy' or 'dgdm'")
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be positive")

        self.angle_embedder = ApproachAngleEmbedder(multires=4)
        angle_embed_dim = 1 + 2 * 4  # 9

        task_input_dim = 2 * angle_embed_dim + 1

        self.time_embed_dim = W
        self.time_encoder = nn.Sequential(
            nn.Linear(W // 2, W),
            nn.SiLU(),
            nn.Linear(W, W),
        )

        if architecture == "dgdm":
            self.design_encoder = nn.Sequential(
                nn.Linear(design_ch, W), nn.ReLU(), nn.Linear(W, W),
            )
            self.context_encoder = nn.Sequential(
                nn.Linear(task_input_dim + init_ch, W), nn.ReLU(), nn.Linear(W, W),
            )
            trunk = []
            for index in range(self.num_hidden_layers):
                trunk.extend((
                    nn.Linear(3 * W if index == 0 else W, W),
                    nn.BatchNorm1d(W),
                    nn.ReLU(),
                ))
            self.linears = nn.Sequential(*trunk)
            self.output = nn.Linear(W, output_ch)
        else:
            self.design_encoder = None
            self.context_encoder = None
            input_dim = task_input_dim + design_ch + init_ch + self.time_embed_dim
            trunk = []
            for index in range(self.num_hidden_layers):
                trunk.extend((
                    nn.Linear(input_dim if index == 0 else W, W), nn.ReLU(),
                ))
            trunk.append(nn.Linear(W, output_ch))
            self.linears = nn.Sequential(*trunk)
            self.output = None

    def forward(self, task_params, design_params, init_config, timesteps):
        approach_angle = task_params[:, 0:1]
        landing_approach_angle = task_params[:, 1:2]
        cyl_radius = task_params[:, 2:3]

        approach_embed = self.angle_embedder.embed(approach_angle)
        landing_approach_embed = self.angle_embedder.embed(landing_approach_angle)
        x_task = torch.cat([approach_embed, landing_approach_embed, cyl_radius], dim=-1)

        time_emb = self.time_encoder(
            timestep_embedding(timesteps, self.W // 2)
        )

        if self.architecture == "dgdm":
            design_features = self.design_encoder(design_params)
            context_features = self.context_encoder(
                torch.cat([x_task, init_config], dim=-1)
            )
            hidden = self.linears(
                torch.cat([design_features, context_features, time_emb], dim=-1)
            )
            return self.output(hidden)

        x = torch.cat([x_task, design_params, init_config, time_emb], dim=-1)
        return self.linears(x)
