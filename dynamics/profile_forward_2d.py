import math
import torch
import torch.nn as nn


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0

        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        n_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=n_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=n_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_embedder(input_dims, multires, i=0, scalar_factor=1):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x / scalar_factor)
    return embed, embedder_obj.out_dim


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) *
        torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
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
        self.task_ch = task_ch
        self.design_ch = design_ch
        self.init_ch = init_ch
        self.output_ch = output_ch

        self.task_embed, task_embed_dim = get_embedder(task_ch, multires=4, i=0, scalar_factor=1)
        self.init_embed, init_embed_dim = get_embedder(init_ch, multires=4, i=0, scalar_factor=1)

        self.time_embed_dim = W
        self.time_encoder = nn.Sequential(
            nn.Linear(W // 2, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.task_encoder = nn.Sequential(
            nn.Linear(task_embed_dim, W),
            nn.ReLU(),
            nn.Linear(W, W),
        )
        self.task_encode_dim = W

        self.design_encoder = nn.Sequential(
            nn.Linear(design_ch, W),
            nn.ReLU(),
            nn.Linear(W, W),
        )
        self.design_encode_dim = W

        self.init_encoder = nn.Sequential(
            nn.Linear(init_embed_dim, W),
            nn.ReLU(),
            nn.Linear(W, W),
        )
        self.init_encode_dim = W

        fusion_dim = self.task_encode_dim + self.design_encode_dim + self.init_encode_dim + self.time_embed_dim
        self.linears = nn.Sequential(
            nn.Linear(fusion_dim, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
        )
        self.output = nn.Linear(W, output_ch)

    def forward(self, task_params, design_params, init_config, timesteps):
        """
        Args:
            task_params  : (B, 2)
            design_params: (B, 12)   <-- noisy design vector
            init_config  : (B, 3)
            timesteps    : (B,)

        Returns:
            pred: (B, 3)
                  [num_contacts, force_closure, stability_margin]
        """
        x_task = self.task_embed(task_params)
        x_init = self.init_embed(init_config)

        feat_task = self.task_encoder(x_task)
        feat_design = self.design_encoder(design_params)
        feat_init = self.init_encoder(x_init)

        time_emb = self.time_encoder(timestep_embedding(timesteps, self.W // 2))

        fused = torch.cat([feat_task, feat_design, feat_init, time_emb], dim=-1)
        x = self.linears(fused)
        pred = self.output(x)
        return pred