import math
import torch
import torch.nn as nn

# Start with 2D for the finger-cylinder contact configuration
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:  # original raw input "x" is also included in the output
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(input_dims, multires, i=0, scalar_factor=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x/scalar_factor)
    return embed, embedder_obj.out_dim

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# class ProfileForward2DModel(nn.Module):

#     def __init__(self, W=256, params_ch=3, ori_ch=1, pos_ch=2, output_ch=1, physics_ch=1, object_ch=2):
#         super().__init__()
        
#         # 1. Individual Encoders (Mapping raw inputs to a shared feature space)
#         self.nodes_encoder = nn.Sequential(
#             nn.Linear(params_ch, W), # params_ch = 160 (80 nodes * 2)
#             nn.ReLU(),
#             nn.Linear(W, W)
#         )
        
#         # Stiffness (E, G)
#         self.stiffness_encoder = nn.Sequential(
#             nn.Linear(physics_ch, W // 4), 
#             nn.ReLU()
#         )
        
#         # Object Params (Cyl_rad, etc.)
#         self.obj_encoder = nn.Sequential(
#             nn.Linear(object_ch, W // 4),
#             nn.ReLU()
#         )

#         # 2. The Fusion Layer
#         # Summing the input dimensions: 
#         # nodes_feat(W) + stiffness(W/4) + obj(W/4) + ori(1) + pos(2)
#         input_dim = W + (W // 4) + (W // 4) + ori_ch + pos_ch
        
#         self.backbone = nn.Sequential(
#             nn.Linear(input_dim, W),
#             nn.ReLU(),
#             nn.Linear(W, W),
#             nn.ReLU(),
#             nn.Linear(W, output_ch)
#         )

#     def forward(self, ctrlpts, input_ori, input_pos, timesteps, nodes, stiffness, cyl_rad):
#         # Flatten nodes if they come in as (B, 80, 2)
#         nodes_flat = nodes.view(nodes.shape[0], -1)
        
#         # Extract features
#         feat_nodes = self.nodes_encoder(nodes_flat)
#         feat_stiff = self.stiffness_encoder(stiffness)
#         feat_obj = self.obj_encoder(cyl_rad)
        
#         # Combine everything into one feature vector
#         combined = torch.cat([
#             feat_nodes, 
#             feat_stiff, 
#             feat_obj, 
#             input_ori, 
#             input_pos
#         ], dim=-1)
        
#         # Pass through the main network
#         output = self.backbone(combined)
#         return output

class ProfileForward2DModel(nn.Module):
    def __init__(self, W=256, params_ch=160, ori_ch=1, pos_ch=1, output_ch=1, design_ch=8, physics_ch=1, object_ch=16):
        super().__init__()
        
        # 1. State Encoder (The Rod Shape)
        # params_ch = 160 (80 nodes * 2D)
        self.state_encoder = nn.Sequential(
            nn.Linear(params_ch, W),
            nn.LayerNorm(W),
            nn.ReLU(),
            nn.Linear(W, W // 2)
        )
        
        # 2. Design Encoder (Finger structural/inertial params)
        # design_ch = 8 (nodes, base_len, base_rad, masses, joint_soft)
        self.design_encoder = nn.Sequential(
            nn.Linear(design_ch, W // 4),
            nn.ReLU(),
            nn.Linear(W // 4, W // 4)
        )
        
        # 3. Physics & Environment Encoder (Cylinder + Contact + Material)
        # physics_ch = 1 (Youngs Modulus)
        # object_ch = 16 (Cyl pos/dir/rad/len + nu/mu contact)
        self.env_encoder = nn.Sequential(
            nn.Linear(physics_ch + object_ch, W // 2),
            nn.ReLU(),
            nn.Linear(W // 2, W // 4)
        )

        # 4. Fusion Backbone
        # state(W/2) + design(W/4) + env(W/4) + ori(1) + pos(1)
        fusion_dim = (W // 2) + (W // 4) + (W // 4) + ori_ch + pos_ch
        
        self.backbone = nn.Sequential(
            nn.Linear(fusion_dim, W),
            nn.ReLU(),
            nn.Linear(W, W // 2),
            nn.ReLU(),
            nn.Linear(W // 2, output_ch),
            nn.Sigmoid() # CRITICAL: Constrains output to [0, 1] stability score
        )

    def forward(self, ctrlpts, input_ori, input_tension, timesteps, design_tensor, physics_tensor, object_tensor):
        """
        Inputs provided by the Trainer._prepare_tensors() helper:
        - ctrlpts: Flattened rod (B, 160)
        - input_ori: Orientation (B, 1)
        - input_tension: Tension (B, 1)
        - design_tensor: (B, 8)
        - physics_tensor: (B, 1)
        - object_tensor: (B, 16)
        """
        
        # Encode separate knowledge branches
        feat_state = self.state_encoder(ctrlpts)
        feat_design = self.design_encoder(design_tensor)
        feat_env = self.env_encoder(torch.cat([physics_tensor, object_tensor], dim=-1))
        
        # Concatenate all features with the direct action inputs
        combined = torch.cat([
            feat_state, 
            feat_design, 
            feat_env, 
            input_ori, 
            input_tension
        ], dim=-1)
        
        # Final prediction
        return self.backbone(combined)