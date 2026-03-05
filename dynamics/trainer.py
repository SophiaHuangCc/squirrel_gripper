import torch
import torch.nn as nn
import torch.optim as optim
from dynamics.profile_forward_2d import ProfileForward2DModel

class Trainer:
    def __init__(self, args):
        self.args = args
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def create_model(self):
        # params_ch: 160 (80 nodes * 2D)
        # design_ch: nodes(3) + base_len(1) + base_rad(1) + f_mass(1) + b_mass(1) + joint_soft(1) = 8
        # physics_ch: youngs_modulus(1)
        # object_ch: cyl_pos(3) + cyl_dir(9) + cyl_rad(1) + cyl_len(1) + nu(1) + mu(1) = 16
        self.model = ProfileForward2DModel(
            W=256, 
            params_ch=162, 
            ori_ch=1,     # input_ori
            pos_ch=1,     # input_tension
            output_ch=1,  # disturbance_params (stability score)
            design_ch=88,
            physics_ch=1,
            object_ch=16
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.criterion = nn.MSELoss()

    def _prepare_tensors(self, ctrlpts, tension, nodes, base_length, base_radius, input_ori, 
                         youngs_modulus, finger_mass, body_mass, joint_softness,
                         cyl_position, cyl_directors, cyl_radius, cyl_length,
                         nu_contact, mu_contact):
        
        # Flatten Rod Geometry
        ctrlpts = ctrlpts.view(ctrlpts.shape[0], -1)
        
        # Design Group [Batch, 8]
        design_tensor = torch.cat([nodes, base_length, base_radius, finger_mass, body_mass, joint_softness], dim=1)
        
        # Physics Group [Batch, 1] - Ensure it is (B, 1)
        physics_tensor = youngs_modulus.view(-1, 1) if youngs_modulus.dim() == 1 else youngs_modulus
        
        # Object & Contact Group [Batch, 16]
        object_tensor = torch.cat([cyl_position, cyl_directors, cyl_radius, cyl_length, nu_contact, mu_contact], dim=1)
        
        timesteps = torch.zeros(tension.shape[0], 1).to(self.device)
        
        return ctrlpts, input_ori, tension, timesteps, design_tensor, physics_tensor, object_tensor

    def step(self, *args):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Unpack, Prepare, and Forward
        score = args[0] # The target is the first arg in train's loop
        prepared_inputs = self._prepare_tensors(*args[1:]) 
        pred = self.model(*prepared_inputs)
        
        loss = self.criterion(pred, score)
        loss.backward()
        self.optimizer.step()
        return loss.item(), pred

    def inference(self, *args):
        self.model.eval()
        with torch.no_grad():
            prepared_inputs = self._prepare_tensors(*args)
            pred = self.model(*prepared_inputs)
            return pred, torch.tensor(0.0)

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)