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
        # params_ch = 160 (80 nodes * 2D coords)
        # pos_ch = 2 (tension, base_rad)
        # ori_ch = 1 (approach_angle)
        # output_ch = 1 (number_of_contacts)
        self.model = ProfileForward2DModel(
            W=256, 
            params_ch=3, 
            ori_ch=1, 
            pos_ch=2, 
            output_ch=1, 
            physics_ch=1,
            object_ch=1, # cyl_rad, cyl_height
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.criterion = nn.MSELoss()

    def step(self, ctrlpts, score, input_ori, input_pos, nodes, stiffness, cyl_rad):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Flatten rod points [batch, 80, 2] -> [batch, 160]
        ctrlpts = ctrlpts.view(ctrlpts.shape[0], -1)
        timesteps = torch.zeros(input_ori.shape[0]).to(self.device)
        
        pred = self.model(ctrlpts, input_ori, input_pos, timesteps, nodes, stiffness, cyl_rad)
        
        loss = self.criterion(pred, score)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), pred

    def inference(self, dummy, ctrlpts, score, input_ori, input_pos, nodes, stiffness, cyl_rad):
        self.model.eval()
        with torch.no_grad():
            ctrlpts = ctrlpts.view(ctrlpts.shape[0], -1)
            timesteps = torch.zeros(input_ori.shape[0]).to(self.device)
            pred = self.model(ctrlpts, input_ori, input_pos, timesteps, nodes, stiffness, cyl_rad)
            loss = self.criterion(pred, score)
        return pred, loss

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)