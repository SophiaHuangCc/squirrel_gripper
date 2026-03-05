import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class DynamicsDataset(Dataset):
    def __init__(self, dataset_dir, **kwargs):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.files = glob.glob(os.path.join(self.dataset_dir, "**/*.npz"), recursive=True)
        
        print(f"Dataset initialized at: {self.dataset_dir}")
        print(f"Found {len(self.files)} .npz files.")

        self.threshold = 50.0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=True) as data:

            # Predicting a continuous value of num_contacts with regression
            max_contacts = 80.0
            num_contacts = torch.tensor([float(data.get('num_contacts', 0)) / max_contacts]).float()
            
            # Input geometry (final frame)
            pos = data["position"][-1]
            ctrlpts = torch.from_numpy(pos[[0, 2], :]).float()

            # Actuation and orientation inputs
            input_tension = torch.tensor([float(data.get('tension', 0))]).float()

            finger_params = {
                'nodes': torch.from_numpy(data['vertebra_nodes']).float(),
                'base_length': torch.from_numpy(data['base_length']).float(),
                'base_radius': torch.tensor([float(data.get('base_radius', 0.005))]).float(),                'input_ori': torch.tensor([float(data.get('arg_approach_deg', 0))]).float(),
                'youngs_modulus': torch.tensor(data['E']).float().flatten(),
                'finger_mass': torch.from_numpy(data.get('mass', 0.0)).float(),
                'body_mass': torch.tensor([float(data.get('body_mass', 0.0))]).float(),
                'joint_softness': torch.tensor([float(data.get('joint_softness', 0.001))]).float(),
            }
            
            cylinder_params = {
                'cyl_position': torch.from_numpy(data['cyl_position']).float().flatten(),
                'cyl_directors': torch.from_numpy(data['cyl_directors']).float().flatten(),
                'cyl_radius': torch.from_numpy(data['cyl_radius']).float(),
                'cyl_length': torch.from_numpy(data['cyl_length']).float().flatten(),
            }

            contact_params = {
                'nu_contact': torch.from_numpy(data['nu_contact']).float(),
                'mu_contact': torch.from_numpy(data['mu_contact']).float(),
            }

            disturbance_params = torch.tensor([float(data.get('force_resistance_score', 0.0))]).float()

            # all_pos = data.get('position')
            
            # initial_shape = all_pos[0, :2, :80].T # Results in (80, 2)
            
            # ctrlpts = torch.from_numpy(initial_shape).float()

        return {
            'num_contacts': num_contacts,
            'ctrlpts': ctrlpts,
            'input_tension': input_tension,
            'finger_params': finger_params,
            'cylinder_params': cylinder_params,
            'contact_params': contact_params,
            'disturbance_params': disturbance_params,
        }
