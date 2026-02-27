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
            
            input_ori = torch.tensor([float(data.get('arg_approach_deg', 0))]).float()
            input_pos = torch.tensor([
                float(data.get('tension', 0)), 
                float(data.get('arg_base_rad', 0.007))
            ]).float()

            design_params = {
                'nodes': torch.from_numpy(data['vertebra_nodes']).float(),
                'rest_kappa': torch.from_numpy(data['rest_kappa']).float(),
                'rest_lengths': torch.from_numpy(data['rest_lengths']).float()
            }

            physics_params = {
                'stiffness': torch.tensor(data['E']).float().flatten(),
                'mass_props': torch.from_numpy(data['mass']).float(),
                'bend_matrix': torch.from_numpy(data['bend_matrix']).float(),
                'softness': torch.tensor([float(data.get('joint_softness', 0.001))]).float()            }
            
            obj_params = {
                'cyl_radius': torch.from_numpy(data['cyl_radius']).float(),
                'cyl_pos': torch.from_numpy(data['cyl_position']).float().flatten(),
                'cyl_directors': torch.from_numpy(data['cyl_directors']).float().flatten(),
            }

            all_pos = data.get('position')
            
            initial_shape = all_pos[0, :2, :80].T # Results in (80, 2)
            
            ctrlpts = torch.from_numpy(initial_shape).float()

        return {
            'num_contacts': num_contacts,
            'input_ori': input_ori,
            'input_pos': input_pos,
            'ctrlpts': ctrlpts,
            'design_params': design_params,
            'physics_params': physics_params,
            'obj_params': obj_params
        }
