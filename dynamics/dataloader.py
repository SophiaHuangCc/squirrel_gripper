import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DynamicsDataset(Dataset):
    def __init__(self, dataset_dir, **kwargs):
        self.dataset_dir = os.path.abspath(dataset_dir)
        # self.files = glob.glob(os.path.join(self.dataset_dir, "*.npz"), recursive=True)
        self.files = sorted(
            glob.glob(os.path.join(self.dataset_dir, "*.npz")) +
            glob.glob(os.path.join(self.dataset_dir, "**", "*.npz"), recursive=True)
        )

        print(f"Dataset initialized at: {self.dataset_dir}")
        print(f"Found {len(self.files)} .npz files.")

    def __len__(self):
        return len(self.files)

    def _get_scalar(self, data, key, default=0.0):
        """
        Safely read a scalar from npz.
        Works whether the stored value is a scalar, shape-(1,) array, etc.
        """
        if key not in data:
            return float(default)

        value = data[key]
        if np.isscalar(value):
            return float(value)

        value = np.asarray(value).reshape(-1)
        if value.size == 0:
            return float(default)

        return float(value[0])

    def _get_array(self, data, key, default=None, dtype=np.float32):
        """
        Safely read an array from npz and flatten it.

        Supports:
        - numeric arrays already stored in npz
        - scalar values
        - strings like "30,46,62"
        - shape-(1,) object/string arrays containing comma-separated values
        """
        if key not in data:
            if default is None:
                raise KeyError(f"Missing required key '{key}' in dataset sample.")
            return np.asarray(default, dtype=dtype).reshape(-1)

        value = data[key]

        # Case 1: direct scalar string/object
        if isinstance(value, (str, bytes)):
            text = value.decode() if isinstance(value, bytes) else value
            return np.asarray(
                [x.strip() for x in text.split(",") if x.strip() != ""],
                dtype=dtype
            ).reshape(-1)

        arr = np.asarray(value)

        # Case 2: object/string array, often shape (1,)
        if arr.dtype.kind in {"U", "S", "O"}:
            flat = arr.reshape(-1)

            # If it is a single comma-separated string, parse it
            if flat.size == 1:
                item = flat[0]
                if isinstance(item, bytes):
                    item = item.decode()
                item = str(item)

                return np.asarray(
                    [x.strip() for x in item.split(",") if x.strip() != ""],
                    dtype=dtype
                ).reshape(-1)

            # Otherwise treat each entry as one element
            parsed = []
            for item in flat:
                if isinstance(item, bytes):
                    item = item.decode()
                item = str(item).strip()
                if item != "":
                    parsed.append(item)

            return np.asarray(parsed, dtype=dtype).reshape(-1)

        # Case 3: already numeric
        return arr.astype(dtype).reshape(-1)

    def _compute_link_lengths_from_joint_positions(self, joint_positions, base_length, n_elements):
        """
        Convert joint positions (indices along the discretized finger) into physical link lengths.

        Example:
            joint_positions = [30, 45, 62]
            n_elements = 80
            base_length = 0.10

            boundaries = [0, 30, 45, 62, 80]
            link_lengths_idx = [30, 15, 17, 18]
            link_lengths_m = link_lengths_idx * (0.10 / 80)
        """
        joint_positions = np.asarray(joint_positions, dtype=np.float32).reshape(-1)

        if joint_positions.size == 0:
            return np.asarray([base_length], dtype=np.float32)

        # Sort just in case
        joint_positions = np.sort(joint_positions)

        # Clip to valid range
        joint_positions = np.clip(joint_positions, 0, n_elements)

        boundaries = np.concatenate((
            np.asarray([0.0], dtype=np.float32),
            joint_positions,
            np.asarray([float(n_elements)], dtype=np.float32)
        ))

        link_lengths_idx = np.diff(boundaries)
        dx = float(base_length) / float(n_elements)
        link_lengths = link_lengths_idx * dx

        return link_lengths.astype(np.float32)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=True) as data:
            ####################################################################
            # 1. TASK PARAMETERS
            ####################################################################
            # [approach angle, cylinder radius]
            approach_angle = self._get_scalar(data, "arg_approach_deg", 0.0)
            cylinder_radius = self._get_scalar(data, "cyl_radius", 0.015)

            task_params = torch.tensor(
                [approach_angle, cylinder_radius],
                dtype=torch.float32
            )

            ####################################################################
            # 2. DESIGN PARAMETERS
            ####################################################################
            # joint stiffness
            joint_softness = self._get_array(
                data,
                "joint_softness",
                default=[0.001, 0.001, 0.001]
            )

            # base geometry
            base_radius = self._get_scalar(data, "base_radius", 0.005)
            base_length = self._get_scalar(data, "base_length", 0.10)
            tension = self._get_scalar(data, "tension", 0.0)
            ankle_wrap_radius = self._get_scalar(data, "arg_ankle_wrap_radius", 0.005)
            ankle_stiffness = self._get_scalar(data, "arg_ankle_stiffness", 500.0)

            # link lengths from joint_positions
            joint_positions = self._get_array(
                data,
                "vertebra_nodes",
                default=[30, 46, 62]
            )

            # need n_elements to convert index spacing -> physical spacing
            n_elements = int(round(self._get_scalar(data, "n_elements", 80.0)))

            link_lengths = self._compute_link_lengths_from_joint_positions(
                joint_positions=joint_positions,
                base_length=base_length,
                n_elements=n_elements
            )

            design_params_np = np.concatenate([
                joint_softness.astype(np.float32),      # e.g. 3 values
                link_lengths.astype(np.float32),         # e.g. 4 values
                np.asarray([
                    base_radius,
                    base_length,
                    tension,
                    ankle_wrap_radius,
                    ankle_stiffness,
                ], dtype=np.float32)
            ])

            design_params = torch.from_numpy(design_params_np).float()

            ####################################################################
            # 3. INITIAL CONFIGURATION
            ####################################################################
            # [drop height, velocity, horizontal position]
            drop_height = self._get_scalar(data, "arg_landing_height", 0.0)
            landing_speed = self._get_scalar(data, "arg_landing_speed", 0.0)
            initial_x_gap = self._get_scalar(data, "arg_initial_x_gap", 0.0)

            init_config = torch.tensor(
                [drop_height, landing_speed, initial_x_gap],
                dtype=torch.float32
            )

            ####################################################################
            # 4. TARGET METRICS
            ####################################################################
            # [number of contacts, disturbance resistance score]
            num_contacts = self._get_scalar(data, "num_contacts", 0.0)
            disturbance_score = self._get_scalar(data, "disturbance_resistance_score", 0.0)

            num_contacts_norm = np.clip(num_contacts / 20.0, 0.0, 1.0)

            target_metrics = torch.tensor(
                [num_contacts_norm, disturbance_score],
                dtype=torch.float32
            )

        return {
            "task_params": task_params,
            "design_params": design_params,
            "init_config": init_config,
            "target_metrics": target_metrics,
        }