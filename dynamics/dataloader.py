import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DynamicsDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        **kwargs,
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        # self.files = glob.glob(os.path.join(self.dataset_dir, "*.npz"), recursive=True)
        self.files = sorted(
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

    def _get_from_links_geometry(self, data, base_length, n_elements):
        """Read the explicit free-link/joint geometry, with legacy fallback."""
        link_key = "arg_link_lengths" if "arg_link_lengths" in data else "link_lengths"
        joint_key = "arg_joint_lengths" if "arg_joint_lengths" in data else "joint_lengths"
        v_mode = ""
        if "arg_v_mode" in data:
            v_mode = str(np.asarray(data["arg_v_mode"]).reshape(-1)[0])

        if link_key in data and joint_key in data:
            links_cm = self._get_array(data, link_key, default=[])
            joints_cm = self._get_array(data, joint_key, default=[])
            if links_cm.size == 4 and joints_cm.size in {1, 3}:
                if joints_cm.size == 1:
                    joints_cm = np.repeat(joints_cm, 3)
                links = links_cm.astype(np.float32) * 1e-2
                joints = joints_cm.astype(np.float32) * 1e-2
                total = float(links.sum() + joints.sum())
                if not np.isclose(total, base_length, rtol=0.0, atol=max(1e-5, base_length / n_elements)):
                    raise ValueError(
                        f"Invalid from_links geometry: links + joints = {total:.6f} m, "
                        f"base_length = {base_length:.6f} m"
                    )
                return links, joints

        if v_mode == "from_links":
            raise ValueError("from_links sample is missing four link lengths and three joint lengths")

        positions = self._get_array(data, "vertebra_nodes", default=[30, 46, 62])
        links = self._compute_link_lengths_from_joint_positions(positions, base_length, n_elements)
        # Legacy archives used an 8-element softened window.
        joints = np.full(3, 8.0 * base_length / n_elements, dtype=np.float32)
        # Legacy link spacing includes joint material. Convert it to free lengths.
        links = links.copy()
        links[0] -= 0.5 * joints[0]
        links[-1] -= 0.5 * joints[-1]
        links[1:-1] -= 0.5 * (joints[:-1] + joints[1:])
        return np.maximum(links, 1e-6), joints

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=True) as data:
            ####################################################################
            # 1. TASK PARAMETERS
            ####################################################################
            # [approach angle, cylinder radius]
            approach_angle = self._get_scalar(data, "arg_approach_deg", 0.0)
            cylinder_radius = self._get_scalar(data, "cyl_radius", 0.015)

            task_params = torch.tensor(
                [approach_angle / 90.0, cylinder_radius / 0.05],
                dtype=torch.float32
            )

            ####################################################################
            # 2. DESIGN PARAMETERS
            ####################################################################
            # Joint stiffness ratios. New datasets specify physical joint E.
            joint_softness = self._get_array(
                data,
                "joint_softness",
                default=[0.001, 0.001, 0.001]
            )
            joint_e_key = "arg_joint_E" if "arg_joint_E" in data else "joint_E"
            if joint_e_key in data:
                joint_e = self._get_array(data, joint_e_key)
                if joint_e.size == 1:
                    joint_e = np.repeat(joint_e, 3)
                if np.all(joint_e <= 1000.0):
                    joint_e = joint_e * 1e6
                base_e = self._get_scalar(data, "arg_E", self._get_scalar(data, "E", 6.74e6))
                joint_softness = joint_e / base_e

            # base geometry
            base_radius = self._get_scalar(data, "base_radius", 0.005)
            base_length = self._get_scalar(data, "base_length", 0.10)
            tension = self._get_scalar(data, "tension", 0.0)
            ankle_wrap_radius = self._get_scalar(data, "arg_ankle_wrap_radius", 0.005)
            ankle_stiffness = self._get_scalar(data, "arg_ankle_stiffness", 500.0)

            n_elements = int(round(self._get_scalar(data, "n_elements", 100.0)))
            link_lengths, joint_lengths = self._get_from_links_geometry(
                data, base_length=base_length, n_elements=n_elements
            )

            design_params_np = np.concatenate([
                joint_softness.astype(np.float32) / 0.001,      # e.g. 3 values
                link_lengths.astype(np.float32) / 0.3,         # e.g. 4 values
                joint_lengths.astype(np.float32) / 0.05,       # 3 finite joints
                np.asarray([
                    base_radius / 0.02,
                    base_length / 0.2,
                    tension / 10.0,
                    ankle_wrap_radius / 0.025,
                    ankle_stiffness / 1000.0,
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
                [drop_height / 0.10, landing_speed / 1.0, initial_x_gap / 0.1],
                dtype=torch.float32
            )

            ####################################################################
            # 4. TARGET METRICS
            ####################################################################
            # [number of contacts, disturbance resistance score]
            num_contacts = self._get_scalar(data, "num_contacts", 0.0)
            disturbance_score = self._get_scalar(data, "disturbance_resistance_score", 0.0)
            angular_span = self._get_scalar(data, "angular_span", 0.0)

            num_contacts_norm = np.log1p(num_contacts) / np.log1p(n_elements)
            angular_span_norm = np.clip(angular_span / 180.0, 0.0, 1.0)
            # angular_span_norm = np.where(
            #     angular_span <= 180.0,
            #     0.8 * angular_span / 180.0,
            #     0.8 + 0.2 * np.clip((angular_span - 180.0) / 180.0, 0.0, 1.0)
            # )

            target_metrics = torch.tensor(
                [num_contacts_norm, disturbance_score, angular_span_norm],
                dtype=torch.float32
            )

        return {
            "task_params": task_params,
            "design_params": design_params,
            "init_config": init_config,
            "target_metrics": target_metrics,
        }
