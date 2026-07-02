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
        curl_contact_ratio=0.8,
        curl_hold_time=0.2,
        curl_min_contacts=3,
        **kwargs,
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.curl_contact_ratio = float(curl_contact_ratio)
        self.curl_hold_time = float(curl_hold_time)
        self.curl_min_contacts = int(curl_min_contacts)
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

    def _derive_curl_speed_score(self, data, base_radius):
        """Backfill curl speed from trajectories in archives created before this metric."""
        if "curl_speed_score" in data:
            return float(np.clip(self._get_scalar(data, "curl_speed_score", 0.0), 0.0, 1.0))

        final_time = self._get_scalar(data, "arg_final_time", 4.0)
        if final_time <= 0.0 or "position" not in data or "cyl_position" not in data:
            return 0.0

        positions = np.asarray(data["position"])
        if positions.ndim != 3 or positions.shape[1] < 3:
            return 0.0

        center = np.asarray(data["cyl_position"]).reshape(3, -1)[:, 0]
        cyl_radius = self._get_scalar(data, "cyl_radius", 0.015)
        dx = positions[:, 0, :] - center[0]
        dz = positions[:, 2, :] - center[2]
        radial_dist = np.sqrt(dx ** 2 + dz ** 2)
        contact_counts = np.sum(
            radial_dist < (cyl_radius + float(base_radius)), axis=1
        )

        peak_contacts = int(contact_counts.max()) if contact_counts.size else 0
        threshold = max(
            self.curl_min_contacts,
            int(np.ceil(self.curl_contact_ratio * peak_contacts)),
        )
        if peak_contacts < threshold:
            return 0.0

        if "time" in data:
            times = np.asarray(data["time"], dtype=float).reshape(-1)
        else:
            times = np.linspace(0.0, final_time, len(contact_counts))
        if len(times) != len(contact_counts):
            return 0.0

        dt = float(np.median(np.diff(times))) if len(times) > 1 else final_time
        hold_frames = max(1, int(np.ceil(self.curl_hold_time / max(dt, 1e-12))))
        meets = contact_counts >= threshold
        for frame_idx in range(0, len(meets) - hold_frames + 1):
            if np.all(meets[frame_idx:frame_idx + hold_frames]):
                curl_time = float(times[frame_idx])
                return 1.0 - float(np.clip(curl_time / final_time, 0.0, 1.0))
        return 0.0

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
            n_elements = int(round(self._get_scalar(data, "n_elements", 100.0)))

            link_lengths = self._compute_link_lengths_from_joint_positions(
                joint_positions=joint_positions,
                base_length=base_length,
                n_elements=n_elements
            )

            design_params_np = np.concatenate([
                joint_softness.astype(np.float32) / 0.001,      # e.g. 3 values
                link_lengths.astype(np.float32) / 0.3,         # e.g. 4 values
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
            curl_speed_score = self._derive_curl_speed_score(data, base_radius)

            num_contacts_norm = np.log1p(num_contacts) / np.log1p(n_elements)
            angular_span_norm = np.clip(angular_span / 180.0, 0.0, 1.0)
            # angular_span_norm = np.where(
            #     angular_span <= 180.0,
            #     0.8 * angular_span / 180.0,
            #     0.8 + 0.2 * np.clip((angular_span - 180.0) / 180.0, 0.0, 1.0)
            # )

            target_metrics = torch.tensor(
                [num_contacts_norm, disturbance_score, angular_span_norm, curl_speed_score],
                dtype=torch.float32
            )

        return {
            "task_params": task_params,
            "design_params": design_params,
            "init_config": init_config,
            "target_metrics": target_metrics,
        }
