"""Continuous finger-pose targets and smooth geometric DGDM objectives."""

import numpy as np
import torch

POSE_POINT_NAMES = ("base", "joint_1", "joint_2", "joint_3", "tip")
POSE_POINT_COUNT = len(POSE_POINT_NAMES)
POSE_OUTPUT_DIM = 2 * POSE_POINT_COUNT
DEFAULT_POSE_SCALE_M = 0.10


def pose_target_from_npz(data, scale_m=DEFAULT_POSE_SCALE_M):
    required = ("final_base_position", "final_joint_positions",
                "final_tip_position", "cyl_position")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError("Pose target requires NPZ fields: " + ", ".join(missing))
    base = np.asarray(data["final_base_position"], dtype=np.float32).reshape(1, 3)
    joints = np.asarray(data["final_joint_positions"], dtype=np.float32).reshape(-1, 3)
    tip = np.asarray(data["final_tip_position"], dtype=np.float32).reshape(1, 3)
    if joints.shape != (3, 3):
        raise ValueError(f"Expected three final joint positions, found {joints.shape}")
    center = np.asarray(data["cyl_position"], dtype=np.float32).reshape(3, -1)[:, 0]
    points = np.concatenate((base, joints, tip), axis=0) - center[None, :]
    if not np.isfinite(points).all():
        raise ValueError("Non-finite final finger/cylinder pose")
    return (points[:, (0, 2)] / float(scale_m)).reshape(-1).astype(np.float32)


def pose_points(prediction):
    if prediction.shape[-1] != POSE_OUTPUT_DIM:
        raise ValueError(f"Expected {POSE_OUTPUT_DIM} pose outputs, got {prediction.shape[-1]}")
    return prediction.reshape(*prediction.shape[:-1], POSE_POINT_COUNT, 2)


def pose_joint_angles_deg(prediction):
    points = pose_points(prediction)
    links = points[..., 1:, :] - points[..., :-1, :]
    headings = torch.atan2(links[..., 1], links[..., 0])
    bends = torch.atan2(torch.sin(headings[..., 1:] - headings[..., :-1]),
                        torch.cos(headings[..., 1:] - headings[..., :-1]))
    return headings * (180.0 / torch.pi), bends * (180.0 / torch.pi)


def pose_geometric_metrics(prediction, task_params, contact_sigma_m=0.005,
                           scale_m=DEFAULT_POSE_SCALE_M):
    """Smooth C/D/A geometric proxies; final evaluation must remain simulator-based."""
    points = pose_points(prediction)
    eps = torch.finfo(points.dtype).eps
    radius = task_params[..., 2] * (0.05 / float(scale_m))
    radial = torch.linalg.vector_norm(points, dim=-1).clamp_min(eps)
    sigma = float(contact_sigma_m) / float(scale_m)
    contact_weights = torch.exp(-0.5 * ((radial - radius[..., None]) / sigma) ** 2)
    contact = contact_weights.mean(dim=-1)
    angles = torch.atan2(points[..., 1], points[..., 0])
    increments = torch.atan2(torch.sin(angles[..., 1:] - angles[..., :-1]),
                             torch.cos(angles[..., 1:] - angles[..., :-1]))
    span = increments.abs().sum(dim=-1).div(2.0 * torch.pi).clamp(0.0, 1.0)
    normals = points / radial[..., None]
    weighted_normal = (contact_weights[..., None] * normals).sum(dim=-2)
    balance = 1.0 - torch.linalg.vector_norm(weighted_normal, dim=-1) / contact_weights.sum(dim=-1).clamp_min(eps)
    disturbance = (contact * balance).clamp(0.0, 1.0)
    return torch.stack((contact, disturbance, span), dim=-1)


def surrogate_metrics(model, prediction, task_params):
    if getattr(model, "target_representation", "metrics") == "pose_keypoints":
        return pose_geometric_metrics(
            prediction, task_params,
            contact_sigma_m=getattr(model, "pose_contact_sigma_m", 0.005),
            scale_m=getattr(model, "pose_scale_m", DEFAULT_POSE_SCALE_M),
        )
    return prediction.clamp(0.0, 1.0)
