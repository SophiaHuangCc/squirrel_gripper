import os
import json
import numpy as np
import torch


def sigmoid_to_range(x, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(x)


def link_lengths_to_v_list(link_lengths, joint_lengths=None, base_length=None, n_elements=100):
    """
    Convert 4 physical link lengths into 3 vertebra node indices.
    """
    link_lengths = np.asarray(link_lengths, dtype=np.float32).reshape(-1)
    if base_length is None:  # backwards-compatible positional call
        base_length = joint_lengths
        joint_lengths = np.zeros(len(link_lengths) - 1, dtype=np.float32)
    joint_lengths = np.asarray(joint_lengths, dtype=np.float32).reshape(-1)
    centers = []
    cursor = 0.0
    for link, joint in zip(link_lengths[:-1], joint_lengths):
        cursor += float(link) + 0.5 * float(joint)
        centers.append(cursor)
        cursor += 0.5 * float(joint)
    joints = np.round(np.asarray(centers) / float(base_length) * n_elements).astype(int)
    joints = np.clip(joints, 1, n_elements - 1)
    return joints.tolist()


def design_to_dict(design_params, task_params=None, n_elements=100):
    """
    Convert one 16D physical From Links design vector into finger.py args.

    design_params:
      [0:3]  joint_softness
      [3:7]  link_lengths
      [7:10] joint_lengths
      [10]   base_radius
      [11]   base_thickness
      [12]   base_length
      [13]   tension
      [14]   ankle_wrap_radius
      [15]   ankle_stiffness

    task_params:
      [0] approach_deg
      [1] landing_approach_deg
      [2] cyl_radius
    """
    d = np.asarray(design_params, dtype=np.float32).reshape(-1)
    if d.size != 16:
        raise ValueError(
            f"Expected a 16D From Links design, got {d.size} values. "
            "Legacy 12D checkpoints/candidates must be retrained or regenerated."
        )

    joint_softness = d[0:3]
    link_lengths = d[3:7]
    joint_lengths = d[7:10]
    base_radius = float(d[10])
    base_thickness = float(d[11])
    base_length = float(d[12])
    tension = float(d[13])
    ankle_wrap_radius = float(d[14])
    ankle_stiffness = float(d[15])

    v_list = link_lengths_to_v_list(
        link_lengths=link_lengths,
        joint_lengths=joint_lengths,
        base_length=base_length,
        n_elements=n_elements,
    )

    out = {
        "joint_softness": joint_softness.tolist(),
        "joint_softness_str": ",".join([f"{x:.6f}" for x in joint_softness]),
        "link_lengths": link_lengths.tolist(),
        "link_lengths_str": ",".join([f"{100*x:.6g}" for x in link_lengths]),
        "joint_lengths": joint_lengths.tolist(),
        "joint_lengths_str": ",".join([f"{100*x:.6g}" for x in joint_lengths]),
        "v_mode": "from_links",
        "v_list": v_list,
        "v_list_str": ",".join([str(x) for x in v_list]),
        "base_rad": base_radius,
        "base_thickness": base_thickness,
        "base_len": base_length,
        "tension": tension,
        "ankle_wrap_radius": ankle_wrap_radius,
        "ankle_stiffness": ankle_stiffness,
    }

    if task_params is not None:
        t = np.asarray(task_params, dtype=np.float32).reshape(-1)
        out["approach_deg"] = float(t[0])
        out["landing_approach_deg"] = float(t[1])
        out["cyl_rad"] = float(t[2])

    return out


def finger_forward(raw_params, args):
    """
    Convert raw optimizer params into:
      task_params   : (B, 3)  = [approach_deg, landing_approach_deg, cyl_rad]
      design_params : (B, 16)

    raw_params is 16D: 15 design parameters plus approach angle.
    """
    if raw_params.dim() == 1:
        raw_params = raw_params.unsqueeze(0)

    x = raw_params

    joint_softness = sigmoid_to_range(
        x[:, 0:3],
        args.joint_soft_min,
        args.joint_soft_max,
    )

    raw_links = sigmoid_to_range(
        x[:, 3:7],
        args.link_min,
        args.link_max,
    )

    base_radius = sigmoid_to_range(
        x[:, 10:11],
        args.base_radius_min,
        args.base_radius_max,
    )

    base_length = sigmoid_to_range(
        x[:, 11:12],
        args.base_length_min,
        args.base_length_max,
    )

    joint_lengths = sigmoid_to_range(x[:, 7:10], args.joint_length_min, args.joint_length_max)
    available = (base_length - joint_lengths.sum(dim=-1, keepdim=True)).clamp_min(1e-6)
    link_lengths = raw_links / torch.sum(raw_links, dim=-1, keepdim=True) * available

    tension = sigmoid_to_range(
        x[:, 12:13],
        args.tension_min,
        args.tension_max,
    )

    ankle_wrap_radius = sigmoid_to_range(
        x[:, 13:14],
        args.ankle_wrap_min,
        args.ankle_wrap_max,
    )

    ankle_stiffness = sigmoid_to_range(
        x[:, 14:15],
        args.ankle_stiff_min,
        args.ankle_stiff_max,
    )

    design_params = torch.cat(
        [
            joint_softness,
            link_lengths,
            joint_lengths,
            base_radius,
            base_length,
            tension,
            ankle_wrap_radius,
            ankle_stiffness,
        ],
        dim=-1,
    )

    approach_deg = sigmoid_to_range(
        x[:, 15:16],
        args.approach_deg_min,
        args.approach_deg_max,
    )

    cyl_rad = torch.full_like(approach_deg, float(args.cyl_rad))

    task_params = torch.cat(
        [
            approach_deg,
            cyl_rad,
        ],
        dim=-1,
    )

    return task_params, design_params


def save_finger(design_params, save_finger_dir, args, task_params=None):
    """
    Save optimized squirrel-finger design in a format sim_test/finger.py can use.

    This is your project's analog of save_gripper(...).
    Instead of saving OBJ meshes, we save:
      - design.json
      - design.npz
    """
    os.makedirs(save_finger_dir, exist_ok=True)

    design_params = np.asarray(design_params, dtype=np.float32)
    if design_params.ndim == 2:
        design_params_one = design_params[0]
    else:
        design_params_one = design_params.reshape(-1)

    task_params_one = None
    if task_params is not None:
        task_params = np.asarray(task_params, dtype=np.float32)
        if task_params.ndim == 2:
            task_params_one = task_params[0]
        else:
            task_params_one = task_params.reshape(-1)

    design_dict = design_to_dict(
        design_params_one,
        task_params=task_params_one,
        n_elements=getattr(args, "n_elements", 100),
    )

    with open(os.path.join(save_finger_dir, "design.json"), "w") as f:
        json.dump(design_dict, f, indent=2)

    np.savez_compressed(
        os.path.join(save_finger_dir, "design.npz"),
        design_params=design_params_one,
        task_params=np.asarray([]) if task_params_one is None else task_params_one,
        joint_softness=np.asarray(design_dict["joint_softness"]),
        link_lengths=np.asarray(design_dict["link_lengths"]),
        joint_lengths=np.asarray(design_dict["joint_lengths"]),
        v_mode=np.asarray(["from_links"]),
        v_list=np.asarray(design_dict["v_list"]),
        base_rad=np.asarray([design_dict["base_rad"]]),
        base_len=np.asarray([design_dict["base_len"]]),
        tension=np.asarray([design_dict["tension"]]),
        ankle_wrap_radius=np.asarray([design_dict["ankle_wrap_radius"]]),
        ankle_stiffness=np.asarray([design_dict["ankle_stiffness"]]),
    )

    return design_dict
