import os
import json
import numpy as np
import torch


def sigmoid_to_range(x, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(x)


def link_lengths_to_v_list(link_lengths, base_length, n_elements=100):
    """
    Convert 4 physical link lengths into 3 vertebra node indices.
    """
    link_lengths = np.asarray(link_lengths, dtype=np.float32).reshape(-1)
    cum = np.cumsum(link_lengths[:-1])
    joints = np.round(cum / float(base_length) * n_elements).astype(int)
    joints = np.clip(joints, 1, n_elements - 1)
    return joints.tolist()


def design_to_dict(design_params, task_params=None, n_elements=100):
    """
    Convert one 12D physical design vector into finger.py-friendly args.

    design_params:
      [0:3]  joint_softness
      [3:7]  link_lengths
      [7]    base_radius
      [8]    base_length
      [9]    tension
      [10]   ankle_wrap_radius
      [11]   ankle_stiffness

    task_params:
      [0] approach_deg
      [1] cyl_radius
    """
    d = np.asarray(design_params, dtype=np.float32).reshape(-1)

    joint_softness = d[0:3]
    link_lengths = d[3:7]
    base_radius = float(d[7])
    base_length = float(d[8])
    tension = float(d[9])
    ankle_wrap_radius = float(d[10])
    ankle_stiffness = float(d[11])

    v_list = link_lengths_to_v_list(
        link_lengths=link_lengths,
        base_length=base_length,
        n_elements=n_elements,
    )

    out = {
        "joint_softness": joint_softness.tolist(),
        "joint_softness_str": ",".join([f"{x:.6f}" for x in joint_softness]),
        "link_lengths": link_lengths.tolist(),
        "v_list": v_list,
        "v_list_str": ",".join([str(x) for x in v_list]),
        "base_rad": base_radius,
        "base_len": base_length,
        "tension": tension,
        "ankle_wrap_radius": ankle_wrap_radius,
        "ankle_stiffness": ankle_stiffness,
    }

    if task_params is not None:
        t = np.asarray(task_params, dtype=np.float32).reshape(-1)
        out["approach_deg"] = float(t[0])
        out["cyl_rad"] = float(t[1])

    return out


def finger_forward(raw_params, args):
    """
    Convert raw optimizer params into:
      task_params   : (B, 2)  = [approach_deg, cyl_rad]
      design_params : (B, 12)

    raw_params is 13D:
      [0:12] design raw params
      [12]   approach angle raw param
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
        x[:, 7:8],
        args.base_radius_min,
        args.base_radius_max,
    )

    base_length = sigmoid_to_range(
        x[:, 8:9],
        args.base_length_min,
        args.base_length_max,
    )

    # Make 4 links sum to base_length.
    link_lengths = raw_links / torch.sum(raw_links, dim=-1, keepdim=True) * base_length

    tension = sigmoid_to_range(
        x[:, 9:10],
        args.tension_min,
        args.tension_max,
    )

    ankle_wrap_radius = sigmoid_to_range(
        x[:, 10:11],
        args.ankle_wrap_min,
        args.ankle_wrap_max,
    )

    ankle_stiffness = sigmoid_to_range(
        x[:, 11:12],
        args.ankle_stiff_min,
        args.ankle_stiff_max,
    )

    design_params = torch.cat(
        [
            joint_softness,
            link_lengths,
            base_radius,
            base_length,
            tension,
            ankle_wrap_radius,
            ankle_stiffness,
        ],
        dim=-1,
    )

    approach_deg = sigmoid_to_range(
        x[:, 12:13],
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
        v_list=np.asarray(design_dict["v_list"]),
        base_rad=np.asarray([design_dict["base_rad"]]),
        base_len=np.asarray([design_dict["base_len"]]),
        tension=np.asarray([design_dict["tension"]]),
        ankle_wrap_radius=np.asarray([design_dict["ankle_wrap_radius"]]),
        ankle_stiffness=np.asarray([design_dict["ankle_stiffness"]]),
    )

    return design_dict