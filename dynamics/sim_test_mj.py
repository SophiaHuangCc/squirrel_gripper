# dynamics/sim_test_mj.py

import os
import json
import glob
import subprocess
import numpy as np
import ray

from dynamics.utils import design_to_dict


def append_finger_arg(cmd, key, value):
    """Append one argparse-compatible finger.py argument."""
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        if bool(value):
            cmd.append(f"--{key}")
        return
    cmd.extend([f"--{key}", str(value)])


def read_latest_npz(run_dir):
    files = glob.glob(os.path.join(run_dir, "**", "master_log_*.npz"), recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"No npz files found in {run_dir}")
    return max(files, key=os.path.getmtime)


def read_metric_from_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        metric = {
            "num_contacts": float(
                np.asarray(data.get("num_contacts", [0.0])).reshape(-1)[0]
            ),
            "disturbance_resistance_score": float(
                np.asarray(data.get("disturbance_resistance_score", [0.0])).reshape(-1)[0]
            ),
            "angular_span": float(
                np.asarray(data.get("angular_span", [0.0])).reshape(-1)[0]
            ),
            "n_elements": float(
                np.asarray(data.get("n_elements", [100.0])).reshape(-1)[0]
            ),
        }

    metric["combined_score"] = (
        metric["disturbance_resistance_score"]
        + 0.1 * metric["num_contacts"]
        + 0.5 * metric["angular_span"]
    )

    return metric


@ray.remote(num_cpus=1)
def sim_test(
    design_params,
    task_params=None,
    finger_idx=0,
    save_dir="sim",
    render=False,
    sim_params=None,
):
    run_dir = os.path.abspath(
        os.path.join(save_dir, f"finger_{finger_idx}")
    )
    os.makedirs(run_dir, exist_ok=True)

    design = design_to_dict(design_params, n_elements=100)

    if task_params is not None:
        approach_deg = float(task_params[0])
        cyl_rad = float(task_params[1])
    else:
        approach_deg = 45.0
        cyl_rad = 0.03

    with open(os.path.join(run_dir, "design.json"), "w") as f:
        json.dump(
            {
                **design,
                "approach_deg": approach_deg,
                "cyl_rad": cyl_rad,
                "sim_params": {} if sim_params is None else dict(sim_params),
            },
            f,
            indent=2,
        )

    cmd = [
        "python3", "finger.py",
        "--approach_deg", str(approach_deg),
        "--cyl_rad", str(cyl_rad),
        "--v_mode", "from_links",
        "--link_lengths", design["link_lengths_str"],
        "--joint_lengths", design["joint_lengths_str"],
        "--joint_softness", design["joint_softness_str"],
        "--base_rad", str(design["base_rad"]),
        "--base_len", str(design["base_len"]),
        "--tension", str(design["tension"]),
        "--ankle_wrap_radius", str(design["ankle_wrap_radius"]),
        "--ankle_stiffness", str(design["ankle_stiffness"]),
        "--output_dir", run_dir,
        "--suffix", f"opt_{finger_idx}",
        "--landing_motion",
        "--landing_mode", "prescribed",
        # "--disable_video_plots",
    ]

    if sim_params is not None:
        for key, value in sim_params.items():
            append_finger_arg(cmd, key, value)

    with open(os.path.join(run_dir, "finger_command.json"), "w") as f:
        json.dump(
            {
                "cwd": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TendonForces")),
                "cmd": cmd,
            },
            f,
            indent=2,
        )

    # if not render:
    #     cmd.append("--no_render_video")  # only if you added this flag in finger.py

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..", "TendonForces"),
    )

    with open(os.path.join(run_dir, "finger_stdout.txt"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(run_dir, "finger_stderr.txt"), "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])

    npz_path = read_latest_npz(run_dir)
    metric = read_metric_from_npz(npz_path)
    used_sim_params = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for key in [
            "damping",
            "nu_contact",
            "vel_damp_contact",
            "k_contact",
            "final_time",
            "E",
            "n_elements",
            "mu_contact",
            "poisson_nu",
            "joint_stiffness_mode",
            "distal_tendon_anchor",
            "distal_tendon_anchor_node",
            "joint_lengths",
            "landing_height",
            "landing_speed",
            "initial_x_gap",
            "landing_approach_deg",
            "prescribed_stop_at_contact",
            "prescribed_contact_margin",
            "min_tension",
            "max_tension",
            "disturbance_force_mag",
            "disturbance_base_nodes",
            "disturbance_steps",
            "disturbance_dt_scale",
            "continuous_disturbance_metric",
        ]:
            arg_key = f"arg_{key}"
            if arg_key in data:
                used_sim_params[key] = np.asarray(data[arg_key]).reshape(-1)[0].item()
            elif key in data:
                used_sim_params[key] = np.asarray(data[key]).reshape(-1)[0].item()

    with open(os.path.join(run_dir, "finger_used_params.json"), "w") as f:
        json.dump(used_sim_params, f, indent=2)

    summary_npz_path = os.path.join(run_dir, f"finger_{finger_idx}.npz")

    np.savez_compressed(
        summary_npz_path,
        design_params=np.asarray(design_params),
        task_params=np.asarray([]) if task_params is None else np.asarray(task_params),
        metric=metric,
        used_sim_params=np.asarray([used_sim_params], dtype=object),
        master_log_path=np.asarray([npz_path]),
    )

    print(f"[sim_test] saved summary npz: {summary_npz_path}")

    return metric, run_dir


def sim_test_batch(
    design_params,
    save_dir,
    num_cpus=8,
    render=False,
    task_params=None,
    sim_params=None,
):
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    design_params = np.asarray(design_params)
    num_fingers = design_params.shape[0]

    if task_params is None:
        task_params = [None] * num_fingers
    else:
        task_params = np.asarray(task_params)

    ray.init(num_cpus=num_cpus, log_to_driver=False, ignore_reinit_error=True)

    ray_tasks = []
    for finger_idx, design in enumerate(design_params):
        task = None if task_params is None else task_params[finger_idx]
        ray_tasks.append(
            sim_test.remote(
                design_params=design,
                task_params=task,
                finger_idx=finger_idx,
                save_dir=save_dir,
                render=render,
                sim_params=sim_params,
            )
        )

    metrics = {}
    save_dirs = {}

    while ray_tasks:
        ready, ray_tasks = ray.wait(ray_tasks, num_returns=1)
        try:
            metric, run_dir = ray.get(ready[0])
            idx = len(metrics)
            metrics[idx] = metric
            save_dirs[idx] = run_dir
        except Exception as e:
            print("[sim_test_batch error]", e)

    ray.shutdown()

    metrics = [v for _, v in sorted(metrics.items())]
    save_dirs = [v for _, v in sorted(save_dirs.items())]

    return metrics, save_dirs
