# dynamics/sim_test_mj.py

import os
import json
import glob
import subprocess
import numpy as np
import ray

from dynamics.utils import design_to_dict

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
            "curl_time": float(
                np.asarray(data.get("curl_time", [np.inf])).reshape(-1)[0]
            ),
            "curl_speed_score": float(
                np.asarray(data.get("curl_speed_score", [0.0])).reshape(-1)[0]
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
            },
            f,
            indent=2,
        )

    cmd = [
        "python3", "finger.py",
        # Match the simulation configuration used to generate the exp3 dataset.
        # Keeping these explicit prevents finger.py defaults from silently
        # changing the dynamics used to verify optimized candidates.
        "--E", "2e7",
        "--damping", "0.1",
        "--n_elements", "100",
        "--final_time", "2.0",
        "--k_contact", "1250.0",
        "--max_penetration_warn", "0.002",
        "--nu_contact", "5.0",
        "--mu_contact", "0.6",
        "--vel_damp_contact", "10",
        "--poisson_nu", "0.4",
        "--v_mass", "0.002",
        "--num_v", "3",
        "--v_height", "0.005",
        "--body_mass", "0.5",
        "--approach_deg", str(approach_deg),
        "--cyl_rad", str(cyl_rad),
        "--v_mode", "manual",
        "--joint_stiffness_mode", "full_material",
        "--v_list", design["v_list_str"],
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
        "--landing_approach_deg", "30.0",
        "--prescribed_stop_at_contact",
        "--prescribed_contact_margin", "0.0",
        "--landing_height", "0.04",
        "--landing_speed", "0.0",
        "--initial_x_gap", "0.06",
        "--base_force_mag", "0.0",
        "--base_force_dir", "0,0,-1",
        "--base_force_nodes", "1",
        "--disturbance_force_mag", "1.0",
        "--disturbance_base_nodes", "5",
        "--disturbance_steps", "40",
        "--disturbance_dt_scale", "1.0",
        "--min_tension", "0.1",
        "--max_tension", "20.0",
        # "--disable_video_plots",
    ]

    # if not render:
    #     cmd.append("--no_render_video")  # only if you added this flag in finger.py

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..", "TendonForces"),
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])

    npz_path = read_latest_npz(run_dir)
    metric = read_metric_from_npz(npz_path)
    summary_npz_path = os.path.join(run_dir, f"finger_{finger_idx}.npz")

    np.savez_compressed(
        summary_npz_path,
        design_params=np.asarray(design_params),
        task_params=np.asarray([]) if task_params is None else np.asarray(task_params),
        metric=metric,
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
