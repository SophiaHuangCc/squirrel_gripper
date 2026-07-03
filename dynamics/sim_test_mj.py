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
        # Stable replay profile used before the two-second tuning attempt.
        # Keep these explicit so future finger.py defaults cannot silently
        # change optimized-candidate verification.
        "--E", "2e7",
        "--damping", "0.8",
        "--n_elements", "100",
        "--final_time", "4.0",
        "--time_step_safety", "0.1",
        "--k_contact", "1250.0",
        "--max_penetration_warn", "0.002",
        "--nu_contact", "5.0",
        "--mu_contact", "0.6",
        "--vel_damp_contact", "2",
        "--poisson_nu", "0.4",
        "--v_mass", "0.002",
        "--num_v", "3",
        "--v_height", "0.005",
        "--body_mass", "0.5",
        "--approach_deg", str(approach_deg),
        "--cyl_rad", str(cyl_rad),
        # The dataset command used uniform nodes [38, 59, 80]. Its v_list
        # argument was ignored by finger.py.
        "--v_mode", "uniform",
        "--v_start", "38",
        "--v_end", "80",
        "--joint_stiffness_mode", "full_material",
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
        # Avoid commanding the base 5 mm inside the nominal contact boundary;
        # that negative dataset margin creates a harsher impact for optimized
        # geometries and is not an optimized design variable.
        "--prescribed_contact_margin", "0.0",
        "--landing_height", "0.03",
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
        # This caps the feedback-amplified tension, not merely the nominal
        # optimized tension. A nominal 3.5 N could otherwise rise to 20 N.
        "--max_tension", "6.0",
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
        failure_log = os.path.join(run_dir, "simulation_failure.log")
        with open(failure_log, "w") as f:
            f.write(result.stdout)
            f.write("\n\n--- STDERR ---\n")
            f.write(result.stderr)

        # Keep batch verification alive and make numerical failures rank below
        # every physically valid candidate.
        metric = {
            "num_contacts": 0.0,
            "disturbance_resistance_score": 0.0,
            "angular_span": 0.0,
            "curl_time": float("inf"),
            "curl_speed_score": 0.0,
            "n_elements": 100.0,
            "combined_score": float("-inf"),
            "simulation_stable": False,
        }
        print(
            f"[sim_test] unstable candidate {finger_idx}; "
            f"saved diagnostics to {failure_log}"
        )
        return metric, run_dir

    npz_path = read_latest_npz(run_dir)
    metric = read_metric_from_npz(npz_path)
    metric["simulation_stable"] = True
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
