import subprocess
import numpy as np
import os
from tqdm import tqdm
import time
import ray
import argparse
import json
from datetime import datetime


def get_next_exp_folder(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("exp") and d[3:].isdigit()]
    next_num = max([int(d[3:]) for d in existing]) + 1 if existing else 1
    new_folder = os.path.join(base_dir, f"exp{next_num}")
    os.makedirs(new_folder, exist_ok=True)
    return new_folder


def maybe_round(x, ndigits=4):
    return round(float(x), ndigits)


def sample_from_list(rng, values):
    return values[rng.integers(0, len(values))]


def sample_joint_positions(rng, n_elements, min_first=40, max_last=90, min_gap=10):
    """
    Sample 3 vertebra/joint positions:
      min_first <= j1 < j2 < j3 <= max_last
      adjacent spacing >= min_gap
    """
    max_last = min(max_last, n_elements - 1)

    # Need room for j1, j2, j3 with two gaps
    max_j1 = max_last - 2 * min_gap
    if max_j1 < min_first:
        raise ValueError(
            f"No valid joint positions: min_first={min_first}, "
            f"max_last={max_last}, min_gap={min_gap}"
        )

    j1 = int(rng.integers(min_first, max_j1 + 1))

    max_j2 = max_last - min_gap
    j2 = int(rng.integers(j1 + min_gap, max_j2 + 1))

    j3 = int(rng.integers(j2 + min_gap, max_last + 1))

    return [j1, j2, j3]


def generate_joint_positions_from_links(base_len, link_lengths, n_elements):
    """
    Convert 4 link lengths into 3 joint positions in element indices.
    """
    cum = np.cumsum(link_lengths[:-1])
    joint_positions = np.round(cum / base_len * n_elements).astype(int)
    joint_positions = np.clip(joint_positions, 2, n_elements - 2)

    for i in range(1, len(joint_positions)):
        if joint_positions[i] <= joint_positions[i - 1]:
            joint_positions[i] = joint_positions[i - 1] + 1

    joint_positions = np.clip(joint_positions, 2, n_elements - 2)
    return joint_positions.tolist()


def build_design_only_sample(rng, split="train"):
    """
    Sweep only design parameters.
    Task params and initial configs are fixed to the known-good/default run.
    """

    # -------------------------
    # Fixed parameters
    # -------------------------
    params = {
        "E": 2e7,
        "damping": 0.1,
        "n_elements": 100,
        "final_time": 2.0,
        "k_contact": 1250.0,
        # "auto_contact_stiffness": True,
        "max_penetration_warn": 0.002,
        "nu_contact": 5.0,
        "mu_contact": 0.6,
        "vel_damp_contact": 10,
        "poisson_nu": 0.4,
        "v_mass": 0.002,
        "num_v": 3,
        "v_height": 0.005,
        "body_mass": 0.5,
        "landing_motion": True,
        "landing_mode": "prescribed",
        "landing_approach_deg": 30.0,
        "prescribed_stop_at_contact": True,
        "prescribed_contact_margin": 0.0,
        "base_force_mag": 0.0,
        "base_force_dir": "0,0,-1",
        "base_force_nodes": 1,
        "force_driven_stabilize": True,
        "force_driven_xy_k": 120.0,
        "force_driven_xy_c": 3.0,
        "force_driven_tendon_ramp": 1.0,
        "force_driven_xy_fmax": 5.0,
        "force_driven_lock_base_xy": True,
        "force_driven_z_stabilize": True,
        "force_driven_z_k": 120.0,
        "force_driven_z_c": 12.0,
        "force_driven_z_fmax": 4.0,
        "force_driven_z_target": "cylinder",
        "force_driven_z_target_offset": -0.01,
        "force_driven_min_damping": 5.0,
        "force_driven_node_drag": 4.0,
        "force_driven_node_drag_axes": "1,1,1",
        "force_driven_rot_stabilize": True,
        "force_driven_rot_k": 0.03,
        "force_driven_rot_c": 0.02,
        "force_driven_rot_tmax": 0.02,
        "disturbance_force_mag": 1.0,
        "disturbance_base_nodes": 5,
        "disturbance_steps": 40,
        "disturbance_dt_scale": 1.0,
        "min_tension": 0.1,
        "max_tension": 20.0,
        "v_mode": "manual",

        # -------------------------
        # Fixed task params
        # -------------------------
        # "approach_deg": 45.0,
        "cyl_rad": 0.03,

        # -------------------------
        # Fixed initial config
        # -------------------------
        "landing_height": 0.04,
        "landing_speed": 0.0,
        "initial_x_gap": 0.06,
    }

    # -------------------------
    # Design-only sweep ranges
    # -------------------------
    if split == "train":
        base_rad_choices = [0.01025, 0.011, 0.012, 0.013]
        base_len_choices = [0.15, 0.20, 0.25]
        tension_choices = [2.0, 3.0, 4.0, 5.0, 6.0]
        ankle_wrap_choices = [0.015, 0.020, 0.025]
        ankle_stiff_choices = [300.0, 500.0, 700.0]
        joint_soft_choices = [
            [0.005, 0.004, 0.003],
            [0.003, 0.002, 0.001],
            [0.002, 0.001, 0.0009],
            [0.001, 0.0009, 0.0008],
            [0.0009, 0.0008, 0.0007],
        ]
        # Added new choice of approach angle
        approach_angle_choices = [45.0, 60.0, 75.0]
    else:
        # Slightly shifted but still nearby
        base_rad_choices = [0.01025, 0.0115, 0.0125]
        base_len_choices = [0.2, 0.3]
        tension_choices = [1.0, 2.5, 4.5]
        ankle_wrap_choices = [0.0175, 0.0225]
        ankle_stiff_choices = [400.0, 600.0]
        joint_soft_choices = [
            [0.0045, 0.0035, 0.0025],
            [0.0035, 0.0025, 0.0015],
            [0.0025, 0.0015, 0.0005],
        ]
        # Added new choice of approach angle
        approach_angle_choices = [50.0, 65.0, 70.0]

    # -------------------------
    # Sample design params
    # -------------------------
    params["base_rad"] = maybe_round(sample_from_list(rng, base_rad_choices), 4)
    params["base_len"] = maybe_round(sample_from_list(rng, base_len_choices), 4)
    params["tension"] = maybe_round(sample_from_list(rng, tension_choices), 4)
    params["ankle_wrap_radius"] = maybe_round(sample_from_list(rng, ankle_wrap_choices), 4)
    params["ankle_stiffness"] = maybe_round(sample_from_list(rng, ankle_stiff_choices), 4)
    params["approach_deg"] = maybe_round(sample_from_list(rng, approach_angle_choices), 4)

    js = sample_from_list(rng, joint_soft_choices)
    params["joint_softness"] = ",".join([f"{x:.6f}" for x in js])

    joint_positions = sample_joint_positions(
        rng,
        n_elements=params["n_elements"],
        min_first=30,
        max_last=95,
        min_gap=20,
    )

    params["v_list"] = ",".join([str(x) for x in joint_positions])

    return params


def generate_dataset_tasks(n_samples, exp_dir, seed, split_name):
    rng = np.random.default_rng(seed)
    tasks = []
    seen = set()

    while len(tasks) < n_samples:
        p = build_design_only_sample(rng, split=split_name)

        key = (
            p["base_rad"],
            p["base_len"],
            p["tension"],
            p["ankle_wrap_radius"],
            p["ankle_stiffness"],
            p["joint_softness"],
            p["v_list"],
        )
        if key in seen:
            continue

        seen.add(key)
        tasks.append((p, exp_dir))

    return tasks


@ray.remote(num_cpus=1)
def run_simulation(bundle):
    params, exp_dir = bundle

    # compact design-only ID + timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_id = (
        f"T{params['tension']}_"
        f"BR{params['base_rad']}_"
        f"BL{params['base_len']}_"
        f"AR{params['ankle_wrap_radius']}_"
        f"AK{params['ankle_stiffness']}_"
        f"JS{params['joint_softness'].replace(',', '-')}_"
        f"JP{params['v_list'].replace(',', '-')}_"
        f"{timestamp}"
    )

    cmd = ["python3", "finger.py"]
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    cmd.extend(["--output_dir", exp_dir, "--suffix", unique_id])

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, params, None
    except subprocess.CalledProcessError as err:
        return False, params, err.stderr


def _drain_ready(ready_refs):
    results = []
    for ref in ready_refs:
        try:
            results.append(ray.get(ref))
        except Exception as exc:
            results.append((False, {}, str(exc)))
    return results


def run_split(tasks_to_run, current_exp_dir, num_workers, split_label):
    print(f"--- UBUNTU LAB SWEEP START ---")
    print(f"Split: {split_label}")
    print(f"Target Folder: {current_exp_dir}")
    print(f"Executing: {len(tasks_to_run)} samples")
    print(f"Parallel Workers: {num_workers}")
    print(f"---------------------------------")

    ray.init(num_cpus=num_workers, log_to_driver=False, include_dashboard=False, ignore_reinit_error=True,)

    start_time = time.time()
    results = []
    pending = [run_simulation.remote(task) for task in tasks_to_run]

    with tqdm(total=len(pending), desc=split_label) as progress:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            results.extend(_drain_ready(ready))
            progress.update(len(ready))

    ray.shutdown()
    end_time = time.time()
    total_duration = end_time - start_time

    successes = sum(1 for r in results if r[0])
    failures = len(results) - successes

    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    print(f"\n{'='*40}")
    print(f"       SWEEP COMPLETE SUMMARY")
    print(f"{'='*40}")
    print(f"Split: {split_label}")
    print(f"Total Runtime: {hours}h {minutes}m {seconds}s")
    print(f"Average time per task: {total_duration/len(tasks_to_run):.2f}s")
    print(f"Results stored in: {current_exp_dir}")
    print(f"Total Samples: {len(results)}")
    print(f"Success: {successes}")
    print(f"Failure: {failures}")
    print(f"{'='*40}")

    if failures > 0:
        print("\nFailure Analysis (First 5):")
        shown = 0
        for success, p, err in results:
            if not success:
                print(
                    f"[-] Crash @ "
                    f"T:{p.get('tension')} "
                    f"BR:{p.get('base_rad')} "
                    f"BL:{p.get('base_len')} "
                    f"JS:{p.get('joint_softness')} "
                    f"JP:{p.get('v_list')}"
                )
                print(f"Error: {err[-150:] if err else 'Unknown'}\n")
                shown += 1
                if shown >= 5:
                    break

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate squirrel finger dataset with design-only sweep.")
    parser.add_argument("--num_train", type=int, default=400, help="Number of training samples")
    parser.add_argument("--num_test", type=int, default=100, help="Number of testing samples")
    parser.add_argument("--train_seed", type=int, default=123, help="Random seed for train set")
    parser.add_argument("--test_seed", type=int, default=456, help="Random seed for test set")
    parser.add_argument("--num_cpus", type=int, default=None, help="Override Ray CPU count")
    args = parser.parse_args()

    train_dir = get_next_exp_folder("runs")
    test_dir = get_next_exp_folder("runs")

    train_tasks = generate_dataset_tasks(
        n_samples=args.num_train,
        exp_dir=train_dir,
        seed=args.train_seed,
        split_name="train",
    )
    test_tasks = generate_dataset_tasks(
        n_samples=args.num_test,
        exp_dir=test_dir,
        seed=args.test_seed,
        split_name="test",
    )

    num_workers = max(1, os.cpu_count() - 2)
    if args.num_cpus is not None:
        num_workers = max(1, args.num_cpus)

    with open(os.path.join(train_dir, "split_info.json"), "w") as f:
        json.dump({"split": "train", "n_samples": len(train_tasks), "seed": args.train_seed}, f, indent=2)

    with open(os.path.join(test_dir, "split_info.json"), "w") as f:
        json.dump({"split": "test", "n_samples": len(test_tasks), "seed": args.test_seed}, f, indent=2)

    run_split(train_tasks, train_dir, num_workers, "train")
    run_split(test_tasks, test_dir, num_workers, "test")