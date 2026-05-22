import argparse
import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import ray
from tqdm import tqdm


def get_next_exp_folder(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if d.startswith("exp") and d[3:].isdigit()
    ]

    next_num = max([int(d[3:]) for d in existing]) + 1 if existing else 1
    exp_dir = os.path.join(base_dir, f"exp{next_num}")

    train_dir = os.path.join(exp_dir, "train")
    test_dir = os.path.join(exp_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return exp_dir, train_dir, test_dir


def maybe_round(x, ndigits=4):
    return round(float(x), ndigits)


def sample_from_list(rng, values):
    return values[int(rng.integers(0, len(values)))]


def sample_joint_positions(rng, n_elements, min_first=30, max_last=95, min_gap=20):
    max_last = min(max_last, n_elements - 1)

    max_j1 = max_last - 2 * min_gap
    if max_j1 < min_first:
        raise ValueError(
            f"No valid joint positions: min_first={min_first}, "
            f"max_last={max_last}, min_gap={min_gap}"
        )

    j1 = int(rng.integers(min_first, max_j1 + 1))
    j2 = int(rng.integers(j1 + min_gap, max_last - min_gap + 1))
    j3 = int(rng.integers(j2 + min_gap, max_last + 1))

    return [j1, j2, j3]


def build_design_only_sample(rng):
    params = {
        "E": 2e7,
        "damping": 0.1,
        "n_elements": 100,
        "final_time": 2.0,
        "k_contact": 1250.0,
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
        "cyl_rad": 0.03,
        "landing_height": 0.04,
        "landing_speed": 0.0,
        "initial_x_gap": 0.06,
    }

    base_rad_choices = [0.01025, 0.011, 0.0115, 0.012, 0.0125, 0.013]
    base_len_choices = [0.15, 0.20, 0.25, 0.30]
    tension_choices = [1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 6.0]
    ankle_wrap_choices = [0.015, 0.0175, 0.020, 0.0225, 0.025]
    ankle_stiff_choices = [300.0, 400.0, 500.0, 600.0, 700.0]
    approach_angle_choices = [45.0, 50.0, 60.0, 65.0, 70.0, 75.0]

    joint_soft_choices = [
        [0.005, 0.004, 0.003],
        [0.0045, 0.0035, 0.0025],
        [0.0035, 0.0025, 0.0015],
        [0.003, 0.002, 0.001],
        [0.0025, 0.0015, 0.0005],
        [0.002, 0.001, 0.0009],
        [0.001, 0.0009, 0.0008],
        [0.0009, 0.0008, 0.0007],
    ]

    params["base_rad"] = maybe_round(sample_from_list(rng, base_rad_choices), 5)
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


def generate_all_dataset_tasks(n_total, train_dir, test_dir, seed, num_train):
    rng = np.random.default_rng(seed)
    samples = []
    seen = set()

    while len(samples) < n_total:
        p = build_design_only_sample(rng)

        key = (
            p["base_rad"],
            p["base_len"],
            p["tension"],
            p["ankle_wrap_radius"],
            p["ankle_stiffness"],
            p["joint_softness"],
            p["v_list"],
            p["approach_deg"],
        )

        if key in seen:
            continue

        seen.add(key)
        samples.append(p)

    rng.shuffle(samples)

    train_samples = samples[:num_train]
    test_samples = samples[num_train:]

    train_tasks = [(p, train_dir) for p in train_samples]
    test_tasks = [(p, test_dir) for p in test_samples]

    return train_tasks, test_tasks


@ray.remote(num_cpus=1)
def run_simulation(bundle):
    params, output_dir = bundle

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_id = (
        f"T{params['tension']}_"
        f"BR{params['base_rad']}_"
        f"BL{params['base_len']}_"
        f"AR{params['ankle_wrap_radius']}_"
        f"AK{params['ankle_stiffness']}_"
        f"A{params['approach_deg']}_"
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

    cmd.extend(["--output_dir", output_dir, "--suffix", unique_id])

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


def run_split(tasks_to_run, output_dir, num_workers, split_label):
    print("\n--- UBUNTU LAB SWEEP START ---")
    print(f"Split: {split_label}")
    print(f"Target Folder: {output_dir}")
    print(f"Executing: {len(tasks_to_run)} samples")
    print(f"Parallel Workers: {num_workers}")
    print("---------------------------------")

    ray.init(
        num_cpus=num_workers,
        log_to_driver=False,
        include_dashboard=False,
        ignore_reinit_error=True,
    )

    start_time = time.time()
    results = []
    pending = [run_simulation.remote(task) for task in tasks_to_run]

    with tqdm(total=len(pending), desc=split_label) as progress:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            results.extend(_drain_ready(ready))
            progress.update(len(ready))

    ray.shutdown()

    total_duration = time.time() - start_time
    successes = sum(1 for r in results if r[0])
    failures = len(results) - successes

    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    print(f"\n{'=' * 40}")
    print("       SWEEP COMPLETE SUMMARY")
    print(f"{'=' * 40}")
    print(f"Split: {split_label}")
    print(f"Total Runtime: {hours}h {minutes}m {seconds}s")
    print(f"Average time per task: {total_duration / max(1, len(tasks_to_run)):.2f}s")
    print(f"Results stored in: {output_dir}")
    print(f"Total Samples: {len(results)}")
    print(f"Success: {successes}")
    print(f"Failure: {failures}")
    print(f"{'=' * 40}")

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
                    f"A:{p.get('approach_deg')} "
                    f"JS:{p.get('joint_softness')} "
                    f"JP:{p.get('v_list')}"
                )
                print(f"Error: {err[-500:] if err else 'Unknown'}\n")
                shown += 1
                if shown >= 5:
                    break

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate squirrel finger dataset with pooled random train/test split."
    )
    parser.add_argument("--num_train", type=int, default=400)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_cpus", type=int, default=None)
    args = parser.parse_args()

    exp_dir, train_dir, test_dir = get_next_exp_folder("runs")

    total_samples = args.num_train + args.num_test

    train_tasks, test_tasks = generate_all_dataset_tasks(
        n_total=total_samples,
        train_dir=train_dir,
        test_dir=test_dir,
        seed=args.seed,
        num_train=args.num_train,
    )

    num_workers = max(1, os.cpu_count() - 2)
    if args.num_cpus is not None:
        num_workers = max(1, args.num_cpus)

    metadata = {
        "exp_dir": exp_dir,
        "num_total": total_samples,
        "num_train": args.num_train,
        "num_test": args.num_test,
        "seed": args.seed,
        "num_workers": num_workers,
        "split_method": "generate_all_then_random_shuffle_split",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(train_dir, "split_info.json"), "w") as f:
        json.dump(
            {
                "split": "train",
                "n_samples": len(train_tasks),
                "seed": args.seed,
                "parent_exp_dir": exp_dir,
            },
            f,
            indent=2,
        )

    with open(os.path.join(test_dir, "split_info.json"), "w") as f:
        json.dump(
            {
                "split": "test",
                "n_samples": len(test_tasks),
                "seed": args.seed,
                "parent_exp_dir": exp_dir,
            },
            f,
            indent=2,
        )

    run_split(train_tasks, train_dir, num_workers, "train")
    run_split(test_tasks, test_dir, num_workers, "test")