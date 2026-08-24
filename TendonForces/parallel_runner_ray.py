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


def format_float_csv(values, ndigits=4):
    return ",".join(f"{float(x):.{ndigits}f}".rstrip("0").rstrip(".") for x in values)


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


def geometry_total_cm(link_lengths_cm, joint_lengths_cm):
    return float(sum(link_lengths_cm) + sum(joint_lengths_cm))


def get_geometry_choices():
    """Explicit from_links geometry candidates.

    All candidates keep the three soft-joint lengths fixed at 2 cm each.
    The link lengths are written directly instead of generated from hidden
    fractions, and each row is validated against base_len.
    """
    fixed_joint_lengths_cm = [2.0, 2.0, 2.0]
    choices = [
        # Current run.sh geometry: 6.6 + 2 + 2.4 + 3 links, plus 6 cm joints = 20 cm.
        {"name": "runsh_20cm", "base_len": 0.20, "link_lengths": [6.6, 2.0, 2.4, 3.0]},

        # Shorter 15 cm candidates.
        {"name": "short_even_15cm", "base_len": 0.15, "link_lengths": [4.0, 1.5, 1.5, 2.0]},
        {"name": "short_longbase_15cm", "base_len": 0.15, "link_lengths": [4.8, 1.2, 1.3, 1.7]},
        {"name": "short_distal_15cm", "base_len": 0.15, "link_lengths": [3.5, 1.5, 1.8, 2.2]},

        # 20 cm candidates around the manufactured design.
        {"name": "balanced_20cm", "base_len": 0.20, "link_lengths": [5.5, 2.5, 3.0, 3.0]},
        {"name": "longbase_20cm", "base_len": 0.20, "link_lengths": [7.5, 1.8, 2.2, 2.5]},
        {"name": "distalwrap_20cm", "base_len": 0.20, "link_lengths": [5.8, 1.8, 2.9, 3.5]},

        # Longer 25 cm candidates.
        {"name": "balanced_25cm", "base_len": 0.25, "link_lengths": [7.5, 3.5, 4.0, 4.0]},
        {"name": "longbase_25cm", "base_len": 0.25, "link_lengths": [10.0, 2.5, 3.0, 3.5]},
        {"name": "distalwrap_25cm", "base_len": 0.25, "link_lengths": [7.5, 2.5, 4.0, 5.0]},

        # 30 cm candidates, useful as bad/edge examples and long-reach tasks.
        {"name": "balanced_30cm", "base_len": 0.30, "link_lengths": [9.0, 4.5, 5.0, 5.5]},
        {"name": "longbase_30cm", "base_len": 0.30, "link_lengths": [13.0, 3.0, 3.5, 4.5]},
        {"name": "distalwrap_30cm", "base_len": 0.30, "link_lengths": [8.5, 3.5, 5.5, 6.5]},
    ]

    for choice in choices:
        total_cm = geometry_total_cm(choice["link_lengths"], fixed_joint_lengths_cm)
        expected_cm = 100.0 * choice["base_len"]
        if abs(total_cm - expected_cm) > 1e-9:
            raise ValueError(
                f"Invalid geometry choice {choice['name']}: "
                f"links + joints = {total_cm:.3f} cm, "
                f"but base_len = {expected_cm:.3f} cm"
            )
        choice["joint_lengths"] = list(fixed_joint_lengths_cm)

    return choices


def build_design_only_sample(rng):
    params = {
        "E": 6.74e6,
        "damping": 0.1,
        "n_elements": 0,
        "final_time": 2.0,
        "k_contact": 500.0,
        "max_penetration_warn": 0.002,
        "nu_contact": 20.0,
        "mu_contact": 0.8,
        "vel_damp_contact": 30,
        "poisson_nu": 0.49,
        "v_mass": 0.04,
        "num_v": 3,
        "v_start": 38,
        "v_end": 80,
        "v_height": 0.002,
        "body_mass": 3.0,
        "landing_motion": True,
        "landing_mode": "prescribed",
        "landing_approach_deg": 30.0,
        "prescribed_stop_at_contact": True,
        "prescribed_contact_margin": -0.005,
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
        "disturbance_force_mag": 5.0,
        "disturbance_base_nodes": 5,
        "disturbance_steps": 100,
        "disturbance_dt_scale": 1.0,
        "continuous_disturbance_metric": True,
        "min_tension": 0.1,
        # Keep this at/above the sweep maximum, otherwise finger.py may clamp
        # high-tension samples even though the sampled design says 25 N.
        "max_tension": 30.0,
        "distal_tendon_anchor": "tip",
        "v_mode": "from_links",
        "joint_stiffness_mode": "bending_only",
        "data_only": True,
        "base_rad": 0.01,
        "cross_section": "rect",
        "base_width": 0.03,
        "landing_speed": 0.0,
        "ankle_wrap_radius": 0.03,
        "ankle_stiffness": 500.0,
    }

    geometry_choices = get_geometry_choices()
    cyl_rad_choices = [0.020, 0.025, 0.030, 0.035]
    initial_x_gap_choices = [0.06, 0.10, 0.15, 0.20]
    landing_height_choices = [0.02, 0.04, 0.06]
    landing_approach_deg_choices = [15.0, 30.0, 45.0]
    base_thickness_choices = [0.015, 0.018, 0.020, 0.023, 0.025]
    tension_choices = [5.0, 7.5, 10.0, 12.5, 14.7, 17.5, 20.0, 22.5, 25.0]
    approach_angle_choices = list(np.arange(5.0, 90.0, 10.0))  # 0-90 exclusive
    joint_E_choices_mpa = [
        [0.05, 0.04, 0.03],
        [0.08, 0.06, 0.04],
        [0.10, 0.08, 0.06],  # run.sh value
        [0.15, 0.12, 0.09],
        [0.20, 0.16, 0.12],
        [0.30, 0.20, 0.15],
    ]

    geometry = sample_from_list(rng, geometry_choices)
    params["base_len"] = maybe_round(geometry["base_len"], 4)
    params["cyl_rad"] = maybe_round(sample_from_list(rng, cyl_rad_choices), 4)
    params["initial_x_gap"] = maybe_round(sample_from_list(rng, initial_x_gap_choices), 4)
    params["landing_height"] = maybe_round(sample_from_list(rng, landing_height_choices), 4)
    params["landing_approach_deg"] = maybe_round(sample_from_list(rng, landing_approach_deg_choices), 4)
    params["base_thickness"] = maybe_round(sample_from_list(rng, base_thickness_choices), 4)
    params["tension"] = maybe_round(sample_from_list(rng, tension_choices), 4)
    params["approach_deg"] = maybe_round(sample_from_list(rng, approach_angle_choices), 4)

    joint_E_mpa = sample_from_list(rng, joint_E_choices_mpa)

    params["geometry_name"] = geometry["name"]
    params["link_lengths"] = format_float_csv(geometry["link_lengths"], ndigits=3)
    params["joint_lengths"] = format_float_csv(geometry["joint_lengths"], ndigits=3)
    params["joint_E"] = format_float_csv(joint_E_mpa, ndigits=4)

    return params


def generate_all_dataset_tasks(n_total, train_dir, test_dir, seed, num_train):
    rng = np.random.default_rng(seed)
    samples = []
    seen = set()

    while len(samples) < n_total:
        p = build_design_only_sample(rng)

        key = (
            p["cyl_rad"],
            p["initial_x_gap"],
            p["landing_height"],
            p["landing_approach_deg"],
            p["base_len"],
            p["base_thickness"],
            p["tension"],
            p["approach_deg"],
            p["link_lengths"],
            p["joint_lengths"],
            p["joint_E"],
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
        f"CR{params['cyl_rad']}_"
        f"XG{params['initial_x_gap']}_"
        f"LH{params['landing_height']}_"
        f"LA{params['landing_approach_deg']}_"
        f"BL{params['base_len']}_"
        f"BT{params['base_thickness']}_"
        f"A{params['approach_deg']}_"
        f"G{params['geometry_name']}_"
        f"JE{params['joint_E'].replace(',', '-')}_"
        f"LL{params['link_lengths'].replace(',', '-')}_"
        f"JL{params['joint_lengths'].replace(',', '-')}_"
        f"{timestamp}"
    )

    cmd = ["python3", "finger.py"]
    internal_keys = {"geometry_name"}

    for key, value in params.items():
        if key in internal_keys:
            continue
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
                    f"BL:{p.get('base_len')} "
                    f"BT:{p.get('base_thickness')} "
                    f"A:{p.get('approach_deg')} "
                    f"JE:{p.get('joint_E')} "
                    f"LL:{p.get('link_lengths')} "
                    f"JL:{p.get('joint_lengths')}"
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
    parser.add_argument("--num_train", type=int, default=8000)
    parser.add_argument("--num_test", type=int, default=2000)
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
