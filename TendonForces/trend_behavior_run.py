import os
import glob
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

OUT_DIR = "trend_test_results"
os.makedirs(OUT_DIR, exist_ok=True)

import csv

BASE_LEN_CM = 20.0   # finger length = 20 cm
N_ELEMENTS = 100

CSV_PATH = os.path.join(OUT_DIR, "final_pose_summary.csv")

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "tension_N",
            "v_list",
            "joint_softness",
            "joint1_location_cm",
            "joint2_location_cm",
            "joint3_location_cm",
            "base_x_m", "base_y_m", "base_z_m",
            "joint1_x_m", "joint1_y_m", "joint1_z_m",
            "joint2_x_m", "joint2_y_m", "joint2_z_m",
            "joint3_x_m", "joint3_y_m", "joint3_z_m",
            "tip_x_m", "tip_y_m", "tip_z_m",
        ])

BASE_CMD = [
    "python", "./finger.py",
    "--approach_deg", "45.0",
    "--base_rad", "0.01025",
    "--v_mode", "manual",
    "--E", "2e7",
    "--damping", "0.1",
    "--n_elements", "100",
    "--final_time", "2.0",
    "--cyl_rad", "0.03",
    "--k_contact", "1250.0",
    "--max_penetration_warn", "0.002",
    "--base_len", "0.20",
    "--nu_contact", "20.0",
    "--mu_contact", "0.8",
    "--vel_damp_contact", "30",
    "--poisson_nu", "0.4",
    "--v_mass", "0.002",
    "--num_v", "3",
    "--v_start", "38",
    "--v_end", "80",
    "--v_height", "0.005",
    "--body_mass", "0.5",
    "--output_dir", OUT_DIR,
    "--landing_motion",
    "--landing_mode", "prescribed",
    "--ankle_wrap_radius", "0.02",
    "--ankle_stiffness", "500.0",
    "--min_tension", "0.1",
    "--max_tension", "20.0",
    "--landing_speed", "0.0",
    "--initial_x_gap", "0.06",
    "--landing_height", "0.04",
    "--landing_approach_deg", "30.0",
    "--prescribed_stop_at_contact",
    "--prescribed_contact_margin", "-0.005",
    "--base_force_mag", "0.0",
    "--base_force_dir", "0,0,-1",
    "--base_force_nodes", "1",
    "--disturbance_force_mag", "5.0",
    "--disturbance_base_nodes", "5",
    "--disturbance_steps", "100",
    "--disturbance_dt_scale", "1.0",
    "--continuous_disturbance_metric",
]


def latest_npz():
    files = glob.glob(os.path.join(OUT_DIR, "master_log_*.npz"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def v_list_to_cm(v_list):
    nodes = [int(x.strip()) for x in v_list.split(",")]
    return [node / N_ELEMENTS * BASE_LEN_CM for node in nodes]


def run_case(name, tension, v_list, joint_softness):
    print("\n" + "=" * 80)
    print(f"Running case: {name}")
    print(f"  tension = {tension}")
    print(f"  v_list = {v_list}")
    print(f"  joint_softness = {joint_softness}")
    print("=" * 80)

    before = set(glob.glob(os.path.join(OUT_DIR, "master_log_*.npz")))

    cmd = BASE_CMD + [
        "--tension", str(tension),
        "--v_list", v_list,
        "--joint_softness", joint_softness,
        "--suffix", name,
    ]

    subprocess.run(cmd, check=True)

    after = set(glob.glob(os.path.join(OUT_DIR, "master_log_*.npz")))
    new_files = list(after - before)
    npz_path = new_files[0] if new_files else latest_npz()

    if npz_path is None:
        print(f"[WARNING] No npz found for {name}")
        return None

    with np.load(npz_path, allow_pickle=True) as data:
        base = data["final_base_position"]
        joints = data["final_joint_positions"]
        tip = data["final_tip_position"]

    with np.load(npz_path, allow_pickle=True) as data:
        base = data["final_base_position"]
        joints = data["final_joint_positions"]
        tip = data["final_tip_position"]

    joint_locations_cm = v_list_to_cm(v_list)

    print(f"\n[FINAL POSE] {name}")
    print(f"  base:  {base}")
    for i, p in enumerate(joints):
        print(f"  joint {i + 1} at {joint_locations_cm[i]:.2f} cm: {p}")
    print(f"  tip:   {tip}")

    with open(CSV_PATH, "a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([
          name,
          tension,
          v_list,
          joint_softness,
          *joint_locations_cm,
          *base.tolist(),
          *joints[0].tolist(),
          *joints[1].tolist(),
          *joints[2].tolist(),
          *tip.tolist(),
      ])

    return {
        "name": name,
        "npz": npz_path,
        "base": base,
        "joints": joints,
        "tip": tip,
        "joint_locations_cm": joint_locations_cm,
    }


def plot_cases(cases, title, filename):
    plt.figure(figsize=(8, 6))

    for case in cases:
        if case is None:
            continue

        pts = np.vstack([
            case["base"].reshape(1, 3),
            case["joints"],
            case["tip"].reshape(1, 3),
        ])

        # X-Z projection, most useful for curl shape
        joint_cm = case["joint_locations_cm"]
        label = f"{case['name']} | joints={joint_cm[0]:.1f},{joint_cm[1]:.1f},{joint_cm[2]:.1f} cm"
        plt.plot(pts[:, 0], pts[:, 2], "-o", label=label)

    plt.xlabel("X position (m)")
    plt.ylabel("Z position (m)")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[OK] saved plot: {path}")


def main():
    all_cases = []

    # --------------------------------------------------
    # 1. Softness sweep
    # fixed v_list = [30, 58, 80], tension = 3N
    # --------------------------------------------------
    softness_cases = []
    for softness in [
        "0.005,0.005,0.005",
        "0.003,0.003,0.003",
        "0.003,0.002,0.001",
    ]:
        name = f"softness_{softness.replace(',', '-')}"
        case = run_case(
            name=name,
            tension=3.0,
            v_list="30,58,80",
            joint_softness=softness,
        )
        softness_cases.append(case)
        all_cases.append(case)

    plot_cases(
        softness_cases,
        "Final Pose Trend: Joint Softness Sweep",
        "softness_sweep_final_pose.png",
    )

    # --------------------------------------------------
    # 2. Joint position sweep
    # fixed softness = [0.003, 0.003, 0.003], tension = 3N
    # --------------------------------------------------
    position_cases = []
    for v_list in [
        "30,58,80",
        "38,58,80",
        "40,65,85",
        "45,70,90",
    ]:
        name = f"joints_{v_list.replace(',', '-')}"
        case = run_case(
            name=name,
            tension=3.0,
            v_list=v_list,
            joint_softness="0.003,0.003,0.003",
        )
        position_cases.append(case)
        all_cases.append(case)

    plot_cases(
        position_cases,
        "Final Pose Trend: Joint Position Sweep",
        "joint_position_sweep_final_pose.png",
    )

    # --------------------------------------------------
    # 3. Tension sweep
    # fixed default softness and position
    # 6 designs: tension 1N to 6N
    # --------------------------------------------------
    tension_cases = []
    for tension in range(1, 7):
        name = f"tension_{tension}N"
        case = run_case(
            name=name,
            tension=float(tension),
            v_list="38,58,80",
            joint_softness="0.003,0.002,0.001",
        )
        tension_cases.append(case)
        all_cases.append(case)

    plot_cases(
        tension_cases,
        "Final Pose Trend: Tension Sweep",
        "tension_sweep_final_pose.png",
    )

    plot_cases(
        all_cases,
        "Final Pose Trend: All Quick Tests",
        "all_trend_tests_final_pose.png",
    )


if __name__ == "__main__":
    main()