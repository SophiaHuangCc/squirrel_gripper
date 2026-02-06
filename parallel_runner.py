


import subprocess
import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

# On Ubuntu, 'spawn' prevents Numba/CUDA-related deadlocks in parallel loops
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

def get_next_exp_folder(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("exp") and d[3:].isdigit()]
    next_num = max([int(d[3:]) for d in existing]) + 1 if existing else 1
    new_folder = os.path.join(base_dir, f"exp{next_num}")
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

def run_simulation(bundle):
    params, exp_dir = bundle
    # Unique ID includes hex to prevent collisions in high-parallelism
    unique_id = f"T{params['tension']}_R{params['base_rad']}_JS{params['joint_softness']}_{os.urandom(2).hex()}"
    
    cmd = ["python3", "finger.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    cmd.extend(["--output_dir", exp_dir, "--suffix", unique_id])
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, params, None
    except subprocess.CalledProcessError as err:
        return False, params, err.stderr

if __name__ == "__main__":
    current_exp_dir = get_next_exp_folder("runs")
    tasks = []

    import argparse
    parser = argparse.ArgumentParser(description="Run Squirrel Finger Parallel.")
    parser.add_argument("--test_mode", type=str, default="full_sweep", choices=["full_sweep", "curvature"], 
                        help="Choose which parameter sweep to run.")
    args = parser.parse_args()


    if args.test_mode == "full_sweep":
        # --- 1. PARAMETER DEFINITIONS ---
        approach_angles = [15, 30, 45, 60, 75]        # 5 steps
        tensions = [3, 5, 7, 10]                      # 4 steps
        cyl_radii = [0.01, 0.015, 0.02]               # 3 steps
        base_radii = [0.005, 0.006, 0.007]            # 3 steps
        joint_softness_vals = [0.01, 0.008]           # 2 steps

        # Vertebrae Configuration (Reduced to 6 each to keep total ~4300)
        uniform_sets = [
            (30, 62), (20, 70), (35, 65), (25, 72), (30, 72)
        ]
        manual_sets = [
            "30,55,70", "20,50,70", "30,60,72", "25,55,65", "40,60,70"
        ]

        # --- 2. TASK GENERATION ---
        # Combinatorics: 12 (sets) * 5 (ang) * 4 (ten) * 3 (rad) * 2 (k) * 3 (br) * 2 (js) = 4,320 tasks
        for deg in approach_angles:
            for t in tensions:
                for cr in cyl_radii:
                    for br in base_radii:
                        for js in joint_softness_vals:
                            # Shared parameters
                            base_p = {
                                "sol": "approach_angle", "approach_deg": deg, 
                                    "tension": t, "cyl_rad": cr, 
                                        "base_rad": br, "joint_softness": js
                                    }
                                    
                            # Add Uniform versions
                            for v_start, v_end in uniform_sets:
                                p_uni = base_p.copy()
                                p_uni.update({"v_mode": "uniform", "v_start": v_start, "v_end": v_end})
                                tasks.append((p_uni, current_exp_dir))
                                
                            # Add Manual versions
                            for v_list in manual_sets:
                                p_man = base_p.copy()
                                p_man.update({"v_mode": "manual", "v_list": v_list})
                                tasks.append((p_man, current_exp_dir))

        # --- 3. EXECUTION SETUP ---
        num_workers = max(1, os.cpu_count() - 2)

    if args.test_mode == "curvature":
        # --- 1. THE CORE THREE SWEEP (Primary Variables) ---
        tensions = [1.0, 3.0, 5.0, 8.0, 10.0, 15.0]       # 6 steps
        base_radii = [0.003, 0.004, 0.005, 0.006, 0.007]  # 5 steps
        softness_vals = [0.01, 0.005, 0.002, 0.001, 0.0005] # 5 steps

        # --- 2. FIXED PARAMETERS (Optimized for Curl) ---
        fixed_params = {
            "sol": "approach_angle",
            "approach_deg": 45.0,     # Your verified successful angle
            "cyl_rad": 0.015,         # Standard branch size
            "v_mode": "uniform",
            "v_start": 20,            # Start hinges earlier for better wrap
            "v_end": 70,
            "E": 2e7,                 # Fixed Young's Modulus
        }

        # --- 3. TASK GENERATION ---
        # Combinations: 6 * 5 * 5 = 150 tasks
        # This is small enough to run very quickly and give you a perfect "Pass/Fail" map
        for t in tensions:
            for br in base_radii:
                for js in softness_vals:
                    p = fixed_params.copy()
                    p.update({
                        "tension": t,
                        "base_rad": br,
                        "joint_softness": js
                    })
                    tasks.append((p, current_exp_dir))

        # --- 4. EXECUTION ---
        num_workers = max(1, os.cpu_count() - 2)
        print(f"--- CURVATURE SENSITIVITY SWEEP ---")
        print(f"Testing {len(tasks)} combinations of Tension, Radius, and Softness...")
    
    # --- TEST MODE SWITCH ---
    # tasks_to_run = tasks[:60]  # Toggle for testing
    tasks_to_run = tasks         # Use for full 4320 samples
    
    print(f"--- UBUNTU LAB SWEEP START ---")
    print(f"Target Folder: {current_exp_dir}")
    print(f"Executing: {len(tasks_to_run)} samples")
    print(f"Parallel Workers: {num_workers}")
    print(f"---------------------------------")

    # --- 4. RUN WITH PROGRESS BAR ---
    start_time = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(run_simulation, tasks_to_run), total=len(tasks_to_run)))
    
    end_time = time.time()
    total_duration = end_time - start_time

    # --- 5. FINAL SUMMARY ---
    successes = sum(1 for r in results if r[0])
    failures = len(results) - successes
    
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    print(f"\n{'='*40}")
    print(f"       SWEEP COMPLETE SUMMARY")
    print(f"{'='*40}")
    print(f"Total Runtime: {hours}h {minutes}m {seconds}s")
    print(f"Average time per task: {total_duration/len(tasks_to_run):.2f}s")
    print(f"Results stored in: {current_exp_dir}")
    print(f"Total Samples: {len(results)}")
    print(f"Success: {successes}")
    print(f"Failure: {failures}")
    print(f"{'='*40}")
    
    if failures > 0:
        print("\nFailure Analysis (First 5):")
        for success, p, err in results:
            if not success:
                print(f"[-] Crash @ Deg:{p['approach_deg']} T:{p['tension']} VM:{p['v_mode']}")
                print(f"Error: {err[-150:] if err else 'Unknown'}\n")

