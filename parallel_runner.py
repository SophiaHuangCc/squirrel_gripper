import subprocess
import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# On Ubuntu, 'spawn' prevents Numba/CUDA-related deadlocks in parallel loops
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

def get_next_exp_folder(base_dir="runs"):
    """Finds the next available exp folder (exp1, exp2, ...)"""
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("exp") and d[3:].isdigit()]
    if not existing:
        next_num = 1
    else:
        nums = [int(d[3:]) for d in existing]
        next_num = max(nums) + 1
    new_folder = os.path.join(base_dir, f"exp{next_num}")
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

def run_simulation(params_bundle):
    """Function to run a single instance of the simulation."""
    # Unpack the bundle: (params_tuple, destination_folder)
    params, exp_dir = params_bundle
    sol, e, t, rad, ang = params
    
    # Create a unique suffix to prevent file overwriting
    suffix = f"{sol}_E{e:.1e}_T{t}_R{rad}_A{ang}".replace(".", "p")
    
    cmd = [
        "python3", "finger.py",
        "--sol", sol,
        "--E", f"{e:.2e}",
        "--tension", str(t),
        "--cyl_rad", str(rad),
        "--approach_deg", f"{ang:.2f}",
        "--suffix", suffix,
        "--output_dir", exp_dir, # Now points to runs/expN
    ]
    
    try:
        # We capture output to check for those 'numerical explosions' in the logs
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return True, params, None
    except subprocess.CalledProcessError as err:
        return False, params, err.stderr

if __name__ == "__main__":
    # --- 0. Setup Experiment Directory ---
    current_exp_dir = get_next_exp_folder("runs")

    # --- 1. Generate Task List ---
    tasks = []
    e_values = np.linspace(1e7, 5e7, 4)
    cyl_radii = [0.01, 0.015]
    approach_angles = np.linspace(0, 90, 8)
    tensions_map = {
        "approach_angle": [2.0, 3.0, 5.0], 
        "nonuniform_tendon": [0.1, 0.2, 0.4]
    }

    for sol, tensions in tensions_map.items():
        for e in e_values:
            for t in tensions:
                for rad in cyl_radii:
                    for ang in approach_angles:
                        # Pack params and the directory together
                        tasks.append(((sol, e, t, rad, ang), current_exp_dir))

    # --- 2. Parallel Execution Setup ---
    num_workers = max(1, os.cpu_count() - 2) 
    
    # --- TEST MODE SWITCH ---
    tasks_to_run = tasks[:60] 
    # tasks_to_run = tasks 
    
    print(f"--- UBUNTU LAB SWEEP START ---")
    print(f"Target Folder: {current_exp_dir}")
    print(f"Total Tasks in Queue: {len(tasks)}")
    print(f"Executing: {len(tasks_to_run)} samples")
    print(f"Parallel Workers: {num_workers}")
    print(f"---------------------------------")

    # --- 3. Run with Progress Bar ---
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tasks_to_run (which now contains bundles)
        results = list(tqdm(executor.map(run_simulation, tasks_to_run), total=len(tasks_to_run)))

    # --- 4. Final Summary ---
    successes = sum(1 for r in results if r[0])
    failures = len(results) - successes
    
    print(f"\n--- SWEEP COMPLETE ---")
    print(f"Results stored in: {current_exp_dir}")
    print(f"Total Completed: {len(results)}")
    print(f"Success: {successes}")
    print(f"Failure: {failures}")
    
    if failures > 0:
        print("\nFailure Analysis (First 5):")
        fail_count = 0
        for success, params, err in results:
            if not success and fail_count < 5:
                print(f"[-] Params {params} crashed. Tail of error log:")
                print(f"{err[-300:]}\n")
                fail_count += 1