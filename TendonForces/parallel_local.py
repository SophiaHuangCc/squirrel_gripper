import subprocess
import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Ensure Mac uses the correct process start method
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

def run_simulation(params):
    """Function to run a single instance of the simulation."""
    sol, e, t, rad, ang = params
    
    # Create a unique suffix to prevent file overwriting
    # Replaces dots with 'p' to keep filenames clean (e.g., 0.01 -> 0p01)
    suffix = f"{sol}_E{e:.1e}_T{t}_R{rad}_A{ang}".replace(".", "p")
    
    cmd = [
        "python3", "finger.py",
        "--sol", sol,
        "--E", f"{e:.2e}",
        "--tension", str(t),
        "--cyl_rad", str(rad),
        "--approach_deg", f"{ang:.2f}",
        "--suffix", suffix,
        "--output_dir", "runs",
    ]
    
    try:
        # capture_output=True keeps the terminal clean
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
                        tasks.append((sol, e, t, rad, ang))

    # --- 2. Parallel Execution Setup ---
    # For Mac, 75% of cores is the "sweet spot" for speed vs system stability
    num_workers = max(1, int(os.cpu_count() * 0.75)) 
    tasks_to_run = tasks[:20]
    
    print(f"--- SQUIRREL GRIP SWEEP START ---")
    print(f"Total Tasks: {len(tasks)}")
    print(f"Parallel Workers: {num_workers}")
    print(f"---------------------------------")

    # --- 3. Run with Progress Bar ---
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # list(tqdm(...)) wraps the generator to show a live progress bar
        # results = list(tqdm(executor.map(run_simulation, tasks), total=len(tasks)))
        results = list(tqdm(executor.map(run_simulation, tasks_to_run), total=len(tasks_to_run)))


    # --- 4. Final Summary ---
    successes = sum(1 for r in results if r[0])
    # failures = len(tasks) - successes
    failures = len(results) - successes
    
    print(f"\n--- SWEEP COMPLETE ---")
    print(f"Success: {successes}")
    print(f"Failure: {failures}")
    
    if failures > 0:
        print("\nFirst 5 Failure Details:")
        fail_count = 0
        for success, params, err in results:
            if not success and fail_count < 5:
                print(f"Params {params} failed with error:\n{err[:200]}...")
                fail_count += 1