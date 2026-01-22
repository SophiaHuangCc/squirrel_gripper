import subprocess
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

def run_simulation(params):
    """Function to run a single instance of the simulation."""
    sol, e, t, rad, ang = params
    cmd = [
        "python", "finger.py",
        "--sol", sol,
        "--E", f"{e:.2e}",
        "--tension", str(t),
        "--cyl_rad", str(rad),
        "--approach_deg", f"{ang:.2f}"
    ]
    try:
        # We use subprocess.DEVNULL to keep the terminal from getting messy
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return f"SUCCESS: {params}"
    except Exception as err:
        return f"ERROR: {params} | {err}"

if __name__ == "__main__":
    # --- 1. Generate Task List ---
    tasks = []
    e_values = np.linspace(1e7, 5e7, 4)
    cyl_radii = [0.01, 0.015]
    approach_angles = np.linspace(0, 90, 8)
    tensions_map = {"approach_angle": [2.0, 3.0, 5.0], "nonuniform_tendon": [0.1, 0.2, 0.4]}

    for sol, tensions in tensions_map.items():
        for e in e_values:
            for t in tensions:
                for rad in cyl_radii:
                    for ang in approach_angles:
                        tasks.append((sol, e, t, rad, ang))

    # --- 2. Parallel Execution ---
    # Set max_workers to the number of CPU cores (e.g., 32 or 64 on a lab server)
    num_workers = os.cpu_count() - 2  # Leave 2 cores free for the OS
    print(f"Launching sweep with {num_workers} parallel workers for {len(tasks)} tasks...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_simulation, tasks))

    print("Sweep Complete.")