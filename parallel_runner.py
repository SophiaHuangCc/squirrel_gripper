# import subprocess
# import numpy as np
# import os
# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm

# # On Ubuntu, 'spawn' prevents Numba/CUDA-related deadlocks in parallel loops
# if __name__ == "__main__":
#     try:
#         multiprocessing.set_start_method('spawn', force=True)
#     except RuntimeError:
#         pass

# def get_next_exp_folder(base_dir="runs"):
#     """Finds the next available exp folder (exp1, exp2, ...)"""
#     os.makedirs(base_dir, exist_ok=True)
#     existing = [d for d in os.listdir(base_dir) if d.startswith("exp") and d[3:].isdigit()]
#     if not existing:
#         next_num = 1
#     else:
#         nums = [int(d[3:]) for d in existing]
#         next_num = max(nums) + 1
#     new_folder = os.path.join(base_dir, f"exp{next_num}")
#     os.makedirs(new_folder, exist_ok=True)
#     return new_folder

# def run_simulation(params_bundle):
#     """Function to run a single instance of the simulation."""
#     # Unpack the bundle: (params_tuple, destination_folder)
#     params, exp_dir = params_bundle
#     sol, e, t, rad, ang = params
    
#     # Create a unique suffix to prevent file overwriting
#     suffix = f"{sol}_E{e:.1e}_T{t}_R{rad}_A{ang}".replace(".", "p")
    
#     cmd = [
#         "python3", "finger.py",
#         "--sol", sol,
#         "--E", f"{e:.2e}",
#         "--tension", str(t),
#         "--cyl_rad", str(rad),
#         "--approach_deg", f"{ang:.2f}",
#         "--suffix", suffix,
#         "--output_dir", exp_dir, # Now points to runs/expN
#     ]
    
#     try:
#         # We capture output to check for those 'numerical explosions' in the logs
#         result = subprocess.run(
#             cmd, 
#             capture_output=True, 
#             text=True, 
#             check=True
#         )
#         return True, params, None
#     except subprocess.CalledProcessError as err:
#         return False, params, err.stderr

# if __name__ == "__main__":
#     # --- 0. Setup Experiment Directory ---
#     current_exp_dir = get_next_exp_folder("runs")

#     # --- 1. Generate Task List ---
#     tasks = []
#     e_values = np.linspace(1e7, 5e7, 4)
#     cyl_radii = [0.01, 0.015]
#     approach_angles = np.linspace(0, 90, 8)
#     tensions_map = {
#         "approach_angle": [2.0, 3.0, 5.0], 
#         "nonuniform_tendon": [0.1, 0.2, 0.4]
#     }

#     for sol, tensions in tensions_map.items():
#         for e in e_values:
#             for t in tensions:
#                 for rad in cyl_radii:
#                     for ang in approach_angles:
#                         # Pack params and the directory together
#                         tasks.append(((sol, e, t, rad, ang), current_exp_dir))

#     # --- 2. Parallel Execution Setup ---
#     num_workers = max(1, os.cpu_count() - 2) 
    
#     # --- TEST MODE SWITCH ---
#     tasks_to_run = tasks[:60] 
#     # tasks_to_run = tasks 
    
#     print(f"--- UBUNTU LAB SWEEP START ---")
#     print(f"Target Folder: {current_exp_dir}")
#     print(f"Total Tasks in Queue: {len(tasks)}")
#     print(f"Executing: {len(tasks_to_run)} samples")
#     print(f"Parallel Workers: {num_workers}")
#     print(f"---------------------------------")

#     # --- 3. Run with Progress Bar ---
#     results = []
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         # Use tasks_to_run (which now contains bundles)
#         results = list(tqdm(executor.map(run_simulation, tasks_to_run), total=len(tasks_to_run)))

#     # --- 4. Final Summary ---
#     successes = sum(1 for r in results if r[0])
#     failures = len(results) - successes
    
#     print(f"\n--- SWEEP COMPLETE ---")
#     print(f"Results stored in: {current_exp_dir}")
#     print(f"Total Completed: {len(results)}")
#     print(f"Success: {successes}")
#     print(f"Failure: {failures}")
    
#     if failures > 0:
#         print("\nFailure Analysis (First 5):")
#         fail_count = 0
#         for success, params, err in results:
#             if not success and fail_count < 5:
#                 print(f"[-] Params {params} crashed. Tail of error log:")
#                 print(f"{err[-300:]}\n")
#                 fail_count += 1


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
    unique_id = f"A{params['approach_deg']}_T{params['tension']}_K{params['k_contact']:.0e}_VM{params['v_mode']}_{os.urandom(2).hex()}"
    
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

    # --- 1. PARAMETER DEFINITIONS ---
    approach_angles = [15, 30, 45, 60, 75]        # 5 steps
    tensions = [3, 5, 7, 10]                      # 4 steps
    cyl_radii = [0.01, 0.015, 0.02, 0.025]        # 4 steps
    k_contacts = [1.25e3, 5e3, 1.25e4]            # 3 steps
    base_radii = [0.005, 0.007, 0.01]             # 3 steps
    poisson_vals = [0.3, 0.4]                     # 2 steps
    joint_softness_vals = [0.01, 0.005]           # 2 steps

    # Vertebrae Configuration (Reduced to 6 each to keep total ~4300)
    uniform_sets = [
        (30, 62), (20, 70), (35, 65), (10, 40), (40, 75), (25, 55)
    ]
    manual_sets = [
        "20,30,60", "10,50,70", "30,35,70", "15,25,35", "50,60,70", "25,50,75"
    ]

    # --- 2. TASK GENERATION ---
    # Combinatorics: 12 (sets) * 5 (ang) * 4 (ten) * 4 (rad) * 3 (k) * 3 (br) * 2 (poi) * 2 (js) = 4,320 tasks
    for deg in approach_angles:
        for t in tensions:
            for cr in cyl_radii:
                for kc in k_contacts:
                    for br in base_radii:
                        for p_nu in poisson_vals:
                            for js in joint_softness_vals:
                                # Shared parameters
                                base_p = {
                                    "sol": "approach_angle", "approach_deg": deg, 
                                    "tension": t, "cyl_rad": cr, "k_contact": kc, 
                                    "base_rad": br, "poisson_nu": p_nu, "joint_softness": js
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
    
    # --- TEST MODE SWITCH ---
    tasks_to_run = tasks[:60]  # Toggle for testing
    # tasks_to_run = tasks         # Use for full 4320 samples
    
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

