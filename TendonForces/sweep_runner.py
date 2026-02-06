import subprocess
import numpy as np
import os

# --- 1. Define Systematic Grids ---
# We use np.linspace to get even intervals across your specified ranges
e_values = np.linspace(1e7, 5e7, 4)           # 4 stiffness levels
cyl_radii = [0.01, 0.015]                     # 2 sizes
approach_angles = np.linspace(0, 90, 8)       # 8 angles (0, 12.8, 25.7, ..., 90)

# Define solution-specific tensions as requested
tensions_map = {
    "approach_angle": [2.0, 3.0, 5.0],        # High tension required for angle
    "nonuniform_tendon": [0.1, 0.2, 0.4]      # Low tension for efficient gradient
}

os.makedirs("squirrel_paw_results", exist_ok=True)

# Calculate total for progress bar
total_runs = len(tensions_map) * len(e_values) * 3 * len(cyl_radii) * len(approach_angles)
count = 0

print(f"Starting Systematic Sweep: {total_runs} total combinations...")

for sol, tensions in tensions_map.items():
    for e in e_values:
        for t in tensions:
            for rad in cyl_radii:
                for ang in approach_angles:
                    count += 1
                    
                    # Build command with --debug OFF for speed
                    cmd = [
                        "python", "finger.py",
                        "--sol", sol,
                        "--E", f"{e:.2e}",
                        "--tension", str(t),
                        "--cyl_rad", str(rad),
                        "--approach_deg", f"{ang:.2f}"
                    ]
                    
                    # Log progress to terminal
                    print(f"[{count}/{total_runs}] | SOL: {sol:18} | E: {e:.1e} | T: {t} | R: {rad} | Ang: {ang:5.1f}")
                    
                    try:
                        # Run simulation
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as err:
                        print(f"!!! Error on run {count} (Sol: {sol}, E: {e:.1e}): {err}")
                        # Continue to next run instead of crashing the whole sweep
                        continue

print("-" * 30)
print(f"Sweep Complete! Results are in squirrel_paw_results/sweep_summary.csv")