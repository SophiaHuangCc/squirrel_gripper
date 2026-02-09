import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data = np.load("squirrel_paw_results/master_log_20260208_201225_default.npz")
data = np.load("squirrel_paw_results/master_log_20260206_000851_default.npz")

print("--- KEYS FOUND IN NPZ ---")
for key in data.files:
    print(f"Key: {key:20}")
print("number of keys in data.files:", len(data.files))

print("\n--- METADATA CHECK ---")
if "final_grasp_score" in data:
    print(f"Stored Grasp Score: {data['final_grasp_score'][0]:.4f}")
if "geometric_fc" in data:
    print(f"Force Closure Success: {data['geometric_fc'][0]}")
