import os
import numpy as np
from glob import glob

dataset_dir = "TendonForces/runs/exp15"  # change if needed

files = glob(os.path.join(dataset_dir, "**/*.npz"), recursive=True)

force_closure_0 = 0
force_closure_1 = 0

stability_0 = 0
stability_1 = 0

for f in files:
    with np.load(f, allow_pickle=True) as data:
        fc = float(data.get("metric_is_force_closure", 0.0))
        sm = float(data.get("stability_margin", 0.0))

        # ---- force closure ----
        if fc >= 0.5:
            force_closure_1 += 1
        else:
            force_closure_0 += 1

        # ---- stability ----
        if sm >= 0.5:
            stability_1 += 1
        else:
            stability_0 += 1

print("=== FORCE CLOSURE ===")
print(f"0: {force_closure_0}")
print(f"1: {force_closure_1}")

print("\n=== STABILITY ===")
print(f"0: {stability_0}")
print(f"1: {stability_1}")

print(f"\nTotal samples: {len(files)}")