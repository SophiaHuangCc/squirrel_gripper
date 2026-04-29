import os
import numpy as np
from glob import glob

dataset_dir = "runs/exp19" 

def print_disturbance_summary(dataset_dir):
    files = glob(os.path.join(dataset_dir, "**/*.npz"), recursive=True)

    score_counts = {0.0: 0, 1/3: 0, 2/3: 0, 1.0: 0}
    success_all_3 = 0
    success_at_least_1 = 0

    for f in files:
        with np.load(f, allow_pickle=True) as data:
            score = float(data.get("disturbance_resistance_score", 0.0))

            # round to nearest third
            score_rounded = round(score * 3) / 3
            score_counts[score_rounded] = score_counts.get(score_rounded, 0) + 1

            if score_rounded >= 1.0:
                success_all_3 += 1
            if score_rounded > 0.0:
                success_at_least_1 += 1

    print("\n=== DISTURBANCE RESISTANCE SUMMARY ===")
    print(f"0 / 3 resisted: {score_counts.get(0.0, 0)}")
    print(f"1 / 3 resisted: {score_counts.get(1/3, 0)}")
    print(f"2 / 3 resisted: {score_counts.get(2/3, 0)}")
    print(f"3 / 3 resisted: {score_counts.get(1.0, 0)}")
    print(f"\nAt least 1 direction resisted: {success_at_least_1}")
    print(f"All 3 directions resisted: {success_all_3}")
    print(f"Total samples: {len(files)}")

if __name__ == "__main__":
    print_disturbance_summary(dataset_dir)