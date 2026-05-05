import os
import numpy as np
from glob import glob
from collections import Counter

dataset_dir = "runs/exp21"


def print_dataset_summary(dataset_dir):
    files = glob(os.path.join(dataset_dir, "**/*.npz"), recursive=True)

    print(f"Looking in: {os.path.abspath(dataset_dir)}")
    print(f"Found {len(files)} npz files")

    if len(files) == 0:
        print("No .npz files found. Check dataset_dir path.")
        return

    # -------------------------
    # Storage
    # -------------------------
    score_counts = {0.0: 0, 1/3: 0, 2/3: 0, 1.0: 0}
    success_all_3 = 0
    success_at_least_1 = 0

    contact_list = []
    score_list = []

    # -------------------------
    # Loop over dataset
    # -------------------------
    for f in files:
        with np.load(f, allow_pickle=True) as data:
            # ---- disturbance score ----
            score = float(np.asarray(data.get("disturbance_resistance_score", [0.0])).reshape(-1)[0])
            score_list.append(score)

            score_rounded = round(score * 3) / 3
            score_counts[score_rounded] = score_counts.get(score_rounded, 0) + 1

            if score_rounded >= 1.0:
                success_all_3 += 1
            if score_rounded > 0.0:
                success_at_least_1 += 1

            # ---- contacts ----
            num_contacts = float(np.asarray(data.get("num_contacts", [0.0])).reshape(-1)[0])
            contact_list.append(num_contacts)
            if num_contacts > 20:
                print(f"High contact count ({num_contacts}) in file: {f}")

    contact_arr = np.array(contact_list)
    score_arr = np.array(score_list)

    # -------------------------
    # Print disturbance summary
    # -------------------------
    print("\n=== DISTURBANCE RESISTANCE SUMMARY ===")
    print(f"0 / 3 resisted: {score_counts.get(0.0, 0)}")
    print(f"1 / 3 resisted: {score_counts.get(1/3, 0)}")
    print(f"2 / 3 resisted: {score_counts.get(2/3, 0)}")
    print(f"3 / 3 resisted: {score_counts.get(1.0, 0)}")

    print(f"\nAt least 1 direction resisted: {success_at_least_1}")
    print(f"All 3 directions resisted: {success_all_3}")

    # -------------------------
    # Print contact stats
    # -------------------------
    print("\n=== CONTACT COUNT SUMMARY ===")
    print(f"min: {contact_arr.min():.2f}")
    print(f"max: {contact_arr.max():.2f}")
    print(f"mean: {contact_arr.mean():.2f}")
    print(f"std: {contact_arr.std():.2f}")

    contact_counter = Counter(contact_arr.astype(int))
    print("\nContact distribution:")
    for k in sorted(contact_counter.keys()):
        print(f"{k}: {contact_counter[k]}")

    # -------------------------
    # Score stats
    # -------------------------
    print("\n=== DISTURBANCE SCORE STATS ===")
    print(f"min: {score_arr.min():.3f}")
    print(f"max: {score_arr.max():.3f}")
    print(f"mean: {score_arr.mean():.3f}")
    print(f"std: {score_arr.std():.3f}")

    # -------------------------
    # Dataset size
    # -------------------------
    print(f"\nTotal samples: {len(files)}")

    return contact_arr, score_arr


import matplotlib.pyplot as plt

# -------------------------
# Plot distributions
# -------------------------
def plot_distributions(contact_arr, score_arr, dataset_dir):
    os.makedirs(os.path.join(dataset_dir, "plots"), exist_ok=True)

    # ---- Contact histogram ----
    plt.figure()
    plt.hist(contact_arr, bins=20)
    plt.xlabel("Number of Contacts")
    plt.ylabel("Frequency")
    plt.title("Contact Count Distribution")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(dataset_dir, "plots/contact_distribution.png"))
    plt.close()

    # ---- Score histogram ----
    plt.figure()
    plt.hist(score_arr, bins=[0, 1/3, 2/3, 1.0, 1.01])  # discrete bins
    plt.xlabel("Disturbance Resistance Score")
    plt.ylabel("Frequency")
    plt.title("Disturbance Score Distribution")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(dataset_dir, "plots/score_distribution.png"))
    plt.close()

    # ---- Scatter (important insight) ----
    plt.figure()
    plt.scatter(contact_arr, score_arr, alpha=0.6)
    plt.xlabel("Number of Contacts")
    plt.ylabel("Disturbance Score")
    plt.title("Contacts vs Stability")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(dataset_dir, "plots/contact_vs_score.png"))
    plt.close()


if __name__ == "__main__":
    contact_arr, score_arr = print_dataset_summary(dataset_dir)    
    plot_distributions(contact_arr, score_arr, dataset_dir)