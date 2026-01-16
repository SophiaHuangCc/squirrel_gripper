import glob
import numpy as np

count = 0
total = 0
good = 0
for file in glob.glob("squirrel_paw_results/*.npz"):
    d = np.load(file)
    if "metric_is_stable" not in d or "geometric_success" not in d:
        continue
    total += 1
    if d["metric_is_stable"]:
        count += 1
    if d["geometric_success"]:
        good += 1
        # print(f"FOUND OPTIMAL GRIP: {file}")
        # print(f"Parameters: E={d['E']}, Tension={d['tension']}")
print(f"Total Stable Grasps Found: {count} out of {total} simulations.")
print(f"Total Geometrically Successful Grasps: {good} out of {total} simulations.")