import glob
import numpy as np

for file in glob.glob("squirrel_paw_results/*.npz"):
    d = np.load(file)
    if d["metric_is_stable"] and d["metric_energy_total"] < 600:
        print(f"FOUND OPTIMAL GRIP: {file}")
        print(f"Parameters: E={d['E']}, Tension={d['tension']}")