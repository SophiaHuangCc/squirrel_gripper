import os
import numpy as np
import csv
import fcntl
from utils import compute_contact_metrics_frame

def save_contact_log(csv_path, data, cyl_params, args, dt_saved, cyl_length):
    pos_data = data["position"]
    vel_data = data["velocity"]
    if len(pos_data) < 2:
        raise RuntimeError("No saved frames")
    cyl_center = cyl_params['center']
    cyl_axis = cyl_params['axis']
    cyl_radius = args.cyl_rad
    base_radius = args.base_rad
    cyl_length = cyl_length
    k_contact = args.k_contact
    mu_contact = args.mu_contact
    
    first_contact_frame = None
    max_normal_force_overall = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "time", "node_idx",
            "node_x", "node_y", "node_z",
            "radial_dist", "overlap",
            "normal_force", "normal_velocity", "tangential_speed", "friction_force",
            "cyl_center_x", "cyl_center_y", "cyl_center_z",
            "cyl_axis_x", "cyl_axis_y", "cyl_axis_z",
            "cyl_radius"
        ])

        for frame_idx, (P, V) in enumerate(zip(pos_data, vel_data)):
            t = frame_idx * dt_saved
            (idx, nF, nV, tS, fF, radial_dist_all, overlap_all) = compute_contact_metrics_frame(
                P, V, cyl_center, cyl_axis, cyl_radius, k_contact, mu_contact, base_radius, cyl_length
            )

            if len(idx) > 0:
                frame_max = float(np.max(nF))
                max_normal_force_overall = max(max_normal_force_overall, frame_max)
                if first_contact_frame is None:
                    first_contact_frame = frame_idx

            for j, node_idx in enumerate(idx):
                node_pos = P[:, node_idx]
                writer.writerow([
                    frame_idx, t, int(node_idx),
                    float(node_pos[0]), float(node_pos[1]), float(node_pos[2]),
                    float(radial_dist_all[node_idx]),
                    float(overlap_all[node_idx]),
                    float(nF[j]),
                    float(nV[j]),
                    float(tS[j]),
                    float(fF[j]),
                    float(cyl_center[0]), float(cyl_center[1]), float(cyl_center[2]),
                    float(cyl_axis[0]), float(cyl_axis[1]), float(cyl_axis[2]),
                    float(cyl_radius),
                ])
    return first_contact_frame, max_normal_force_overall

def update_sweep_summary(summary_file, row_data):
    file_exists = os.path.isfile(summary_file)
    with open(summary_file, mode='a', newline='') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["run_id", "sol", "E", "tension", "cyl_rad", "approach_deg", 
                                "geometric_fc", "angular_span", "num_contacts", "total_energy", 
                                "final_score", "breakdown"])
            writer.writerow(row_data)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)