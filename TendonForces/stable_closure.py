import numpy as np
import pandas as pd

def evaluate_hanging_stability(csv_path, body_mass, finger_mass, mu=0.6):
    df = pd.read_csv(csv_path)
    final_frame = df[df['frame_idx'] == df['frame_idx'].max()]
    
    # Total load (Squirrel body + finger)
    g = 9.80665
    total_load = (body_mass + finger_mass) * g
    
    # In your CSV, the 'normal_force' and 'friction_force' are magnitudes.
    # To be precise, we look at the vertical components (Z-axis).
    # Assuming 'node_z' and cylinder geometry, the reaction force upward is:
    # F_up = Normal_Force_Z + Friction_Force_Z
    
    # Simplified check: Is the friction capacity enough to hold the load?
    total_normal = final_frame['normal_force'].sum()
    max_friction_support = total_normal * mu
    
    is_stable = max_friction_support > total_load
    return is_stable, max_friction_support / total_load
