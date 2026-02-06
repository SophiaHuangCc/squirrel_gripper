import numpy as np

def compute_weighted_grasp_score(data_dict, weights=None):
    """
    Pareto-Optimized Grasp Score.
    Weights are applied to the 'Performance' group, while Efficiency 
    acts as a multiplicative filter to penalize brute-force hacks.
    """
    if weights is None:
        # These weights balance the performance aspect
        weights = {
            "wrap": 0.4, 
            "stability": 0.4, 
            "distribution": 0.2 
        }

    # 1. Geometry (Wrap)
    q_wrap = np.clip(data_dict['angular_span'] / (1.5 * np.pi), 0, 1)
    
    # 2. Stability (Safety Margin)
    margin = data_dict['vertical_support'] / (data_dict['body_weight'] + 1e-6)
    q_stab = np.clip((margin - 1.0) / 1.0, 0, 1) 

    # 3. Distribution (Pressure Evenness)
    # Ratio of Avg Force to Max Force. 
    # If one node does all the work, q_dist -> 0. If even, q_dist -> 1.
    avg_force = data_dict['total_normal_force'] / (data_dict['contact_count'] + 1e-6)
    q_dist = np.clip(avg_force / (data_dict['max_normal_force'] + 1e-6), 0, 1)

    # 4. Efficiency Gatekeeper (The Brute-Force Filter)
    # Penalty for high internal strain energy
    energy_threshold = 2.0 
    q_eff = np.exp(-max(0, data_dict['total_energy'] - energy_threshold) / energy_threshold)

    # --- Final Pareto Calculation ---
    
    # Performance is the weighted sum of the 'Grip Quality'
    performance_score = (
        weights['wrap'] * q_wrap + 
        weights['stability'] * q_stab + 
        weights['distribution'] * q_dist
    )
    
    # Total Score is Performance MULTIPLIED by Efficiency
    # This ensures that Trash Efficiency = Trash Score
    total_score = performance_score * q_eff 

    # Hard Failure (Falling)
    if margin < 1.0: 
        total_score = 0.0

    return total_score, {
        "total": total_score, 
        "q_wrap": q_wrap,      # Explicitly returning these now
        "q_stab": q_stab,
        "q_dist": q_dist,      # <--- Added back to dictionary
        "q_eff": q_eff, 
        "margin": margin
    }