import numpy as np

###################################################
# NIST-INSPIRED METRIC CALCULATIONS
###################################################

def calculate_nist_scores(data, cyl_radius, base_radius, k_contact, mu_contact, body_mass, gravity):
    pos_history = data["position"]
    vel_history = data["velocity"]
    time_history = data["time"]
    
    # Thresholds for "Stable Contact"
    contact_threshold = cyl_radius + base_radius
    force_plateau_threshold = 0.05  # Force variation less than 5% to be "stable"
    
    # Initialization
    max_normal_force_all_frames = 0.0
    first_contact_time = None
    stable_contact_time = None
    prev_total_force = 0.0
    
    # 1. & 2. Grasp Cycle Time and Finger Strength
    for i in range(len(pos_history)):
        # Calculate radial distances for this frame
        dx = pos_history[i][0, :] - 0.025 # cyl_x
        dz = pos_history[i][2, :] - (-0.03) # cyl_z
        radial_dist = np.sqrt(dx**2 + dz**2)
        overlaps = contact_threshold - radial_dist
        
        # Current normal forces
        normal_forces = k_contact * np.where(overlaps > 0, overlaps, 0.0)
        current_total_force = np.sum(normal_forces)
        
        # Track Finger Strength (Peak Force)
        max_normal_force_all_frames = max(max_normal_force_all_frames, np.max(normal_forces))
        
        # Track First Contact
        if first_contact_time is None and np.any(overlaps > 0):
            first_contact_time = time_history[i]
            
        # Track Stabilization (Cycle Time)
        # Defined as when force is significant and stopped changing rapidly
        if first_contact_time is not None and stable_contact_time is None:
            force_diff = abs(current_total_force - prev_total_force)
            if current_total_force > 1.0 and force_diff < force_plateau_threshold:
                stable_contact_time = time_history[i]
        
        prev_total_force = current_total_force

    # Score A: Grasp Cycle Time
    # If it never stabilizes, we penalize with max time
    cycle_time = (stable_contact_time - first_contact_time) if stable_contact_time else time_history[-1]
    
    # Score B: Finger Strength
    # NIST looks at the max force the finger can exert
    finger_strength_score = max_normal_force_all_frames
    
    # 3. Slip Resistance (Static Security)
    # Based on the final frame's ability to resist the 0.5kg load
    final_pos = pos_history[-1]
    final_dx = final_pos[0, :] - 0.025
    final_dz = final_pos[2, :] - (-0.03)
    final_overlaps = contact_threshold - np.sqrt(final_dx**2 + final_dz**2)
    final_normal_forces = k_contact * np.where(final_overlaps > 0, final_overlaps, 0.0)
    
    total_normal_force = np.sum(final_normal_forces)
    friction_capacity = mu_contact * total_normal_force
    required_force = body_mass * gravity
    
    # slip_resistance = available friction / required resistance
    slip_resistance_score = friction_capacity / (required_force + 1e-6)

    return cycle_time, finger_strength_score, slip_resistance_score

def compute_weighted_grasp_score(data_dict, weights=None):
    """
    Pareto-Optimized Grasp Score.
    Weights are applied to the 'Performance' group, while Efficiency 
    acts as a multiplicative filter to penalize brute-force hacks.
    """
    if weights is None:
        # These weights balance the performance aspect
        weights = {
            "wrap": 0.3, 
            "stability": 0.2, 
            "distribution": 0.1,
            "cycle_time": 0.1,
            "strength": 0.1,
            "slip_resistance": 0.2
        }

    # 1. Geometry (Wrap)
    q_wrap = np.clip(data_dict['angular_span'] / (1.5 * np.pi), 0, 1)
    
    # 2. Stability (Safety Margin)
    margin_against_gravity = data_dict['vertical_support'] / (data_dict['body_weight'] + 1e-6)
    q_stab = np.clip((margin_against_gravity - 1.0) / 1.0, 0, 1) 

    # 3. Distribution (Pressure Evenness)
    # Ratio of Avg Force to Max Force. 
    # If one node does all the work, q_dist -> 0. If even, q_dist -> 1.
    avg_force = data_dict['total_normal_force'] / (data_dict['contact_count'] + 1e-6)
    q_dist = np.clip(avg_force / (data_dict['max_normal_force'] + 1e-6), 0, 1)

    # 4. Efficiency Gatekeeper (The Brute-Force Filter)
    # Penalty for high internal strain energy
    energy_threshold = 1.0 
    q_eff = np.exp(-max(0, data_dict['total_energy'] - energy_threshold) / energy_threshold)

    # 5. Cycle Time (Speed of Stabilization)
    q_time = np.clip(1.0 - (data_dict['cycle_time'] / 2.0), 0, 1)

    # 6. Strength (Peak Force Capability)
    target_strength = 20.0 
    q_strength = np.clip(data_dict['strength'] / target_strength, 0, 1)

    # 7. Slip Resistance (Static Security)
    margin_against_slip = data_dict['slip_resistance']
    q_slip = np.clip((margin_against_slip - 1.0) / 1.0, 0, 1)

    # --- Final Pareto Calculation ---
    
    # Performance is the weighted sum of the 'Grip Quality'
    performance_score = (
        weights['wrap'] * q_wrap + 
        weights['stability'] * q_stab + 
        weights['distribution'] * q_dist +
        weights['cycle_time'] * q_time +
        weights['strength'] * q_strength +
        weights['slip_resistance'] * q_slip
    )
    
    # Total Score is Performance MULTIPLIED by Efficiency
    # This ensures that Trash Efficiency = Trash Score
    total_score = performance_score * q_eff 

    # Hard Failure (Falling)
    if margin_against_gravity < 1.0 or margin_against_slip < 1.0:  # 1.0 means it can just barely support the weight
        total_score = 0.0

    return total_score, {
        "total": total_score, 
        "q_wrap": q_wrap,      # Explicitly returning these now
        "q_stab": q_stab,
        "q_dist": q_dist,      # <--- Added back to dictionary
        "q_eff": q_eff, 
        "margin": margin_against_gravity,
        "cycle_time": q_time,
        "strength": q_strength,
        "slip_resistance": q_slip,
        "margin_against_slip": margin_against_slip
    }