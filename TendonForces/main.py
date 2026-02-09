import os
import sys
import datetime
import numpy as np
from collections import defaultdict

# PyElastica imports
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
from elastica.boundary_conditions import OneEndFixedBC, FixedConstraint
from elastica.external_forces import GravityForces
from elastica.dissipation import AnalyticalLinearDamper
from elastica.contact_forces import RodCylinderContact

import matplotlib
import matplotlib.pyplot as plt

if "--debug" in sys.argv:
    plt.switch_backend("TkAgg")
else:
    plt.switch_backend("Agg")

# Custom Module Imports
from parser import parser
from simulator import SquirrelFingerSimulator, setup_rod_and_cylinder
from forces import TendonForcesRamp, BodyWeightForcing, TendonForces
from callback import SquirrelCallback
from datasaver import save_contact_log, update_sweep_summary
from plot import generate_simulation_video

# Metrics Imports
from grasp_metrics import check_force_closure, compute_total_energy
from metrics import analyze_grasp_from_log, plot_contacts_2d_from_log, analyze_stable_closure_from_log
from scoring_metrics import compute_weighted_grasp_score, calculate_nist_scores

def main():
    args = parser()
    E = args.E
    nu = args.poisson_nu
    G = E / (2 * (1 + nu))
    density = 1200
    base_length = args.base_len
    base_radius = args.base_rad
    
    # Cylinder Params
    cyl_radius = args.cyl_rad
    cyl_length = 0.20
    cyl_start = np.array([0.025, -cyl_length / 2.0, -0.03])
    cyl_direction = np.array([0.0, 1.0, 0.0])
    cyl_normal = np.array([1.0, 0.0, 0.0])
    
    # Output Setup
    os.makedirs(args.output_dir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = args.suffix
    
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    start_pos = np.array([0.0, 0.0, 0.0])
    v_height_dir = np.array([0.0, 0.0, -1.0])

    if args.sol == "approach_angle":
        angle_rad = np.deg2rad(args.approach_deg)
        start_x = cyl_start[0] - 2 * cyl_radius
        start_z = -0.03 - (cyl_radius) * np.sin(angle_rad)
        start_pos = np.array([start_x, 0.0, start_z])
        direction = np.array([np.cos(angle_rad), 0.0, np.sin(angle_rad)])
        world_side = np.array([0.0, 1.0, 0.0]) 
        normal = np.cross(world_side, direction)
        normal /= np.linalg.norm(normal)
        v_height_dir = normal.copy()

    sim = SquirrelFingerSimulator()
    
    cyl_params = {
        'start': cyl_start, 'direction': cyl_direction, 
        'normal': cyl_normal, 'length': cyl_length
    }
    
    finger, cylinder, vertebra_nodes = setup_rod_and_cylinder(
        args, start_pos, direction, normal, cyl_params
    )
    
    sim.append(finger)
    sim.append(cylinder)

    sim.constrain(finger).using(OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,))
    sim.constrain(cylinder).using(FixedConstraint, constrained_position_idx=(0,), constrained_director_idx=(0,))

    if args.sol in ["nonuniform_tendon", "change_tendon_direction"]:
        cyl_center_fixed = np.array([0.025, 0.0, -0.03])
        sim.add_forcing_to(finger).using(
            TendonForcesRamp, 
            vertebra_height=args.v_height, 
            num_vertebrae=len(vertebra_nodes),
            first_vertebra_node=args.v_start, 
            final_vertebra_node=args.v_end,
            vertebra_mass=args.v_mass,
            tension=args.tension, 
            vertebra_height_orientation=v_height_dir,
            n_elements=args.n_elements, 
            ramp_up_time=1.0, 
            use_gradient=True,
            center_seek=True, 
            cyl_center=cyl_center_fixed, 
            vertebra_nodes_list=vertebra_nodes
        )
    else:
        sim.add_forcing_to(finger).using(
            TendonForces, 
            vertebra_height=args.v_height, 
            num_vertebrae=len(vertebra_nodes),
            first_vertebra_node=args.v_start, 
            final_vertebra_node=args.v_end,
            vertebra_mass=args.v_mass,
            tension=args.tension, 
            vertebra_height_orientation=v_height_dir,
            n_elements=args.n_elements, 
            vertebra_nodes_list=vertebra_nodes
        )

    gravity = 9.80665

    sim.add_forcing_to(finger).using(GravityForces, np.array([0.0, 0.0, -gravity]))
    
    body_weight_force = np.array([0.0, 0.0, -args.body_mass * gravity])
    sim.add_forcing_to(finger).using(
        BodyWeightForcing, force_vector=body_weight_force, node_indices=np.arange(0, 5)
    )

    # Damping and Contact
    wave_speed = np.sqrt(E / density)
    dt_critical = (base_length / args.n_elements) / wave_speed
    time_step = 0.1 * dt_critical
    
    sim.dampen(finger).using(AnalyticalLinearDamper, damping_constant=args.damping, time_step=time_step)
    sim.detect_contact_between(finger, cylinder).using(
        RodCylinderContact, k=args.k_contact, nu=args.nu_contact,
        velocity_damping_coefficient=args.vel_damp_contact, friction_coefficient=args.mu_contact
    )

    data = defaultdict(list)
    rendering_fps = 30.0
    step_skip = int(1.0 / (rendering_fps * time_step))
    
    sim.collect_diagnostics(finger).using(SquirrelCallback, step_skip=step_skip, callback_params=data)
    sim.finalize()
    
    final_time = 2.0
    total_steps = int(final_time / time_step)
    integrate(PositionVerlet(), sim, final_time, total_steps)

    # --- Contact Logging (Uses compute_contact_metrics_frame inside) ---
    csv_path = os.path.join(args.output_dir, f"contact_log_{run_id}_{suffix}.csv")
    cyl_display = {'center': cylinder.position_collection[:, 0], 'axis': cylinder.director_collection[2, :, 0]}
    save_contact_log(csv_path, data, cyl_display, args, (step_skip * time_step), cyl_length)

    # --- Grasp Metrics ---
    is_fc_geom, metrics = analyze_grasp_from_log(csv_path)
    
    # Force Closure Check (Specifically calling check_force_closure)
    final_pos = data["position"][-1]
    final_forces = data["external_forces"][-1]
    normal_forces = np.linalg.norm(final_forces, axis=0)
    contact_idx = np.where(normal_forces > 0.1)[0]
    
    if len(contact_idx) > 0:
        c_verts = final_pos[:, contact_idx].T
        c_norms = (c_verts - cyl_display['center'])
        c_norms /= np.linalg.norm(c_norms, axis=1)[:, None]
        body_wrench = np.concatenate([body_weight_force, [0,0,0]])
        is_force_closure = check_force_closure(c_verts, c_norms, args.mu_contact, external_wrench=body_wrench)
    else:
        is_force_closure = False

    energy_score = compute_total_energy(finger, normal_forces[contact_idx], args.k_contact)

    # Stability and NIST
    is_stable, stab_metrics = analyze_stable_closure_from_log(
        csv_path, rod_mass=np.sum(finger.mass), v_mass_total=args.num_v * args.v_mass, args=args
    )
    
    cycle_time, strength, slip_res = calculate_nist_scores(
        data, cyl_radius, base_radius, args.k_contact, args.mu_contact, args.body_mass, gravity
    )

    grasp_data = {
        "angular_span": metrics['angular_span'], 
        "vertical_support": stab_metrics['total_support_n'],
        "body_weight": args.body_mass * gravity, 
        "total_energy": energy_score,
        "contact_count": metrics['num_contacts'], 
        "max_normal_force": np.max(normal_forces),
        "total_normal_force": np.sum(normal_forces),
        "cycle_time": cycle_time, 
        "strength": strength, 
        "slip_resistance": slip_res
    }
    final_score, breakdown = compute_weighted_grasp_score(grasp_data)

    data_to_save = {key: np.array(value) for key, value in data.items()}
    data_to_save.update({"final_grasp_score": np.array([final_score]), "geometric_fc": np.array([is_force_closure])})
    np.savez_compressed(os.path.join(args.output_dir, f"master_{run_id}_{suffix}.npz"), **data_to_save)

    update_sweep_summary(
        os.path.join(args.output_dir, "sweep_summary.csv"),
        [run_id, args.sol, E, args.tension, cyl_radius, args.approach_deg, is_force_closure, 
         metrics['angular_span'], metrics['num_contacts'], energy_score, final_score, str(breakdown)]
    )

    plot_contacts_2d_from_log(csv_path, output_path=os.path.join(args.output_dir, f"plot2d_{run_id}.png"), show_plot=False)
    
    generate_simulation_video(
        os.path.join(args.output_dir, f"sim_{run_id}_{suffix}.mp4"),
        data, cylinder, args, vertebra_nodes, rendering_fps, final_time, body_weight_force
    )

    print(f"\n[FINISH] Run ID: {run_id} | Score: {final_score:.4f} | Force Closure: {is_force_closure}")

if __name__ == "__main__":
    main()