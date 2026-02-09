""""
This file uses PyElastica to model a squirrel-inspired, tendon-driven soft robotic 
finger and visualizes its interaction with a rigid cylindrical branch.

The modeling approach is based on Cosserat rod theory, utilizing high-fidelity 
simulations to solve the "distal tip curl" problem common in tendon-driven 
continuum systems.

Finger Configuration (Squirrel-Inspired):
    Base Length: 100 mm
    Base Radius: 3 mm
    Actuation: Tendon-driven via discrete vertebrae (tendon routing points).
    Material: TPU-equivalent (Young's Modulus E ~ 20 MPa, Poisson's Ratio 0.5).

Environment (Branch):
    Geometry: Rigid Cylinder.
    Radius: Variable (10-15 mm).
    Contact: Rod-Cylinder penalty-based contact with friction (RodCylinderContact).

Visualization:
    Frame coordinates are as follows: X - to the right, Y - into the screen, Z - upwards.

Key Solutions for Distal Tip Curling:
    1. Dynamic Approach Angle: Modifying the landing orientation (0° to 90°).
    2. Non-uniform Tendon Force: Linear force gradient increasing toward the tip. #TODO: not sure if realistic
    3. Center-Seeking Direction: Dynamic tendon force vectors steered toward the 
       cylinder axis to maximize angular span and form/force closure.

Modeling Principles & Assumptions:
    - Backbone modeled as a Cosserat rod with 80 elements for high-fidelity bending.
    - Soft joints: Localized reduction of bending stiffness (EI) at vertebrae 
      locations to mimic biological joint flexibility.
    - Tendon Force: Applied as external forces and torques at discrete nodes, 
      mimicking biological flexor tendons.
    - Systematic Sweep: Automated logging of Form/Force Closure (Geometric/Friction) 
      and Total Energy Metrics for parameter optimization.
"""

# Ideas improve contact performance:
# solution 1: approach angle change --works
# solution 2: increase tendon force towards tip --realistic at manufacturing?
# solution 3: change tendon force direction to curl --realistic at manufacturing?
# TODO: different vertebra positions

###################################################
# IMPORTS
###################################################
import os
import numpy as np
from collections import defaultdict
import sys
import argparse
import time
import datetime
import matplotlib
from mpl_toolkits.mplot3d import Axes3D 
import csv
import fcntl # Standard on Linux/Mac/Lab Servers
# parallel -j 8 python finger.py --sol {1} --tension {2} ::: nonuniform_tendon approach_angle ::: 0.2 0.4 0.6

# Matplotlib backend setup
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
if "--debug" in sys.argv:
    plt.switch_backend("TkAgg")
    print("[DEBUG] Matplotlib interactive backend enabled.")
else:
    plt.switch_backend("Agg")

# MoviePy for video rendering
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

# PyElastica imports
from elastica.modules import BaseSystemCollection, Connections, Constraints, Forcing, CallBacks, Damping, Contact
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC, FixedConstraint
from elastica.external_forces import GravityForces, NoForces
from elastica.dissipation import AnalyticalLinearDamper
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
from elastica.rigidbody.cylinder import Cylinder
from elastica.contact_forces import RodCylinderContact
from TendonForces import TendonForces

# Custom metrics functions
from grasp_metrics import check_force_closure, compute_total_energy
from metrics import analyze_grasp_from_log, plot_contacts_2d_from_log, check_stable_closure_hanging, analyze_stable_closure_from_log
from scoring_metrics import compute_weighted_grasp_score, calculate_nist_scores


###################################################
# SIMULATOR CLASS
###################################################
class SquirrelFingerSimulator(
    BaseSystemCollection,
    Connections,
    Constraints,
    Forcing,
    CallBacks,
    Damping,
    Contact,
):
    pass


###################################################
# UTILITY FUNCTIONS
###################################################

# Function to draw a cylinder branch in 3D matplotlib
def draw_cylinder(ax, center, axis_dir, radius, length,
                  color="gray", alpha=0.3, resolution=40):
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    if np.allclose(axis_dir, [0, 0, 1]):
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.cross(axis_dir, [0, 0, 1])
        v /= np.linalg.norm(v)

    w = np.cross(axis_dir, v)

    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(-length / 2, length / 2, 20)
    theta, z = np.meshgrid(theta, z)

    X = center[0] + axis_dir[0] * z + radius * (v[0] * np.cos(theta) + w[0] * np.sin(theta))
    Y = center[1] + axis_dir[1] * z + radius * (v[1] * np.cos(theta) + w[1] * np.sin(theta))
    Z = center[2] + axis_dir[2] * z + radius * (v[2] * np.cos(theta) + w[2] * np.sin(theta))

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=True)

# Function to compute contact metrics for a single frame
def compute_contact_metrics_frame(
    rod_pos, rod_vel, cyl_center, cyl_axis, cyl_radius, k, mu, base_radius, cyl_length,
):
    # 1. Direct XZ Plane Distance
    dx = rod_pos[0, :] - cyl_center[0]
    dz = rod_pos[2, :] - cyl_center[2]
    radial_dist = np.sqrt(dx**2 + dz**2)

    # 2. Overlap (Cylinder 15mm + Rod 5mm = 20mm)
    contact_threshold = cyl_radius + base_radius # 0.020
    overlaps = contact_threshold - radial_dist

    # 3. Contact Mask
    contact_mask = overlaps > 0.0
    
    # 4. Directions (Normal is in the XZ plane)
    normal_vec = np.zeros((rod_pos.shape[1], 3))
    normal_vec[:, 0] = dx
    normal_vec[:, 2] = dz
    
    norms = np.linalg.norm(normal_vec, axis=1)
    mask = norms > 1e-12
    normal_vec[mask] /= norms[mask, None]

    # 5. Physics
    normal_force_mag = k * np.where(contact_mask, overlaps, 0.0)
    
    vel = rod_vel.T
    normal_vel = np.sum(vel * normal_vec, axis=1)
    vel_t = vel - normal_vel[:, None] * normal_vec
    tangential_speed = np.linalg.norm(vel_t, axis=1)
    
    friction_force_mag = mu * normal_force_mag

    idx = np.where(contact_mask)[0]
    return (idx, 
            normal_force_mag[idx], 
            normal_vel[idx], 
            tangential_speed[idx], 
            friction_force_mag[idx], 
            radial_dist, 
            overlaps)


###################################################
# CUSTOM FORCING CLASS WITH RAMP-UP AND DIRECTIONAL CONTROL
###################################################
class TendonForcesRamp(TendonForces):
    def __init__(self, *args, ramp_up_time=0.2, use_gradient=False, center_seek=False, cyl_center=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ramp_up_time = float(ramp_up_time)
        self.use_gradient = bool(use_gradient)
        self.center_seek = bool(center_seek)
        self.cyl_center = np.array(cyl_center) if cyl_center is not None else None
        
        self.first_vertebra_node = kwargs.get("first_vertebra_node")
        self.final_vertebra_node = kwargs.get("final_vertebra_node")
        self._tension_nominal = float(kwargs.get("tension", 0.0))

    def _update_geometry_and_get_factor(self, system, time, i, node_idx):
        """Helper to ensure vectors are 2D and compute the combined scaling factor."""
        if self.vertebra_height_vector.ndim == 1:
            self.vertebra_height_vector = np.tile(
                self.vertebra_height_vector.reshape(3, 1), (1, len(self.vertebra_nodes))
            )

        if self.center_seek and self.cyl_center is not None:
            to_cyl = self.cyl_center - system.position_collection[:, node_idx]
            dist = np.linalg.norm(to_cyl)
            if dist > 1e-6:
                self.vertebra_height_vector[:, i] = to_cyl / dist

        s = 1.0 if self.ramp_up_time <= 0 else min(1.0, max(0.0, float(time) / self.ramp_up_time))
        time_factor = 0.5 * (1.0 - np.cos(np.pi * s))
        
        spatial_factor = 1.0
        if self.use_gradient:
            n_start = self.first_vertebra_node
            n_end = self.final_vertebra_node
            alpha = (node_idx - n_start) / (max(1, n_end - n_start))
            spatial_factor = 0.5 + alpha 
            
        return time_factor * spatial_factor

    def apply_forces(self, system, time=0.0):
        for i, node_idx in enumerate(self.vertebra_nodes):
            factor = self._update_geometry_and_get_factor(system, time, i, node_idx)
            current_tension = self._tension_nominal * factor
            force = current_tension * self.vertebra_height_vector[:, i]
            system.external_forces[:, node_idx] += force

    def apply_torques(self, system, time=0.0):
        for i, node_idx in enumerate(self.vertebra_nodes):
            factor = self._update_geometry_and_get_factor(system, time, i, node_idx)
            current_tension = self._tension_nominal * factor
            
            tangent = system.tangents[:, node_idx]
            torque = current_tension * np.cross(self.vertebra_height_vector[:, i], tangent)
            system.external_torques[:, node_idx] += torque

class BodyWeightForcing(NoForces):
    def __init__(self, force_vector, node_indices):
        self.force_vector = force_vector
        self.node_indices = node_indices
        self.total_force_mag = np.linalg.norm(force_vector)

    def apply_forces(self, system, time=0.0):
        # Distribute the total body weight across the first few nodes
        force_per_node = self.force_vector / len(self.node_indices)
        for idx in self.node_indices:
            system.external_forces[:, idx] += force_per_node

###################################################
# MAIN SIMULATION FUNCTION
###################################################
def main():
    # Argument parsing inputs
    parser = argparse.ArgumentParser(description="Run Squirrel Finger Simulation.")
    parser.add_argument(
        "--sol", type=str, default="approach_angle",
        choices=["standard",
                 "approach_angle", "nonuniform_tendon", "change_tendon_direction"],
        help="Select the solution for improved curl")
    parser.add_argument(
        "--approach_deg", type=float, default=45.0,
        help="Angle of approach in degrees (0 = horizontal, 90 = vertical)")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode with plot of total force magnitude and direction.")
    # material and simulation params
    parser.add_argument("--E", type=float, default=2e7, help="Youngs Modulus (Pa)")
    parser.add_argument("--tension", type=float, default=0.4, help="Tendon tension (N)")
    parser.add_argument("--damping", type=float, default=0.8, help="Internal damping constant")
    parser.add_argument("--n_elements", type=int, default=80, help="Number of rod elements")
    # cylinder and contact params
    parser.add_argument("--cyl_rad", type=float, default=0.015, help="Cylinder radius (m)")
    parser.add_argument("--k_contact", type=float, default=1.25e3, help="Contact stiffness")
    # finger arguments
    parser.add_argument("--base_len", type=float, default=0.10, help="Finger length in meters")
    parser.add_argument("--base_rad", type=float, default=0.005, help="Finger radius in meters")
    parser.add_argument("--nu_contact", type=float, default=5.0, help="Contact damping")
    parser.add_argument("--mu_contact", type=float, default=0.6, help="Friction coefficient")
    parser.add_argument("--vel_damp_contact", type=int, default=10, help="Numerical contact stability")
    parser.add_argument("--poisson_nu", type=float, default=0.4, help="Poisson's ratio")
    # vertebrae arguments
    parser.add_argument("--v_mass", type=float, default=0.002, help="Mass of each vertebra")
    parser.add_argument("--num_v", type=int, default=3, help="Number of vertebrae")
    parser.add_argument("--v_start", type=int, default=30, help="Node index of first vertebra")
    parser.add_argument("--v_end", type=int, default=62, help="Node index of final vertebra")
    parser.add_argument("--v_height", type=float, default=0.005, help="Tendon distance from center")
    parser.add_argument("--joint_softness", type=float, default=0.01, help="Multiplier for bending stiffness at vertebrae (0.01 = 1% of original stiffness)")
    # vertebra selector: 'uniform' or 'manual'
    parser.add_argument("--v_mode", type=str, default="uniform", 
                        choices=["uniform", "manual"], 
                        help="How to place vertebrae: 'uniform' uses linspace, 'manual' uses v_list")
    # Manual list input (passed as a string of comma-separated integers)
    parser.add_argument("--v_list", type=str, default="30,46,62", 
                        help="Comma-separated node indices for manual vertebrae placement")
    # squirrel body mass for stability calculation
    parser.add_argument("--body_mass", type=float, default=0.5, help="Mass of the squirrel body in kg")
    parser.add_argument("--suffix", type=str, default="default", 
                        help="Suffix for output filenames to prevent overwriting")
    parser.add_argument("--output_dir", type=str, default="squirrel_paw_results", 
                        help="Directory to save output files")

    args = parser.parse_args()

    # Parameters from args (key variables to tune)
    E = args.E
    nu = args.poisson_nu  # Poisson's ratio (0.3~0.4 for TPU)
    G = E / (2 * (1 + nu))  # Shear modulus in Pa
    tension = args.tension
    n_elements = args.n_elements
    damping_constant = args.damping
    k_contact = args.k_contact
    cyl_radius = args.cyl_rad
    suffix = args.suffix
    # Geometry for finger (optimize later?)
    base_length = args.base_len
    base_radius = args.base_rad
    density = 1200
    mass_second_moment_of_inertia = 0.25 * np.pi * base_radius**4
    
    # Cylinder (branch) parameters
    cyl_length = 0.20
    cyl_density = 1200.0
    cyl_x = 0.025
    cyl_z = -0.03
    cyl_start = np.array([cyl_x, -cyl_length / 2.0, cyl_z])
    cyl_direction = np.array([0.0, 1.0, 0.0])
    cyl_normal = np.array([1.0, 0.0, 0.0])
    nu_contact = args.nu_contact # 5.0~20.0
    mu_contact = args.mu_contact # 0.6~1.0
    vel_damp_contact = args.vel_damp_contact # 10~50

    # Vertebrae parameters
    vertebra_mass = args.v_mass
    num_vertebrae = args.num_v
    first_vertebra_node = args.v_start
    final_vertebra_node = args.v_end
    vertebra_height = args.v_height
    if args.v_mode == "uniform":
        vertebra_nodes = np.linspace(first_vertebra_node, final_vertebra_node, num_vertebrae, dtype=int)
    elif args.v_mode == "manual":
        vertebra_nodes = np.array([int(x) for x in args.v_list.split(",")], dtype=int)
        num_vertebrae = len(vertebra_nodes)
    print(f"[VISUALIZATION] Drawing red disks at nodes: {vertebra_nodes}")

    # Mass of squirrel body
    body_mass = args.body_mass
    gravity = 9.80665
    body_weight_force = np.array([0.0, 0.0, -body_mass * gravity])
    
    # Time stepping parameters
    wave_speed = np.sqrt(E / density)
    dx = base_length / n_elements
    dt_critical = dx / wave_speed
    final_time = 2.0
    time_step = 0.1 * dt_critical
    rendering_fps = 30.0
    total_steps = int(final_time / time_step)
    step_skip = int(1.0 / (rendering_fps * time_step))

    # Output setup
    base_outdir = args.output_dir
    os.makedirs(base_outdir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(base_outdir, f"output_{run_id}_{suffix}.mp4")
    print(f"\n==== RUNNING: {args.sol} ====")
    print(f"E={E:.2e}  G={G:.2e}  tension={tension:.2f}  cyl_radius={cyl_radius:.3f}")
    print(f"contact: k={k_contact} nu={nu_contact} mu={mu_contact} vel_damp={vel_damp_contact}")
    print(f"damping_constant={damping_constant}")

    # Initialize simulator
    sim = SquirrelFingerSimulator()

    # Standard/default finger setup
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    start_pos = np.array([0.0, 0.0, 0.0])
    v_height_dir = np.array([0.0, 0.0, -1.0])
    clearance = 0.02

    # 2 alternative solutions to improve curling around cylinder
    if args.sol == "approach_angle":
        angle_rad = np.deg2rad(args.approach_deg)
        print(f">>> MODE: Dynamic Approach at {args.approach_deg} degrees")
        start_x = cyl_start[0] - 2 * cyl_radius
        start_z = cyl_z - (cyl_radius) * np.sin(angle_rad)
        print(f"[APPROACH ANGLE] Calculated start position: x={start_x:.3f}, z={start_z:.3f}")
        
        start_pos = np.array([start_x, 0.0, start_z])
        
        # Define direction: Points 'down and forward' toward the cylinder
        dir_x = np.cos(angle_rad)
        dir_z = np.sin(angle_rad)
        direction = np.array([dir_x, 0.0, dir_z])
        
        # Ensure normal is perpendicular for tendon routing
        world_side = np.array([0.0, 1.0, 0.0]) 
        normal = np.cross(world_side, direction)
        normal /= np.linalg.norm(normal)
        v_height_dir = normal.copy()
    elif args.sol == "nonuniform_tendon" or args.sol == "change_tendon_direction":
        print(">>> MODE: Combined Gradient Magnitude + Center-Seeking Direction")
        # Setup the cylinder center reference
        cyl_center = cyl_start + (cyl_direction * (cyl_length / 2.0))
        cyl_center_fixed = np.array([cyl_center[0], 0.0, cyl_center[2]])
    else:
        print(">>> MODE: Standard Horizontal")

    # Create finger rod
    finger = CosseratRod.straight_rod(
        n_elements=n_elements,
        start=start_pos,
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=E,
        shear_modulus=G,
        mass_second_moment_of_inertia=mass_second_moment_of_inertia,
    )
    sim.append(finger)

    # Apply soft joints by modifying bend_matrix at vertebrae locations
    joint_mult = args.joint_softness   

    # Apply soft joints at vertebrae locations
    print(f"[SOFT JOINTS] using mode '{args.v_mode}' at nodes: {vertebra_nodes}")        
    for j in vertebra_nodes:
        idx = int(np.clip(j, 0, finger.bend_matrix.shape[2] - 1))
        finger.bend_matrix[1, 1, idx] *= joint_mult
        finger.bend_matrix[2, 2, idx] *= joint_mult

    # Create cylinder (branch)
    cylinder = Cylinder(
        start=cyl_start,
        direction=cyl_direction,
        normal=cyl_normal,
        base_length=cyl_length,
        base_radius=cyl_radius,
        density=cyl_density,
    )
    sim.append(cylinder)

    # Apply boundary conditions
    sim.constrain(finger).using(
        OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )
    sim.constrain(cylinder).using(
        FixedConstraint,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    if args.sol == "nonuniform_tendon" or args.sol == "change_tendon_direction":
        tendon_actuation = sim.add_forcing_to(finger).using(
            TendonForcesRamp,
            vertebra_height=vertebra_height,
            num_vertebrae=num_vertebrae,
            first_vertebra_node=first_vertebra_node,
            final_vertebra_node=final_vertebra_node,
            vertebra_mass=vertebra_mass,
            tension=tension,
            vertebra_height_orientation=v_height_dir,
            n_elements=n_elements,
            ramp_up_time=1.0,
            use_gradient=True,
            center_seek=True, # Direction Steering ON
            cyl_center=cyl_center_fixed,
            vertebra_nodes_list=vertebra_nodes,
        )
    else:
        tendon_actuation = sim.add_forcing_to(finger).using(
            TendonForces,
            vertebra_height=vertebra_height,
            num_vertebrae=num_vertebrae,
            first_vertebra_node=first_vertebra_node,
            final_vertebra_node=final_vertebra_node,
            vertebra_mass=vertebra_mass,
            tension=tension,
            vertebra_height_orientation=v_height_dir,
            n_elements=n_elements,
            vertebra_nodes_list=vertebra_nodes,
        )

    # Gravity
    sim.add_forcing_to(finger).using(GravityForces, np.array([0.0, 0.0, -9.80665]))

    # Apply it to the first 5 nodes (proximal region)
    sim.add_forcing_to(finger).using(
        BodyWeightForcing, 
        force_vector=body_weight_force, 
        node_indices=np.arange(0, 5)
    )

    # Damping
    if damping_constant > 0.0:
        sim.dampen(finger).using(
            AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=time_step,
        )

    # Contact between finger and cylinder
    sim.detect_contact_between(finger, cylinder).using(
        RodCylinderContact,
        k=k_contact, 
        nu=nu_contact,
        velocity_damping_coefficient=vel_damp_contact,
        friction_coefficient=mu_contact,
    )

    # --- PHYSICAL CONTACT STIFFNESS CALCULATION ---
    dx = base_length / n_elements
    # Assuming contact width is roughly the radius of the finger for a solid grip
    contact_width = base_radius 
    area_per_element = contact_width * dx

    # k_contact = (E * Area) / L_characteristic
    # L_characteristic is the thickness of the material being compressed (finger diameter)
    k_contact_physical = (E * area_per_element) / (2 * base_radius)

    print(f"[PHYSICS] Calculated k_contact: {k_contact_physical:.2e} N/m")


    # Callbacks for data collection
    class CB(CallBackBaseClass):
        def __init__(self, step_skip, callback_params, tendon_forcing_object):
            super().__init__()
            self.every = step_skip
            self.callback_params = callback_params
            self.tendon_forcing = tendon_forcing_object

        def make_callback(self, system, time, current_step):
            if np.isnan(system.position_collection).any():
                print(f"[ERROR] NaN in position at step {current_step}, time {time:.4f}")
                sys.exit(1)
            if np.isnan(system.velocity_collection).any():
                print(f"[ERROR] NaN in velocity at step {current_step}, time {time:.4f}")
                sys.exit(1)
            if current_step % self.every == 0:
                # --- Essential Tracking ---
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                
                # --- Geometry & Kinematics ---
                self.callback_params["position"].append(system.position_collection.copy())
                self.callback_params["velocity"].append(system.velocity_collection.copy())
                self.callback_params["acceleration"].append(system.acceleration_collection.copy())
                self.callback_params["omega"].append(system.omega_collection.copy())
                self.callback_params["alpha"].append(system.alpha_collection.copy())
                self.callback_params["directors"].append(system.director_collection.copy())
                self.callback_params["radius"].append(system.radius.copy())
                self.callback_params["lengths"].append(system.lengths.copy())
                self.callback_params["tangents"].append(system.tangents.copy())

                # --- Forces & Moments ---
                self.callback_params["internal_forces"].append(system.internal_forces.copy())
                self.callback_params["internal_torques"].append(system.internal_torques.copy())
                self.callback_params["external_forces"].append(system.external_forces.copy())
                self.callback_params["external_torques"].append(system.external_torques.copy())

                # --- Strains & Stresses ---
                self.callback_params["sigma"].append(system.sigma.copy())      # Shear/Stretch
                self.callback_params["kappa"].append(system.kappa.copy())      # Curvature/Twist
                self.callback_params["internal_stress"].append(system.internal_stress.copy())
                self.callback_params["internal_couple"].append(system.internal_couple.copy())
                self.callback_params["dilatation"].append(system.dilatation.copy())
                self.callback_params["dilatation_rate"].append(system.dilatation_rate.copy())
                self.callback_params["voronoi_dilatation"].append(system.voronoi_dilatation.copy())

    data = defaultdict(list)
    sim.collect_diagnostics(finger).using(CB, step_skip=step_skip, callback_params=data, 
                                         tendon_forcing_object=tendon_actuation)

    sim.finalize()

    timestepper = PositionVerlet()
    integrate(timestepper, sim, final_time, total_steps)

    data_to_save = {key: np.array(value) for key, value in data.items()}

    for arg_name, arg_value in vars(args).items():
        data_to_save[f"arg_{arg_name}"] = np.array([arg_value])

    # Capture calculated constants and fixed geometry
    data_to_save["E"] = np.array([E])
    data_to_save["G"] = np.array([G])
    data_to_save["tension"] = np.array([tension])
    data_to_save["nu_contact"] = np.array([nu_contact])
    data_to_save["mu_contact"] = np.array([mu_contact])
    data_to_save["base_length"] = np.array([base_length])
    data_to_save["vertebra_nodes"] = vertebra_nodes
    data_to_save["dt_critical"] = np.array([dt_critical])
    data_to_save["time_step"] = np.array([time_step])

    data_to_save["bend_matrix"] = finger.bend_matrix
    data_to_save["shear_matrix"] = finger.shear_matrix
    data_to_save["mass"] = finger.mass
    data_to_save["density"] = finger.density
    data_to_save["rest_lengths"] = finger.rest_lengths
    data_to_save["rest_kappa"] = finger.rest_kappa
    data_to_save["rest_sigma"] = finger.rest_sigma
    data_to_save["mass_second_moment_of_inertia"] = finger.mass_second_moment_of_inertia

    data_to_save["cyl_position"] = cylinder.position_collection.copy() # Center point
    data_to_save["cyl_directors"] = cylinder.director_collection.copy() # Orientation
    data_to_save["cyl_radius"] = np.array([cylinder.radius])
    data_to_save["cyl_length"] = np.array([cylinder.length])

    # --- Automated Metric Calculation at T=Final ---
    # Get final state
    final_pos = data["position"][-1]      # (3, N)
    final_forces = data["external_forces"][-1] # (3, N)
    normal_forces = np.linalg.norm(final_forces, axis=0)

    contact_idx = np.where(normal_forces > 1e-1)[0]
    contact_vertices = final_pos[:, contact_idx].T
    cyl_axis_pos = cylinder.position_collection[:, 0]
    contact_normals = contact_vertices - cyl_axis_pos
    contact_normals /= np.linalg.norm(contact_normals, axis=1)[:, None]

    ext_force = np.array([0.0, 0.0, -body_mass * gravity])
    ext_torque = np.array([0.0, 0.0, 0.0])
    body_wrench = np.concatenate([ext_force, ext_torque])
    is_force_closure = check_force_closure(contact_vertices, contact_normals, mu_contact, external_wrench=body_wrench)
    energy_score = compute_total_energy(finger, normal_forces[contact_idx], k_contact)

    data_to_save["metric_is_force_closure"] = np.array([is_force_closure])
    data_to_save["metric_energy_total"] = np.array([energy_score])
    data_to_save["metric_contact_count"] = np.array([len(contact_idx)])
    print(f"[METRICS] Force Closure Stable: {is_force_closure} under {body_mass}kg load  Total Energy: {energy_score:.6f} J  Contact Points: {len(contact_idx)}")

    csv_path = os.path.join(base_outdir, f"contact_log_{run_id}_{suffix}.csv")

    pos_data = data["position"]
    vel_data = data["velocity"]
    if len(pos_data) < 2:
        raise RuntimeError("No saved frames")
    
    # --------------------------
    # Contact logging (debug)
    # --------------------------
    cyl_center = cylinder.position_collection[:, 0].copy()
    axis_cand = [
        cylinder.director_collection[2, :, 0].copy(),
        cylinder.director_collection[0, :, 0].copy(),
    ]

    dt_saved = step_skip * time_step

    best = None
    for a in axis_cand:
        total_contacts = 0
        min_rad = np.inf
        for P, V in zip(pos_data, vel_data):
            (_, _, _, _, _, radial_dist, _) = compute_contact_metrics_frame(
                P, V, cyl_center, a, cyl_radius, k_contact, mu_contact, base_radius, cyl_length
            )
            min_rad = min(min_rad, float(np.min(radial_dist)))
            total_contacts += int(np.sum((cyl_radius - radial_dist) > 0.0))
        best = max(best, (total_contacts, min_rad, a), key=lambda x: x[0]) if best else (total_contacts, min_rad, a)

    total_contacts, min_rad, cyl_axis = best
    print(f"[CONTACT DEBUG] total_contact_node_hits_over_all_frames={total_contacts}")
    print(f"[CONTACT DEBUG] min_distance_to_cylinder_axis={min_rad:.6f} (radius={cyl_radius:.6f})")
    if total_contacts == 0:
        print("[CONTACT DEBUG] No penetration detected => rod never actually reaches the cylinder (or axis dir wrong).")

    first_contact_frame = None
    max_normal_force_overall = 0.0
    contact_data = []

    with open(os.path.join(base_outdir, f"contact_log_{run_id}_{suffix}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "time", "node_idx",
            "node_x", "node_y", "node_z",  # Contact node position
            "radial_dist", "overlap",
            "normal_force", "normal_velocity", "tangential_speed", "friction_force",
            "cyl_center_x", "cyl_center_y", "cyl_center_z",  # Cylinder center
            "cyl_axis_x", "cyl_axis_y", "cyl_axis_z",  # Cylinder axis direction
            "cyl_radius"  # Cylinder radius
        ])

        for frame_idx, (P, V) in enumerate(zip(pos_data, vel_data)):
            t = frame_idx * dt_saved
            (idx, nF, nV, tS, fF, radial_dist_all, overlap_all) = compute_contact_metrics_frame(
                P, V, cyl_center, cyl_axis, cyl_radius, k_contact, mu_contact, base_radius, cyl_length
            )
            # print("overlap average value:", np.mean(overlap_all))

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
                    float(cylinder.director_collection[2, 0, 0]), float(cylinder.director_collection[2, 1, 0]), float(cylinder.director_collection[2, 2, 0]),
                    float(cyl_radius),
                ])

    if first_contact_frame is None:
        print("[CONTACT] No contact in any saved frame. (Likely cylinder too far, too small, or axis mismatch.)")
    else:
        print(f"[CONTACT] First contact frame={first_contact_frame}  t={first_contact_frame*dt_saved:.4f}s")
        print(f"[CONTACT] Max normal force over all frames: {max_normal_force_overall:.6f} N")
        print(f"[CONTACT] Wrote {os.path.join(base_outdir, f'contact_log_{run_id}_{suffix}.csv')}")

    print(f"\n[CONTACT] Starting geometric metrics check for: {csv_path}")
    is_fc = False
    metrics = {
        "num_contacts": 0,
        "angular_span": 0.0,
        "total_normal_force": 0.0,
        "total_friction_force": 0.0,
        "max_normal_force": 0.0
    }
    try:
        is_fc, metrics = analyze_grasp_from_log(csv_path)
        data_to_save["geometric_success"] = np.array([is_fc])
        data_to_save["num_contacts"] = np.array([metrics["num_contacts"]])
        data_to_save["angular_span"] = np.array([metrics["angular_span"]])
        data_to_save["total_normal_force"] = np.array([metrics["total_normal_force"]])
        data_to_save["total_friction_force"] = np.array([metrics["total_friction_force"]])
        print(f"\n[FORM CLOSURE] {'ACHIEVED' if is_fc else 'NOT ACHIEVED'}")
        print(f"  Contacts: {metrics['num_contacts']}")
        print(f"  Angular span: {metrics['angular_span']:.1f}°")
        print(f"  Total normal force: {metrics['total_normal_force']:.6f} N")
        print(f"  Max normal force: {metrics['max_normal_force']:.6f} N")
        print(f"  Total friction force: {metrics['total_friction_force']:.6f} N")



        # Automatically save a 2D projection plot for this specific run
        plot_path = os.path.join(base_outdir, f"contact_plot_{run_id}_{suffix}.png")
        plot_contacts_2d_from_log(csv_path, output_path=plot_path, show_plot=False)
        print(f"[CONTACT] 2D Contact plot saved to: {plot_path}")

        # stable closure: can the squirrel hang without falling?
        # Calculate masses
        rod_mass = np.sum(finger.mass)
        v_mass_total = num_vertebrae * vertebra_mass
        
        is_stable, stab_metrics = analyze_stable_closure_from_log(
            csv_path,
            rod_mass=rod_mass,
            v_mass_total=v_mass_total,
            args=args
        )
        data_to_save["stability_success"] = np.array([is_stable])
        data_to_save["total_load_n"] = np.array([stab_metrics["total_load_n"]])
        data_to_save["total_support_n"] = np.array([stab_metrics["total_support_n"]])
        data_to_save["stability_margin"] = np.array([stab_metrics["margin"]])
        
        print(f"\n[STABILITY - HANGING]")
        print(f"  Load: {stab_metrics['total_load_n']:.4f} N")
        print(f"  Vertical Support: {stab_metrics['total_support_n']:.4f} N")
        print(f"  Stable: {is_stable} (Margin: {stab_metrics['margin']:.2f}x)")


        cycle_time, strength, slip_res = calculate_nist_scores(
            data, cyl_radius, base_radius, k_contact, mu_contact, args.body_mass, gravity
        )
        grasp_data = {
            "angular_span": metrics['angular_span'], # Calculate from your contact log
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

        data_to_save["final_grasp_score"] = np.array([final_score])
        data_to_save["breakdown_of_score"] = np.array([breakdown])

        filename = os.path.join(base_outdir, f"master_log_{run_id}_{suffix}.npz")
        np.savez_compressed(filename, **data_to_save)

        print(f"Archive Complete: {filename}")

        print(f"\n{'='*30}")
        print(f"FINAL GRASP QUALITY: {final_score:.4f}")
        print(f"{'='*30}")
        for key, val in breakdown.items():
            print(f" - {key}: {val:.3f}")

    except Exception as e:
        print(f"[ERROR] Error during metrics calculation: {e}")

    summary_file = os.path.join(base_outdir, f"sweep_summary.csv")
    file_exists = os.path.isfile(summary_file)

    # The data we want to save
    row_data = [
        run_id, args.sol, E, tension, cyl_radius, args.approach_deg,
        is_fc, metrics['angular_span'], metrics['num_contacts'], 
        data_to_save.get("metric_energy_total", [0])[0], final_score, breakdown,
    ]

    # We use 'a' for append mode
    with open(summary_file, mode='a', newline='') as f:
        # --- LOCKING MECHANISM ---
        # This prevents other processes from writing to the file at the same time
        try:
            fcntl.flock(f, fcntl.LOCK_EX) # Exclusive lock
            
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["run_id", "sol", "E", "tension", "cyl_rad", "approach_deg", 
                                "geometric_fc", "angular_span", "num_contacts", "total_energy", 
                                "final_score", "breakdown"])
            
            writer.writerow(row_data)
            f.flush() # Force write to disk
            
        finally:
            fcntl.flock(f, fcntl.LOCK_UN) # Release the lock

        if args.debug:
            print("[DEBUG] Plotting total force magnitudes and directions.")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    idx = 0

    def make_frame(t):
        nonlocal idx
        ax.clear()
        rod_radius = base_radius
        P = pos_data[idx]
        ax.scatter(P[0], P[1], P[2], s=6)
        step = 5 
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
        for i in range(0, P.shape[1], step):
            x = rod_radius * np.cos(u) * np.sin(v) + P[0, i]
            y = rod_radius * np.sin(u) * np.sin(v) + P[1, i]
            z = rod_radius * np.cos(v) + P[2, i]
            ax.plot_surface(x, y, z, alpha=0.2)
        for v_idx in vertebra_nodes:
            v_idx = int(np.clip(v_idx, 0, P.shape[1] - 1))
            ax.scatter(
                P[0, v_idx], P[1, v_idx], P[2, v_idx],
                color="red",
                s=20,
                depthshade=False,
                zorder=10,
            )

        ax.set_xlim(-0.02, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_zlim(-0.10, 0.10)

        # --- Force visualization of forces ---
        if args.debug:
            F = data["external_forces"][idx]          # (3, n_nodes)
            mag = np.linalg.norm(F, axis=0)  # (n_nodes,)

            force_scale = 0.02
            step_nodes  = 4

            for i in range(0, F.shape[1], step_nodes):
                if mag[i] < 1e-6:
                    continue
                x, y, z = P[0, i], P[1, i], P[2, i]
                fx, fy, fz = F[0, i], F[1, i], F[2, i]
                ax.quiver(
                    x, y, z,
                    fx, fy, fz,
                    length=force_scale,
                    normalize=True,
                    color="magenta",
                )

            for v_idx in vertebra_nodes:
                v_idx = int(v_idx)
                f_vec = F[:, v_idx]
                if np.linalg.norm(f_vec) > 1e-5:
                    ax.quiver(
                        P[0, v_idx], P[1, v_idx], P[2, v_idx],
                        f_vec[0], f_vec[1], f_vec[2],
                        length=0.02,
                        color="cyan",
                    )

            base_pos = data["position"][-1][:, :5].mean(axis=1)
            scale_factor = 0.01 
            f_vec = body_weight_force * scale_factor
            
            # Draw the arrow
            ax.quiver(
                base_pos[0], base_pos[1], base_pos[2],
                f_vec[0], f_vec[1], f_vec[2],
                color='red', linewidth=3, label=f'Body Weight ({args.body_mass}kg)',
                arrow_length_ratio=0.3
            )
            
        center = cylinder.position_collection[:, 0]
        axis_dir = cylinder.director_collection[2, :, 0]
        draw_cylinder(ax, center, axis_dir, cyl_radius, cyl_length, color="black", alpha=0.35)

        # front view: from -Y looking toward +Y
        ax.view_init(elev=0, azim=-90)

        idx = min(idx + 1, len(pos_data) - 1)
        return mplfig_to_npimage(fig)

    clip = VideoClip(make_frame, duration=final_time)
    clip.write_videofile(video_path, codec="libx264", fps=rendering_fps, logger=None)
    plt.close(fig)

    if args.debug:
        fig_live = plt.figure(figsize=(10, 8))
        ax_live = fig_live.add_subplot(111, projection="3d")

        P = pos_data[-1] 
        
        ax_live.scatter(P[0], P[1], P[2], s=10)
        for v_idx in vertebra_nodes:
            v_idx = int(np.clip(v_idx, 0, P.shape[1] - 1))
            ax_live.scatter(P[0, v_idx], P[1, v_idx], P[2, v_idx], color="red", s=40)

        center = cylinder.position_collection[:, 0]
        axis_dir = cylinder.director_collection[2, :, 0]
        draw_cylinder(ax_live, center, axis_dir, cyl_radius, cyl_length, color="black", alpha=0.3)

        ax_live.set_xlim(-0.02, 0.12)
        ax_live.set_ylim(-0.12, 0.12)
        ax_live.set_zlim(-0.10, 0.10)
        ax_live.set_title(f"Steady State - Rotate with Mouse (E={E:.1e})")

        ax_live.view_init(elev=30, azim=45)

        plt.show()
    else:
        plt.close(fig)

    print(f"[OK] saved {video_path}")


if __name__ == "__main__":
    main()