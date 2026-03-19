import os
import csv
import argparse
import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from elastica.modules import BaseSystemCollection, Constraints, Forcing, CallBacks, Damping
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import NoForces, GravityForces
from elastica.dissipation import AnalyticalLinearDamper
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate


###################################################
# SIMULATOR
###################################################
class SingleJointSimulator(
    BaseSystemCollection,
    Constraints,
    Forcing,
    CallBacks,
    Damping,
):
    pass


###################################################
# UTILITIES
###################################################
def parse_vec3_from_csv(text, arg_name):
    try:
        vals = [float(x.strip()) for x in text.split(",")]
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be 3 comma-separated numbers, got '{text}'") from exc
    if len(vals) != 3:
        raise ValueError(f"{arg_name} must have exactly 3 values, got '{text}'")
    return np.array(vals, dtype=float)


def parse_float_list_with_resize(text, target_len, arg_name):
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be a comma-separated list of floats, got '{text}'") from exc

    if len(vals) == 0:
        raise ValueError(f"{arg_name} must contain at least one float")

    if len(vals) < target_len:
        vals.extend([vals[-1]] * (target_len - len(vals)))
    elif len(vals) > target_len:
        vals = vals[:target_len]

    return np.array(vals, dtype=float)


def safe_unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(c))


###################################################
# FORCING
###################################################
class ConstantNodeForce(NoForces):
    """
    Applies a constant force vector distributed across selected nodes.
    """

    def __init__(self, force_vector, node_indices):
        self.force_vector = np.asarray(force_vector, dtype=float).reshape(3)
        self.node_indices = np.asarray(node_indices, dtype=int)

    def apply_forces(self, system, time=0.0):
        if self.node_indices.size == 0:
            return
        force_per_node = self.force_vector / float(self.node_indices.size)
        for idx in self.node_indices:
            system.external_forces[:, idx] += force_per_node


###################################################
# CALLBACK
###################################################
class JointDiagnostics(CallBackBaseClass):
    def __init__(self, step_skip, callback_params, joint_element_idx, base_position0, tip_position0):
        super().__init__()
        self.every = int(step_skip)
        self.callback_params = callback_params
        self.joint_element_idx = int(joint_element_idx)
        self.base_position0 = np.asarray(base_position0, dtype=float).reshape(3)
        self.tip_position0 = np.asarray(tip_position0, dtype=float).reshape(3)

    def make_callback(self, system, time, current_step):
        if current_step % self.every != 0:
            return

        self.callback_params["time"].append(float(time))
        self.callback_params["step"].append(int(current_step))
        self.callback_params["position"].append(system.position_collection.copy())
        self.callback_params["velocity"].append(system.velocity_collection.copy())
        self.callback_params["tangents"].append(system.tangents.copy())
        self.callback_params["external_forces"].append(system.external_forces.copy())
        self.callback_params["internal_forces"].append(system.internal_forces.copy())
        self.callback_params["kappa"].append(system.kappa.copy())

        j = self.joint_element_idx
        t_before = system.tangents[:, max(j - 1, 0)]
        t_after = system.tangents[:, min(j, system.tangents.shape[1] - 1)]
        theta = angle_between(t_before, t_after)
        self.callback_params["joint_angle"].append(theta)

        base_pos = system.position_collection[:, 0]
        tip_pos = system.position_collection[:, -1]
        self.callback_params["base_position"].append(base_pos.copy())
        self.callback_params["tip_position"].append(tip_pos.copy())
        self.callback_params["tip_disp_from_initial"].append((tip_pos - self.tip_position0).copy())
        self.callback_params["tip_disp_mag"].append(float(np.linalg.norm(tip_pos - self.tip_position0)))


###################################################
# MAIN
###################################################
def main():
    parser = argparse.ArgumentParser(description="Single joint benchmark extracted from squirrel finger.")

    # Match finger.py defaults
    parser.add_argument("--E", type=float, default=2e7, help="Young's modulus (Pa)")
    parser.add_argument("--poisson_nu", type=float, default=0.4, help="Poisson's ratio")
    parser.add_argument("--density", type=float, default=1200.0, help="Material density (kg/m^3)")
    parser.add_argument("--damping", type=float, default=0.8, help="Internal damping constant")
    parser.add_argument("--base_len_full", type=float, default=0.05, help="Full finger length (m)")
    parser.add_argument("--base_rad", type=float, default=0.005, help="Finger radius (m)")
    parser.add_argument("--n_elements_full", type=int, default=80, help="Full finger element count")

    # Match finger joint placement defaults
    parser.add_argument("--num_v", type=int, default=3, help="Number of vertebrae in full finger")
    parser.add_argument("--v_start", type=int, default=30, help="First vertebra node index in full finger")
    parser.add_argument("--v_end", type=int, default=62, help="Last vertebra node index in full finger")
    parser.add_argument(
        "--joint_softness",
        type=str,
        default="0.001",
        help="Comma-separated joint softness list from full finger; first value is used here",
    )

    # Extracted local specimen settings
    parser.add_argument(
        "--half_window_elems",
        type=int,
        default=8,
        help="How many full-finger elements to keep on each side of the first joint",
    )
    parser.add_argument(
        "--n_elements",
        type=int,
        default=40,
        help="Element count of the extracted specimen",
    )

    # Loading
    parser.add_argument("--load_mag", type=float, default=5.0, help="Applied distal load magnitude (N)")
    parser.add_argument(
        "--load_dir",
        type=str,
        default="0,0,-1",
        help="Applied load direction as x,y,z",
    )
    parser.add_argument(
        "--load_nodes",
        type=int,
        default=1,
        help="Number of distal nodes sharing the applied load",
    )
    parser.add_argument(
        "--include_gravity",
        action="store_true",
        help="Also include gravity in the reduced joint simulation",
    )

    # Time integration
    parser.add_argument("--final_time", type=float, default=1.0, help="Simulation duration (s)")
    parser.add_argument("--save_plot", action="store_true", help="Save a diagnostic plot")
    parser.add_argument("--debug", action="store_true", help="Print extra debug info")

    # Output
    parser.add_argument("--output_dir", type=str, default="single_joint_results", help="Output directory")
    parser.add_argument("--suffix", type=str, default="default", help="Suffix for output files")

    args = parser.parse_args()

    ###################################################
    # Match finger joint properties
    ###################################################
    E = args.E
    nu = args.poisson_nu
    G = E / (2.0 * (1.0 + nu))
    density = args.density
    damping_constant = args.damping
    base_radius = args.base_rad
    base_len_full = args.base_len_full
    n_elements_full = args.n_elements_full

    full_vertebra_nodes = np.linspace(args.v_start, args.v_end, args.num_v, dtype=int)
    first_joint_node_full = int(full_vertebra_nodes[0])

    joint_softness_list = parse_float_list_with_resize(
        args.joint_softness,
        target_len=len(full_vertebra_nodes),
        arg_name="--joint_softness",
    )
    first_joint_softness = float(joint_softness_list[0])

    ###################################################
    # Extract local specimen around first joint
    ###################################################
    dx_full = base_len_full / n_elements_full
    half_window_elems = int(args.half_window_elems)
    if half_window_elems < 2:
        raise ValueError("--half_window_elems should be at least 2")

    specimen_len = 2.0 * half_window_elems * dx_full
    n_elements = int(args.n_elements)
    if n_elements < 4:
        raise ValueError("--n_elements must be at least 4")

    # Put the extracted joint in the center of the reduced specimen
    joint_element_idx = 30

    ###################################################
    # Loading
    ###################################################
    load_dir = parse_vec3_from_csv(args.load_dir, "--load_dir")
    load_dir = safe_unit(load_dir)
    if np.linalg.norm(load_dir) < 1e-12 and args.load_mag > 0.0:
        raise ValueError("--load_dir cannot be zero if --load_mag > 0")

    load_vector = float(args.load_mag) * load_dir
    load_nodes = max(1, int(args.load_nodes))

    ###################################################
    # Time step
    ###################################################
    wave_speed = np.sqrt(E / density)
    dx = specimen_len / n_elements
    dt_critical = dx / wave_speed
    time_step = 0.1 * dt_critical
    total_steps = max(1, int(args.final_time / time_step))
    step_skip = max(1, total_steps // 300)

    ###################################################
    # Output paths
    ###################################################
    os.makedirs(args.output_dir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"single_joint_{run_id}_{args.suffix}"
    npz_path = os.path.join(args.output_dir, stem + ".npz")
    csv_path = os.path.join(args.output_dir, stem + "_summary.csv")
    plot_path = os.path.join(args.output_dir, stem + ".png")

    print("\n" + "=" * 70)
    print("SINGLE JOINT BENCHMARK")
    print("=" * 70)
    print(f"E                     = {E:.3e} Pa")
    print(f"G                     = {G:.3e} Pa")
    print(f"density               = {density:.1f} kg/m^3")
    print(f"base_radius           = {base_radius:.6f} m")
    print(f"full finger length    = {base_len_full:.6f} m")
    print(f"full finger n_elem    = {n_elements_full}")
    print(f"first full joint node = {first_joint_node_full}")
    print(f"first joint softness  = {first_joint_softness:.6f}")
    print(f"specimen_len          = {specimen_len:.6f} m")
    print(f"specimen n_elements   = {n_elements}")
    print(f"joint_element_idx     = {joint_element_idx}")
    print(f"load_vector           = {load_vector}")
    print(f"time_step             = {time_step:.3e} s")
    print(f"total_steps           = {total_steps}")
    print("=" * 70 + "\n")

    ###################################################
    # Build system
    ###################################################
    sim = SingleJointSimulator()

    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    mass_second_moment_of_inertia = 0.25 * np.pi * base_radius**4

    rod = CosseratRod.straight_rod(
        n_elements=n_elements,
        start=start,
        direction=direction,
        normal=normal,
        base_length=specimen_len,
        base_radius=base_radius,
        density=density,
        youngs_modulus=E,
        shear_modulus=G,
        mass_second_moment_of_inertia=mass_second_moment_of_inertia,
    )
    sim.append(rod)

    # Apply same joint-softening concept as finger.py, using first joint softness
    rod.bend_matrix[1, 1, joint_element_idx] *= first_joint_softness
    rod.bend_matrix[2, 2, joint_element_idx] *= first_joint_softness

    # Fix base
    sim.constrain(rod).using(
        OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    # Distal point load
    distal_nodes = np.arange(max(0, n_elements + 1 - load_nodes), n_elements + 1, dtype=int)
    sim.add_forcing_to(rod).using(
        ConstantNodeForce,
        force_vector=load_vector,
        node_indices=distal_nodes,
    )

    if args.include_gravity:
        sim.add_forcing_to(rod).using(
            GravityForces,
            np.array([0.0, 0.0, -9.80665]),
        )

    if damping_constant > 0.0:
        sim.dampen(rod).using(
            AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=time_step,
        )

    ###################################################
    # Diagnostics
    ###################################################
    data = defaultdict(list)
    base_position0 = rod.position_collection[:, 0].copy()
    tip_position0 = rod.position_collection[:, -1].copy()

    sim.collect_diagnostics(rod).using(
        JointDiagnostics,
        step_skip=step_skip,
        callback_params=data,
        joint_element_idx=joint_element_idx,
        base_position0=base_position0,
        tip_position0=tip_position0,
    )

    sim.finalize()

    timestepper = PositionVerlet()
    integrate(timestepper, sim, final_time=args.final_time, n_steps=total_steps)

    ###################################################
    # Convert to arrays
    ###################################################
    arr = {k: np.array(v) for k, v in data.items()}

    ###################################################
    # Final metrics
    ###################################################
    joint_angle_final = float(arr["joint_angle"][-1])

    tip_position_final = arr["tip_position"][-1]
    tip_disp_vec = tip_position_final - tip_position0
    tip_disp_mag = float(np.linalg.norm(tip_disp_vec))

    j = joint_element_idx
    tangents_final = arr["tangents"][-1]
    tangent_before = tangents_final[:, max(j - 1, 0)]
    tangent_after = tangents_final[:, min(j, tangents_final.shape[1] - 1)]

    # Force at loaded distal nodes
    ext_force_last = arr["external_forces"][-1]
    distal_force_vec = np.sum(ext_force_last[:, distal_nodes], axis=1)
    distal_force_mag = float(np.linalg.norm(distal_force_vec))

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Applied distal force vector     : {distal_force_vec}")
    print(f"Applied distal force magnitude  : {distal_force_mag:.6f} N")
    print(f"Final joint angle               : {joint_angle_final:.6f} rad ({np.degrees(joint_angle_final):.3f} deg)")
    print(f"Final tip displacement vector   : {tip_disp_vec}")
    print(f"Final tip displacement magnitude: {tip_disp_mag:.6f} m")
    print(f"Tangent before joint            : {tangent_before}")
    print(f"Tangent after joint             : {tangent_after}")
    print("=" * 70 + "\n")

    ###################################################
    # Save NPZ
    ###################################################
    np.savez_compressed(
        npz_path,
        time=arr["time"],
        step=arr["step"],
        position=arr["position"],
        velocity=arr["velocity"],
        tangents=arr["tangents"],
        external_forces=arr["external_forces"],
        internal_forces=arr["internal_forces"],
        kappa=arr["kappa"],
        joint_angle=arr["joint_angle"],
        base_position=arr["base_position"],
        tip_position=arr["tip_position"],
        tip_disp_from_initial=arr["tip_disp_from_initial"],
        tip_disp_mag=arr["tip_disp_mag"],
        E=np.array([E]),
        G=np.array([G]),
        density=np.array([density]),
        damping_constant=np.array([damping_constant]),
        base_radius=np.array([base_radius]),
        specimen_len=np.array([specimen_len]),
        n_elements=np.array([n_elements]),
        first_joint_node_full=np.array([first_joint_node_full]),
        first_joint_softness=np.array([first_joint_softness]),
        joint_element_idx=np.array([joint_element_idx]),
        load_vector=load_vector,
        distal_nodes=distal_nodes,
        time_step=np.array([time_step]),
        joint_angle_final=np.array([joint_angle_final]),
        tip_disp_vec_final=tip_disp_vec,
        tip_disp_mag_final=np.array([tip_disp_mag]),
    )
    print(f"[OK] saved {npz_path}")

    ###################################################
    # Save CSV summary
    ###################################################
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id",
                "suffix",
                "E",
                "G",
                "density",
                "damping_constant",
                "base_radius",
                "specimen_len",
                "n_elements",
                "first_joint_node_full",
                "first_joint_softness",
                "joint_element_idx",
                "load_x",
                "load_y",
                "load_z",
                "load_mag",
                "joint_angle_rad_final",
                "joint_angle_deg_final",
                "tip_disp_x_final",
                "tip_disp_y_final",
                "tip_disp_z_final",
                "tip_disp_mag_final",
            ])
        writer.writerow([
            run_id,
            args.suffix,
            E,
            G,
            density,
            damping_constant,
            base_radius,
            specimen_len,
            n_elements,
            first_joint_node_full,
            first_joint_softness,
            joint_element_idx,
            load_vector[0],
            load_vector[1],
            load_vector[2],
            np.linalg.norm(load_vector),
            joint_angle_final,
            np.degrees(joint_angle_final),
            tip_disp_vec[0],
            tip_disp_vec[1],
            tip_disp_vec[2],
            tip_disp_mag,
        ])
    print(f"[OK] saved {csv_path}")

    ###################################################
    # Optional plot
    ###################################################
    if args.save_plot:
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(arr["time"], arr["joint_angle"])
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("joint angle (rad)")
        ax1.set_title("Joint angle vs time")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(arr["time"], arr["tip_disp_mag"])
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("tip displacement (m)")
        ax2.set_title("Tip displacement vs time")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved {plot_path}")

    if args.debug:
        print("\n[DEBUG]")
        print(f"distal_nodes      = {distal_nodes}")
        print(f"tip_position0     = {tip_position0}")
        print(f"tip_position_final= {tip_position_final}")
        print(f"joint kappa shape = {arr['kappa'][-1].shape}")


if __name__ == "__main__":
    main()