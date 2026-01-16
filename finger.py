"""
single_finger_manual.py

Manually tune:
- E, G, tension
- joint model (rigid_mult, joint_indices, joint_mult)
- cylinder position/size
- contact params
- damping

Outputs:
  ./manual_videos/manual_run.mp4
"""

# TODO ideas to improve contact performance:
# solution 1: approach angle change
# solution 2: increase tendon force towards tip
# solution 3: change tendon force direction to curl
# softer cylinder surface (lower k_contact) --done
# different joint positions

import os
import numpy as np
from collections import defaultdict
import sys
import argparse
import time

import matplotlib
# matplotlib.use("Agg")  # headless
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import csv

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
if "--debug" in sys.argv:
    plt.switch_backend("TkAgg")
    print("[DEBUG] Matplotlib interactive backend enabled.")
else:
    plt.switch_backend("Agg")

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from elastica.modules import BaseSystemCollection, Connections, Constraints, Forcing, CallBacks, Damping, Contact
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC, FixedConstraint
from elastica.external_forces import GravityForces
from elastica.dissipation import AnalyticalLinearDamper
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
from elastica.rigidbody.cylinder import Cylinder
from elastica.contact_forces import RodCylinderContact

from TendonForces import TendonForces


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


# def apply_rigid_links_soft_joints(finger, rigid_mult, joint_indices, joint_mult):
#     """
#     Make rod globally rigid-ish, then locally soften a few "joint" elements.
#     - rigid_mult multiplies bend_matrix everywhere (links)
#     - joint_mult multiplies bend_matrix at joint_indices (joints)
#     """
#     finger.bend_matrix *= rigid_mult

#     for j in joint_indices:
#         j = int(np.clip(j, 0, finger.bend_matrix.shape[2] - 1))
#         finger.bend_matrix[1, 1, j] *= joint_mult
#         finger.bend_matrix[2, 2, j] *= joint_mult

def apply_rigid_links_soft_joints(
    finger,
    rigid_mult,
    joint_indices,
    joint_mult,
    joint_half_width_elems,   # <-- NEW: joint "length" control
):
    """
    - rigid_mult multiplies bend_matrix everywhere
    - for each joint index, soften a band of elements [j-w, ..., j+w]
      so the joint has finite length (not a single hinge element)
    """
    finger.bend_matrix *= rigid_mult

    ne = finger.bend_matrix.shape[2]
    w = int(joint_half_width_elems)

    for j0 in joint_indices:
        j0 = int(np.clip(j0, 0, ne - 1))
        j_start = max(0, j0 - w)
        j_end   = min(ne - 1, j0 + w)

        finger.bend_matrix[1, 1, j_start:j_end+1] *= joint_mult
        finger.bend_matrix[2, 2, j_start:j_end+1] *= joint_mult

def compute_contact_metrics_frame(
    rod_pos,      # (3, n_nodes)
    rod_vel,      # (3, n_nodes)
    cyl_center,   # (3,)
    cyl_axis,     # (3,)
    cyl_radius,   # float
    k,            # normal stiffness (same as RodCylinderContact.k)
    mu,           # friction coefficient
):
    cyl_axis = cyl_axis / (np.linalg.norm(cyl_axis) + 1e-12)

    rel = rod_pos.T - cyl_center[None, :]          # (N,3)
    proj_len = np.dot(rel, cyl_axis)               # (N,)
    proj = np.outer(proj_len, cyl_axis)            # (N,3)
    radial = rel - proj                            # (N,3)
    radial_dist = np.linalg.norm(radial, axis=1)   # (N,)

    # outward unit normal from cylinder axis to node
    normal_vec = np.zeros_like(radial)
    mask = radial_dist > 1e-12
    normal_vec[mask] = radial[mask] / radial_dist[mask, None]

    overlap = cyl_radius - radial_dist             # >0 => penetration/contact
    contact_mask = overlap > 0.0

    normal_force_mag = k * np.clip(overlap, 0.0, None)

    vel = rod_vel.T
    normal_vel = np.sum(vel * normal_vec, axis=1)
    vel_t = vel - normal_vel[:, None] * normal_vec
    tangential_speed = np.linalg.norm(vel_t, axis=1)

    friction_force_mag = mu * normal_force_mag

    idx = np.where(contact_mask)[0]
    return (
        idx,
        normal_force_mag[idx],
        normal_vel[idx],
        tangential_speed[idx],
        friction_force_mag[idx],
        radial_dist,          # return full arrays for debugging
        overlap,              # return full arrays for debugging
    )

class TendonForcesRamp(TendonForces):
    """
    Wrapper that ramps tension from 0 -> tension over ramp_up_time.
    Doesn't require modifying your TendonForces implementation.
    """
    def __init__(self, *args, ramp_up_time=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._tension_nominal = float(kwargs.get("tension", getattr(self, "tension", 0.0)))
        self.ramp_up_time = float(ramp_up_time)

    def _ramp_factor(self, time):
        if self.ramp_up_time <= 0:
            return 1.0
        s = min(1.0, max(0.0, float(time) / self.ramp_up_time))
        # smooth cosine ramp (less jerk than linear)
        return 0.5 * (1.0 - np.cos(np.pi * s))

    def apply_forces(self, system, time=0.0):
        factor = self._ramp_factor(time)
        old = self.tension
        self.tension = self._tension_nominal * factor
        super().apply_forces(system, time)
        self.tension = old

    def apply_torques(self, system, time=0.0):
        factor = self._ramp_factor(time)
        old = self.tension
        self.tension = self._tension_nominal * factor
        super().apply_torques(system, time)
        self.tension = old

def main():

    parser = argparse.ArgumentParser(description="Run Squirrel Finger Simulation.")
    parser.add_argument(
        "--sol", type=str, default="standard",
        choices=["standard", "approach_angle", "nonuniform_tendon", "change_tendon_direction"],
        help="Select the solution for improved curl")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode with plot of total force magnitude and direction.")
    
    # --- MATERIAL & GEOMETRY PARAMS (Defaulted to your current values) ---
    parser.add_argument("--E", type=float, default=2e7, help="Youngs Modulus (Pa)")
    parser.add_argument("--G", type=float, default=7e6, help="Shear Modulus (Pa)")
    parser.add_argument("--tension", type=float, default=0.4, help="Tendon tension (N)")
    parser.add_argument("--damping", type=float, default=0.8, help="Internal damping constant")
    parser.add_argument("--n_elements", type=int, default=80, help="Number of rod elements")
    
    # --- CYLINDER PARAMS ---
    parser.add_argument("--cyl_rad", type=float, default=0.01, help="Cylinder radius (m)")
    parser.add_argument("--k_contact", type=float, default=2e3, help="Contact stiffness")
    
    args = parser.parse_args()

    # Parameters from args (key variables to tune)
    E = args.E
    G = args.G
    tension = args.tension
    n_elements = args.n_elements
    damping_constant = args.damping
    k_contact = args.k_contact
    cyl_radius = args.cyl_rad
    
    # Fixed Geometry
    base_length = 0.10
    base_radius = 0.005
    density = 1200
    rigid_mult = 1 # links stiffness multiplier
    # joint_indices = [30, 50, 68]
    joint_indices = [30, 46, 62]
    joint_mult = 1e-2 # joints relative to links (smaller = softer joints)

    wave_speed = np.sqrt(E / density)
    dx = base_length / n_elements
    dt_critical = dx / wave_speed
    
    cyl_length = 0.20
    cyl_density = 1200.0
    cyl_start = np.array([0.025, -cyl_length / 2.0, -0.025])
    cyl_direction = np.array([0.0, 1.0, 0.0])
    cyl_normal = np.array([1.0, 0.0, 0.0])

    nu_contact = 5.0 # 5.0~20.0
    mu_contact = 0.6 # 0.6~1.0
    vel_damp_contact = 10 # 10~50

    vertebra_mass = 0.002
    num_vertebrae = 4
    first_vertebra_node = 30
    final_vertebra_node = n_elements - 1
    vertebra_height = 0.010
    
    final_time = 2.0
    # time_step = 1.8e-6
    time_step = 0.1 * dt_critical
    rendering_fps = 30.0
    total_steps = int(final_time / time_step)
    step_skip = int(1.0 / (rendering_fps * time_step))

    # Output setup
    base_outdir = "squirrel_paw_results"
    os.makedirs(base_outdir, exist_ok=True)
    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    video_path = os.path.join(base_outdir, f"output_{run_id}.mp4")
    log_path = os.path.join(base_outdir, f"log_{run_id}.txt")

    # Log all arguments used
    with open(log_path, "w") as f:
        f.write(f"Simulation Run: {run_id}\n" + "-"*30 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write(f"Calculated Time Steps: {total_steps}\n")

    print(f"\n==== RUNNING: {args.sol} ====")
    print(f"Logging to: {log_path}")
    print(f"E={E:.2e}  G={G:.2e}  tension={tension:.2f}")
    print(f"rigid_mult={rigid_mult:.1e}  joint_mult={joint_mult:.1e}  joints={joint_indices}")
    print(f"cyl_start={cyl_start}  cyl_radius={cyl_radius}  cyl_length={cyl_length}")
    print(f"contact: k={k_contact} nu={nu_contact} mu={mu_contact} vel_damp={vel_damp_contact}")
    print(f"damping_constant={damping_constant}")

    sim = SquirrelFingerSimulator()

    # Standard/default finger setup
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    start_pos = np.array([0.0, 0.0, 0.0])
    v_height_dir = np.array([0.0, 0.0, -1.0])

    # 3 alternative solutions to improve curling around cylinder
    if args.sol == "approach_angle":
        print(">>> MODE: Approach Angle Change")
        direction = np.array([1.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        normal = np.array([0.0, 1.0, 0.0])
        start_pos = np.array([0.0, 0.0, -0.02])
        v_height_dir = np.array([-1.0, 0.0, 1.0])
        v_height_dir /= np.linalg.norm(v_height_dir)
    elif args.sol == "nonuniform_tendon":
        print(">>> MODE: Non-uniform Tendon Force (Increasing toward tip)")
        # Logic for Solution 2 will go here
        pass

    elif args.sol == "change_tendon_direction":
        print(">>> MODE: Change Tendon Direction to Curl")
        # Logic for Solution 3 will go here
        # (Usually involves varying the v_height_dir along the rod)
        pass
    
    else:
        print(">>> MODE: Standard Horizontal")

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
    )
    sim.append(finger)

    # apply_rigid_links_soft_joints(
    #     finger,
    #     rigid_mult=rigid_mult,
    #     joint_indices=joint_indices,
    #     joint_mult=joint_mult,
    # )

    apply_rigid_links_soft_joints(
        finger,
        rigid_mult=rigid_mult,
        joint_indices=joint_indices,
        joint_mult=joint_mult,
        joint_half_width_elems=1,
    )

    cylinder = Cylinder(
        start=cyl_start,
        direction=cyl_direction,
        normal=cyl_normal,
        base_length=cyl_length,
        base_radius=cyl_radius,
        density=cyl_density,
    )
    sim.append(cylinder)

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

    sim.add_forcing_to(finger).using(
        TendonForcesRamp,
        vertebra_height=vertebra_height,
        num_vertebrae=num_vertebrae,
        first_vertebra_node=first_vertebra_node,
        final_vertebra_node=final_vertebra_node,
        vertebra_mass=vertebra_mass,
        tension=tension,
        vertebra_height_orientation=v_height_dir,
        n_elements=n_elements,
        ramp_up_time=0.2, # New feature to ramp tension
    )

    sim.add_forcing_to(finger).using(GravityForces, np.array([0.0, 0.0, -9.80665]))

    if damping_constant > 0.0:
        sim.dampen(finger).using(
            AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=time_step,
        )

    sim.detect_contact_between(finger, cylinder).using(
        RodCylinderContact,
        k=k_contact,
        nu=nu_contact,
        velocity_damping_coefficient=vel_damp_contact,
        friction_coefficient=mu_contact,
    )



    class CB(CallBackBaseClass):
        def __init__(self, step_skip, callback_params):
            super().__init__()
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step):
            if np.isnan(system.position_collection).any():
                print(f"!!! CRASH !!! NaN detected at step {current_step}, time {time:.4f}")
                sys.exit(1)
            if current_step % self.every == 0:
                self.callback_params["pos"].append(system.position_collection.copy())
                self.callback_params["vel"].append(system.velocity_collection.copy())
                self.callback_params["forces"].append(system.external_forces.copy())
                if np.isnan(system.position_collection).any():
                    raise RuntimeError("NaN encountered")

    data = defaultdict(list)
    sim.collect_diagnostics(finger).using(CB, step_skip=step_skip, callback_params=data)

    sim.finalize()

    timestepper = PositionVerlet()
    integrate(timestepper, sim, final_time, total_steps)

    pos_data = data["pos"]
    vel_data = data["vel"]
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

    k_contact = k_contact 
    mu_contact = mu_contact
    dt_saved = step_skip * time_step

    best = None
    for a in axis_cand:
        total_contacts = 0
        min_rad = np.inf
        for P, V in zip(pos_data, vel_data):
            (_, _, _, _, _, radial_dist, _) = compute_contact_metrics_frame(
                P, V, cyl_center, a, cyl_radius, k_contact, mu_contact
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

    with open("contact_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "time", "node_idx",
            "radial_dist", "overlap",
            "normal_force", "normal_velocity", "tangential_speed", "friction_force"
        ])

        for frame_idx, (P, V) in enumerate(zip(pos_data, vel_data)):
            t = frame_idx * dt_saved
            (idx, nF, nV, tS, fF, radial_dist_all, overlap_all) = compute_contact_metrics_frame(
                P, V, cyl_center, cyl_axis, cyl_radius, k_contact, mu_contact
            )

            if len(idx) > 0:
                frame_max = float(np.max(nF))
                max_normal_force_overall = max(max_normal_force_overall, frame_max)
                if first_contact_frame is None:
                    first_contact_frame = frame_idx

            for j, node_idx in enumerate(idx):
                writer.writerow([
                    frame_idx, t, int(node_idx),
                    float(radial_dist_all[node_idx]),
                    float(overlap_all[node_idx]),
                    float(nF[j]),
                    float(nV[j]),
                    float(tS[j]),
                    float(fF[j]),
                ])

    if first_contact_frame is None:
        print("[CONTACT] No contact in any saved frame. (Likely cylinder too far, too small, or axis mismatch.)")
    else:
        print(f"[CONTACT] First contact frame={first_contact_frame}  t={first_contact_frame*dt_saved:.4f}s")
        print(f"[CONTACT] Max normal force over all frames: {max_normal_force_overall:.6f} N")
        print("[CONTACT] Wrote contact_log.csv")

    if args.debug:
        print("[DEBUG] Plotting total force magnitudes and directions.")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    idx = 0

    def make_frame(t):
        nonlocal idx
        ax.clear()

        P = pos_data[idx]
        ax.scatter(P[0], P[1], P[2], s=6)
        for j in joint_indices:
            j = int(np.clip(j, 0, P.shape[1] - 1))
            ax.scatter(
                P[0, j], P[1, j], P[2, j],
                color="red",
                s=20,
                depthshade=False,
                zorder=10,
            )

        ax.set_xlim(-0.02, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_zlim(-0.10, 0.10)

        # --- Force visualization (magenta) ---
        if args.debug:
            F = data["forces"][idx]          # (3, n_nodes)
            mag = np.linalg.norm(F, axis=0)  # (n_nodes,)

            force_scale = 0.02   # tune for visibility in your axis limits
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
        for j in joint_indices:
            j = int(np.clip(j, 0, P.shape[1] - 1))
            ax_live.scatter(P[0, j], P[1, j], P[2, j], color="red", s=40)

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