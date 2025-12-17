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

import os
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import csv

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


def apply_rigid_links_soft_joints(finger, rigid_mult, joint_indices, joint_mult):
    """
    Make rod globally rigid-ish, then locally soften a few "joint" elements.
    - rigid_mult multiplies bend_matrix everywhere (links)
    - joint_mult multiplies bend_matrix at joint_indices (joints)
    """
    finger.bend_matrix *= rigid_mult

    for j in joint_indices:
        j = int(np.clip(j, 0, finger.bend_matrix.shape[2] - 1))
        finger.bend_matrix[1, 1, j] *= joint_mult
        finger.bend_matrix[2, 2, j] *= joint_mult


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


def run_one_manual():

    outdir = "manual_videos"
    os.makedirs(outdir, exist_ok=True)
    video_path = os.path.join(outdir, "manual_run.mp4")

    # Material (Pa)
    E = 3.0e5
    G = 1.2e5

    # Tendon
    tension = 0.5 # N

    # Finger geometry (10cm long, 1cm diameter)
    base_length = 0.10
    base_radius = 0.005
    n_elements = 80 # ~1.25mm per element
    density = 997.7

    # Joint model (3 joints / 4 links)
    rigid_mult = 1e2 # links stiffness multiplier
    joint_indices = [30, 50, 68]
    joint_mult = 1e-7 # joints relative to links (smaller = softer joints)

    # Damping (internal)
    damping_constant = 0

    # Cylinder geometry/pose
    cyl_radius = 0.02
    cyl_length = 0.20
    cyl_density = 1200.0

    cyl_start = np.array([0.025, -cyl_length / 2.0, -0.02])

    cyl_direction = np.array([0.0, 1.0, 0.0])
    cyl_normal = np.array([1.0, 0.0, 0.0])

    # Contact params
    k_contact = 2e2
    nu_contact = 5.0
    mu_contact = 6 # 0.4
    vel_damp_contact = 2e1 # 5e1

    # Tendon geometry
    vertebra_mass = 0.003
    num_vertebrae = 16
    first_vertebra_node = 3
    final_vertebra_node = n_elements - 3
    vertebra_height = 0.010
    vertebra_height_orientation = np.array([0.0, 0.0, -1.0])

    # Simulation time
    final_time = 2.0
    time_step = 1.8e-6
    rendering_fps = 30.0
    total_steps = int(final_time / time_step)
    step_skip = int(1.0 / (rendering_fps * time_step))

    print("\n==== MANUAL RUN ====")
    print(f"E={E:.2e}  G={G:.2e}  tension={tension:.2f}")
    print(f"rigid_mult={rigid_mult:.1e}  joint_mult={joint_mult:.1e}  joints={joint_indices}")
    print(f"cyl_start={cyl_start}  cyl_radius={cyl_radius}  cyl_length={cyl_length}")
    print(f"contact: k={k_contact} nu={nu_contact} mu={mu_contact} vel_damp={vel_damp_contact}")
    print(f"damping_constant={damping_constant}")

    sim = SquirrelFingerSimulator()

    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    finger = CosseratRod.straight_rod(
        n_elements=n_elements,
        start=np.array([0.0, 0.0, 0.0]),
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=E,
        shear_modulus=G,
    )
    sim.append(finger)

    apply_rigid_links_soft_joints(
        finger,
        rigid_mult=rigid_mult,
        joint_indices=joint_indices,
        joint_mult=joint_mult,
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
        TendonForces,
        vertebra_height=vertebra_height,
        num_vertebrae=num_vertebrae,
        first_vertebra_node=first_vertebra_node,
        final_vertebra_node=final_vertebra_node,
        vertebra_mass=vertebra_mass,
        tension=tension,
        vertebra_height_orientation=vertebra_height_orientation,
        n_elements=n_elements,
    )

    # sim.add_forcing_to(finger).using(GravityForces, np.array([0.0, 0.0, -9.80665]))

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
        def __init__(self, step_skip, data):
            super().__init__()
            self.every = step_skip
            self.data = data

        def make_callback(self, system, time, current_step):
            if current_step % self.every == 0:
                self.data["pos"].append(system.position_collection.copy())
                self.data["vel"].append(system.velocity_collection.copy())
                if np.isnan(system.position_collection).any():
                    raise RuntimeError("NaN encountered")

    data = defaultdict(list)
    sim.collect_diagnostics(finger).using(CB, step_skip=step_skip, data=data)

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
    # IMPORTANT: cylinder axis direction — depending on Elastica's cylinder director convention,
    # this might be [2,:,0] or [0,:,0]. We'll try both and pick the one that yields more contact.
    axis_cand = [
        cylinder.director_collection[2, :, 0].copy(),
        cylinder.director_collection[0, :, 0].copy(),
    ]

    k_contact = k_contact        # uses your manual params
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

    print(f"[OK] saved {video_path}")


if __name__ == "__main__":
    run_one_manual()