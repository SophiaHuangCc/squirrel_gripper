"""
single_finger_sweep.py

Batch sweep for a single "finger" Cosserat rod perching on a cylinder.

Key design change vs earlier version:
- Model finger as "rigid-ish links + compliant joints"
  (3 joints / 4 links to start).

Outputs one mp4 per combination into ./sweep_videos/
"""

import os
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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


# ---------------- Simulator class ---------------- #
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


# ---------------- Helpers ---------------- #
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
    Make the rod globally rigid-ish, then locally soften a few "joint" elements.
    - rigid_mult multiplies bend_matrix everywhere (links)
    - joint_mult multiplies bend_matrix at joint_indices (joints)
    """
    finger.bend_matrix *= rigid_mult

    for j in joint_indices:
        # guard against out-of-range
        j = int(np.clip(j, 0, finger.bend_matrix.shape[2] - 1))
        finger.bend_matrix[1, 1, j] *= joint_mult
        finger.bend_matrix[2, 2, j] *= joint_mult


def apply_segmented_profile(finger, segment_mults):
    """
    10-segment profile: each segment gets a uniform multiplier.
    segment_mults: list length n_segments
    """
    n_segments = len(segment_mults)
    n = finger.bend_matrix.shape[2]  # ~ n_elements-1
    edges = np.linspace(0, n, n_segments + 1).astype(int)

    for s in range(n_segments):
        a, b = edges[s], edges[s + 1]
        finger.bend_matrix[:, :, a:b] *= segment_mults[s]


# ---------------- One full simulation run ---------------- #
def run_one(
    run_id: str,
    # material
    E: float,
    G: float,
    # tendon
    tension: float,
    # stiffness design
    stiffness_mode: str,  # "joints" or "segments"
    rigid_mult: float = 1.0,
    joint_indices=None,
    joint_mult: float = 1.0,
    segment_mults=None,
):
    # --- outputs --- #
    outdir = "sweep_videos"
    os.makedirs(outdir, exist_ok=True)
    video_path = os.path.join(outdir, f"{run_id}.mp4")

    print(f"\n==== {run_id} ====")
    print(f"Mode={stiffness_mode} | E={E:.2e} G={G:.2e} T={tension:.2f}")

    # --- sim params --- #
    final_time = 2.0
    time_step = 5e-6
    rendering_fps = 30.0
    total_steps = int(final_time / time_step)
    step_skip = int(1.0 / (rendering_fps * time_step))

    damping_constant = 5e-4

    # finger: 10cm length, 1cm diameter
    base_length = 0.10
    base_radius = 0.005
    n_elements = 80  # ~1.25mm per element
    density = 997.7 

    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    # cylinder
    cyl_radius = 0.01
    cyl_length = 0.20
    cyl_density = 1200.0
    cyl_start = np.array([0.02, -cyl_length / 2.0, -0.03])
    cyl_direction = np.array([0.0, 1.0, 0.0])
    cyl_normal = np.array([1.0, 0.0, 0.0])

    # contact params
    k_contact = 2e2
    nu_contact = 2.0
    mu_contact = 0.4
    vel_damp_contact = 5e1

    # tendon params
    vertebra_mass = 0.003
    num_vertebrae = 16
    first_vertebra_node = 3
    final_vertebra_node = n_elements - 3
    vertebra_height = 0.010
    vertebra_height_orientation = np.array([0.0, 0.0, -1.0])

    # create simulator
    sim = SquirrelFingerSimulator()

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

    # apply stiffness design
    if stiffness_mode == "joints":
        if joint_indices is None:
            raise ValueError("joint_indices required for stiffness_mode='joints'")
        apply_rigid_links_soft_joints(
            finger,
            rigid_mult=rigid_mult,
            joint_indices=joint_indices,
            joint_mult=joint_mult,
        )
    elif stiffness_mode == "segments":
        if segment_mults is None:
            raise ValueError("segment_mults required for stiffness_mode='segments'")
        apply_segmented_profile(finger, segment_mults=segment_mults)
    else:
        raise ValueError("stiffness_mode must be 'joints' or 'segments'")

    # cylinder
    cylinder = Cylinder(
        start=cyl_start,
        direction=cyl_direction,
        normal=cyl_normal,
        base_length=cyl_length,
        base_radius=cyl_radius,
        density=cyl_density,
    )
    sim.append(cylinder)

    # constraints
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

    # tendon
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

    # gravity
    sim.add_forcing_to(finger).using(GravityForces, np.array([0.0, 0.0, -9.80665]))

    # internal damping (material-ish damping)
    sim.dampen(finger).using(
        AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=time_step,
    )

    # contact
    sim.detect_contact_between(finger, cylinder).using(
        RodCylinderContact,
        k=k_contact,
        nu=nu_contact,
        velocity_damping_coefficient=vel_damp_contact,
        friction_coefficient=mu_contact,
    )

    # callback
    class CB(CallBackBaseClass):
        def __init__(self, step_skip, data):
            super().__init__()
            self.every = step_skip
            self.data = data

        def make_callback(self, system, time, current_step):
            if current_step % self.every == 0:
                self.data["pos"].append(system.position_collection.copy())
                if np.isnan(system.position_collection).any():
                    raise RuntimeError("NaN encountered")

    data = defaultdict(list)
    sim.collect_diagnostics(finger).using(CB, step_skip=step_skip, data=data)

    sim.finalize()

    # integrate
    timestepper = PositionVerlet()
    try:
        integrate(timestepper, sim, final_time, total_steps)
    except Exception as e:
        print(f"[FAIL] {run_id}: {e}")
        return

    pos_data = data["pos"]
    if len(pos_data) < 2:
        print(f"[FAIL] {run_id}: no saved frames")
        return

    # video render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    idx = 0

    def make_frame(t):
        nonlocal idx
        ax.clear()

        P = pos_data[idx]
        ax.scatter(P[0], P[1], P[2], s=6)

        # limits tuned for 10cm finger + cylinder
        ax.set_xlim(-0.02, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_zlim(-0.10, 0.10)

        # cylinder (fixed)
        center = cylinder.position_collection[:, 0]
        axis_dir = cylinder.director_collection[2, :, 0]
        draw_cylinder(ax, center, axis_dir, cyl_radius, cyl_length, color="black", alpha=0.35)

        # "front view": from -Y looking toward +Y
        ax.view_init(elev=0, azim=-90)

        idx = min(idx + 1, len(pos_data) - 1)
        return mplfig_to_npimage(fig)

    clip = VideoClip(make_frame, duration=final_time)
    clip.write_videofile(video_path, codec="libx264", fps=rendering_fps, logger=None)
    plt.close(fig)

    print(f"[OK] saved {video_path}")


if __name__ == "__main__":
    E_list = [3e5, 1e6, 3e6]              # Pa
    G_ratio_list = [0.3, 0.4]             # G = ratio * E (keep stable-ish)

    T_list = [0.5, 1.0, 2.0, 3.0]         # N

    # Start with 3 joints (4 links)
    rigid_mult_list = [1e2, 1e3]
    joint_mult_list = [1e-3, 1e-4]

    joint_sets = [
        [15, 30, 45],     # evenly spaced
        [10, 30, 55],     # shorter first link, longer last link
        [20, 35, 50],     # shifted toward tip
    ]

    segmented_profiles = [
        [1.0]*10,
        [5.0,1.0,5.0,1.0,5.0,1.0,5.0,1.0,5.0,1.0],   # alternating rigid/soft-ish
        [2.0,2.0,1.0,1.0,0.7,0.7,0.5,0.5,0.3,0.3],   # gradually softer toward tip
    ]

    mode = "joints" 

    for E in E_list:
        for r in G_ratio_list:
            G = r * E
            for T in T_list:
                if mode == "joints":
                    for rigid_mult in rigid_mult_list:
                        for joint_mult in joint_mult_list:
                            for joints in joint_sets:
                                run_id = f"modeJ_E{E:.1e}_G{G:.1e}_T{T:.2f}_R{rigid_mult:.0e}_J{joint_mult:.0e}_idx{'-'.join(map(str,joints))}"
                                run_one(
                                    run_id=run_id,
                                    E=E,
                                    G=G,
                                    tension=T,
                                    stiffness_mode="joints",
                                    rigid_mult=rigid_mult,
                                    joint_indices=joints,
                                    joint_mult=joint_mult,
                                )
                else:
                    for seg in segmented_profiles:
                        run_id = f"modeS_E{E:.1e}_G{G:.1e}_T{T:.2f}_seg{'-'.join([str(x) for x in seg])}"
                        run_one(
                            run_id=run_id,
                            E=E,
                            G=G,
                            tension=T,
                            stiffness_mode="segments",
                            segment_mults=seg,
                        )