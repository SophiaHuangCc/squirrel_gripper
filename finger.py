"""
single_finger_manual.py

Single-run version of your sweep script (no sweeping).
You manually tune:
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


def run_one_manual():

    outdir = "manual_videos"
    os.makedirs(outdir, exist_ok=True)
    video_path = os.path.join(outdir, "manual_run.mp4")

    # Material (Pa)
    E = 3.0e5
    G = 1.2e5

    # Tendon
    tension = 2.0 # N

    # Finger geometry (10cm long, 1cm diameter)
    base_length = 0.10
    base_radius = 0.005
    n_elements = 80 # ~1.25mm per element
    density = 997.7

    # Joint model (3 joints / 4 links)
    rigid_mult = 1e2 # links stiffness multiplier
    joint_indices = [20, 50, 80]
    joint_mult = 1e-4 # joints relative to links (smaller = softer joints)

    # Damping (internal)
    damping_constant = 0

    # Cylinder geometry/pose
    cyl_radius = 0.01
    cyl_length = 0.20
    cyl_density = 1200.0

    cyl_start = np.array([0.02, -cyl_length / 2.0, -0.02])

    cyl_direction = np.array([0.0, 1.0, 0.0])
    cyl_normal = np.array([1.0, 0.0, 0.0])

    # Contact params
    k_contact = 2e2
    nu_contact = 2.0
    mu_contact = 0.8 # 0.4
    vel_damp_contact = 0 # 5e1

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
                if np.isnan(system.position_collection).any():
                    raise RuntimeError("NaN encountered")

    data = defaultdict(list)
    sim.collect_diagnostics(finger).using(CB, step_skip=step_skip, data=data)

    sim.finalize()

    timestepper = PositionVerlet()
    integrate(timestepper, sim, final_time, total_steps)

    pos_data = data["pos"]
    if len(pos_data) < 2:
        raise RuntimeError("No saved frames")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    idx = 0

    def make_frame(t):
        nonlocal idx
        ax.clear()

        P = pos_data[idx]
        ax.scatter(P[0], P[1], P[2], s=6)

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