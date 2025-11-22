"""
Squirrel finger: 1D tendon-driven rod with non-uniform stiffness
perching on a rigid cylinder, visualized with matplotlib + moviepy.
"""

from elastica.modules import (
    BaseSystemCollection,
    Connections,
    Constraints,
    Forcing,
    CallBacks,
    Damping,
    Contact
)
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC
from elastica.boundary_conditions import FixedConstraint
from elastica.external_forces import GravityForces
from TendonForces import TendonForces # install using Git, not manual installation options
from elastica.dissipation import AnalyticalLinearDamper
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
from elastica.rigidbody.cylinder import Cylinder
from elastica.contact_forces import RodCylinderContact

import numpy as np
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sys

import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

plt.switch_backend("TkAgg")
plt.close("all")


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


sim = SquirrelFingerSimulator()

# Helper function for drawing cylinder
def draw_cylinder(ax, center, axis_dir, radius, length, color="gray", alpha=0.3, resolution=40):

    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # find two perpendicular vectors to axis_dir
    if np.allclose(axis_dir, [0, 0, 1]):
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.cross(axis_dir, [0, 0, 1])
        v /= np.linalg.norm(v)

    w = np.cross(axis_dir, v)

    # parametric grid
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(-length / 2, length / 2, 20)

    theta, z = np.meshgrid(theta, z)

    X = center[0] + axis_dir[0] * z + radius * (v[0] * np.cos(theta) + w[0] * np.sin(theta))
    Y = center[1] + axis_dir[1] * z + radius * (v[1] * np.cos(theta) + w[1] * np.sin(theta))
    Z = center[2] + axis_dir[2] * z + radius * (v[2] * np.cos(theta) + w[2] * np.sin(theta))

    ax.plot_surface(
        X, Y, Z,
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=True
    )

# Simulation parameters
final_time = 2.0
time_step = 1.8e-5
rendering_fps = 30.0
total_steps = int(final_time / time_step)
step_skip = int(1.0 / (rendering_fps * time_step))

# Create rod (finger)
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
base_length = 0.3
n_elements = 100
base_radius = 0.011/2
density = 997.7
youngs_modulus = 3e5 # 16.598637e6
shear_modulus = 1.2e5 # 7.216880e6

dtmax = (base_length / n_elements) * np.sqrt(
        density / max(youngs_modulus, shear_modulus))
print("Maximum time_step magnitude: ", dtmax)

finger = CosseratRod.straight_rod(
    n_elements=n_elements,
    start=np.array([0.0, 0.0, 0.0]),
    direction=direction,
    normal=normal,
    base_length=base_length,
    base_radius=base_radius,
    density=density,
    youngs_modulus=youngs_modulus,
    shear_modulus=shear_modulus,
)

# Add rod to simulator
sim.append(finger)

# Non-uniform stiffness profile (more flexible toward tip)
eps = 0.5
# profile = np.linspace(1.0 + eps, 1.0 - eps, n_elements - 1)
s = np.linspace(0, 1, n_elements - 1)
profile = 1.0 + eps * (1.0 - s**2)

for i in range(n_elements - 1):
    finger.bend_matrix[1, 1, i] *= profile[i]
    finger.bend_matrix[2, 2, i] *= profile[i]

# Create cylinder
cyl_radius = 0.03
cyl_length = 0.4
cyl_density = 1200.0
# cyl_start = np.array([0.10, -cyl_length / 2.0, -cyl_radius])
cyl_start = np.array([0.10, -cyl_length / 2.0, -0.08])
cyl_direction = np.array([0.0, 1.0, 0.0])
cyl_normal = np.array([1.0, 0.0, 0.0])

cylinder = Cylinder(
    start=cyl_start,
    direction=cyl_direction,
    normal=cyl_normal,
    base_length=cyl_length,
    base_radius=cyl_radius,
    density=cyl_density,
)

sim.append(cylinder)

# Constrain rod
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

# Tendon on dorsal side, curling around cylinder
sim.add_forcing_to(finger).using(
    TendonForces,
    vertebra_height=0.010, 
    num_vertebrae=30,
    first_vertebra_node=3,
    final_vertebra_node=n_elements - 3,
    vertebra_mass=0.01,
    tension=0.3, # TODO: tuning
    vertebra_height_orientation=np.array([0.0, 0.0, -1.0]),
    n_elements=n_elements,
)

gravity_magnitude = -9.80665
acc_gravity = np.zeros(3)
acc_gravity[2] = gravity_magnitude

sim.add_forcing_to(finger).using(
    GravityForces,
    acc_gravity=acc_gravity,
)

sim.dampen(finger).using(
    AnalyticalLinearDamper,
    damping_constant=0.001, # TODO: tuning
    time_step=time_step,
)

# Rod-cylinder contact
contact = sim.detect_contact_between(finger, cylinder).using(
    RodCylinderContact,
    k=1e2,
    nu=10,
    velocity_damping_coefficient=5e2,
    friction_coefficient=0.3,
)

# MyCallBack class is derived from the base call back class.
class MyCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            for i in range(len(system.position_collection)):
                for j in range(len(system.position_collection[i])):
                    if np.isnan(system.position_collection[i][j]):
                        print(
                            "NAN VALUE ENCOUNTERED at position: ",
                            (i, j),
                            "AT TIME: ",
                            time,
                        )
                        sys.exit()
            return

# Create dictionary to hold data from callback function
callback_data_finger = defaultdict(list)

# Add MyCallBack to SystemSimulator for each rod telling it how often to save data (step_skip)
sim.collect_diagnostics(finger).using(
    MyCallBack, step_skip=step_skip, callback_params=callback_data_finger
)

sim.finalize()


timestepper = PositionVerlet()
integrate(timestepper, sim, final_time, total_steps)

position_data = callback_data_finger["position"]
directors_data = callback_data_finger["directors"]


# Creating a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# initialize counter
count = 0

# method to get frames
def make_frame(t):

    global count

    # clear
    ax.clear()
     
    # Scatter plot
    ax.scatter(position_data[count][0], position_data[count][1], position_data[count][2])
    ax.axes.set_zlim3d(bottom=-base_length,top=base_length)
    ax.axes.set_ylim3d(bottom=-base_length,top=base_length)
    ax.axes.set_xlim(-base_length/2,base_length)

    # Labeling axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    calculated_x = position_data[count][0][-1]
    calculated_y = position_data[count][1][-1]
    calculated_z = position_data[count][2][-1]

    x_point = round(calculated_x,3)
    y_point = round(calculated_y,3)
    z_point = round(calculated_z,3)
    annotation_text = f'Tip Pos: ({x_point}, {y_point}, {z_point})'
    ax.text(x_point, y_point, z_point, annotation_text, color='red')

    original_directors = directors_data[count][...,0]
    
    num_vertebrae = 6
    first_vertebra_node = 2
    final_vertebra_node = 98
    vertebra_nodes = []
    vertebra_increment = (final_vertebra_node - first_vertebra_node)/(num_vertebrae - 1)
    for i in range(num_vertebrae):
        vertebra_nodes.append(round(i * vertebra_increment + first_vertebra_node))

    for node in vertebra_nodes:
        local_directors = directors_data[count][...,node]
        vertebra_pos = np.array([position_data[count][0][node], position_data[count][1][node], position_data[count][2][node]])
        vertebra_coord_syst_x = np.array([local_directors[0][0], local_directors[0][1], local_directors[0][2]])
        vertebra_coord_syst_y = np.array([local_directors[1][0], local_directors[1][1], local_directors[1][2]])
        vertebra_coord_syst_z = np.array([local_directors[2][0], local_directors[2][1], local_directors[2][2]])

        # Scale the arrows for better visibility
        if count > 1:
            scale = 0.1
        else:
            scale = 0.03

        # Plot the arrows using quiver
        ax.quiver(vertebra_pos[0], vertebra_pos[1], vertebra_pos[2],
            vertebra_coord_syst_x[0], vertebra_coord_syst_x[1], vertebra_coord_syst_x[2],
            length=scale, color='r', normalize=True)
        ax.quiver(vertebra_pos[0], vertebra_pos[1], vertebra_pos[2],
            vertebra_coord_syst_y[0], vertebra_coord_syst_y[1], vertebra_coord_syst_y[2],
            length=scale, color='g', normalize=True)
        ax.quiver(vertebra_pos[0], vertebra_pos[1], vertebra_pos[2],
            vertebra_coord_syst_z[0], vertebra_coord_syst_z[1], vertebra_coord_syst_z[2],
            length=scale, color='b', normalize=True)

    # Update counter
    count=count+1
    center = cylinder.position_collection[:, 0]
    axis_dir = cylinder.director_collection[2, :, 0]

    # Draw cylinder
    draw_cylinder(
        ax,
        center=center,
        axis_dir=axis_dir,
        radius=cyl_radius,
        length=cyl_length,
        color="black",
        alpha=0.4,
    )
    ax.view_init(elev=-4, azim=-82) # set view angle for video

    # Update counter (don’t run past the last saved frame)
    count = min(count + 1, len(position_data) - 1)
    # returning numpy imagedef make_frame(t):
    return mplfig_to_npimage(fig)

# creating animation
clip = VideoClip(make_frame, duration=final_time)

# displaying animation with auto play and looping
clip.write_videofile("squirrel_finger_perching.mp4", codec="libx264", fps=rendering_fps)


# Compute contact forces and evaluations
def compute_contact_forces_frame(rod_pos, cyl_center, cyl_axis, cyl_radius, k):
    """
    rod_pos   : (3, n_nodes) array (e.g. position_data[frame])
    cyl_center: (3,) cylinder axis center
    cyl_axis  : (3,) unit vector along cylinder axis
    cyl_radius: float
    k         : normal stiffness used in RodCylinderContact

    Returns:
        indices: list of node indices in contact
        forces:  list of normal force magnitudes at those nodes
    """
    cyl_axis = cyl_axis / np.linalg.norm(cyl_axis)
    rel = rod_pos.T - cyl_center[None, :] # (n_nodes, 3)
    proj_len = np.dot(rel, cyl_axis) # scalar projection along axis
    proj = np.outer(proj_len, cyl_axis) # axis component
    radial = rel - proj # radial vector
    radial_dist = np.linalg.norm(radial, axis=1)

    overlap = cyl_radius - radial_dist # >0 means penetration
    mask = overlap > 0.0

    contact_indices = np.where(mask)[0].tolist()
    normal_forces = (k * overlap[mask]).tolist()

    return contact_indices, normal_forces

# geometry at the end (cylinder is rigid, so one frame is enough)
cyl_center = cylinder.position_collection[:, 0]
cyl_axis   = cylinder.director_collection[2, :, 0]  # axis direction
k_contact  = 1e2 

cyl_center = cylinder.position_collection[:, 0]
cyl_axis   = cylinder.director_collection[2, :, 0]
k_contact  = 1e2  # same as in RodCylinderContact

first_contact_frame = None
max_force_overall = 0.0

for frame_idx, rod_pos in enumerate(position_data):
    indices, forces = compute_contact_forces_frame(
        rod_pos, cyl_center, cyl_axis, cyl_radius, k_contact
    )

    if forces:
        frame_max = max(forces)
        if frame_max > max_force_overall:
            max_force_overall = frame_max

        if first_contact_frame is None:
            first_contact_frame = frame_idx
            print(f"First contact at frame {frame_idx}, nodes={indices[:5]}, "
                  f"max normal force in this frame={frame_max:.3f}")
min_dist_overall = np.inf

for frame_idx, rod_pos in enumerate(position_data):
    cyl_axis = cyl_axis / np.linalg.norm(cyl_axis)
    rel = rod_pos.T - cyl_center[None, :]
    proj_len = np.dot(rel, cyl_axis)
    proj = np.outer(proj_len, cyl_axis)
    radial = rel - proj
    radial_dist = np.linalg.norm(radial, axis=1)

    frame_min = np.min(radial_dist)
    if frame_min < min_dist_overall:
        min_dist_overall = frame_min

print("Minimum rod-to-cylinder axis distance over trajectory:", min_dist_overall)
print("Cylinder radius:", cyl_radius)
if first_contact_frame is None:
    print("No contact in ANY saved frame.")
else:
    print(f"Max normal force over all frames: {max_force_overall:.3f}")

print("Contact nodes:", indices)
print("Contact normal force magnitudes:", forces)
print("Max normal force:", max(forces) if forces else 0.0)


# Static snapshot of final frame
plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_title(
    f"Squirrel finger perching\nE_base = {youngs_modulus*1e-6:.2f} MPa",
    fontsize=16,
)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")

pos_final = position_data[-1]
ax.scatter(pos_final[0], pos_final[1], pos_final[2], s=8)

# tighten view around the action
ax.set_zlim(-base_length * 0.5, base_length * 0.5)
ax.set_ylim(-0.15, 0.15)
ax.set_xlim(-0.01, base_length)

# tip marker + text
tip_x = pos_final[0][-1]
tip_y = pos_final[1][-1]
tip_z = pos_final[2][-1]
ax.text(
    tip_x,
    tip_y,
    tip_z,
    f"Tip: ({tip_x:.3f}, {tip_y:.3f}, {tip_z:.3f})",
    color="red",
    fontsize=12,
)

# draw cylinder like in the video
center = cylinder.position_collection[:, 0]
axis_dir = cylinder.director_collection[2, :, 0]

draw_cylinder(
    ax,
    center=center,
    axis_dir=axis_dir,
    radius=cyl_radius,
    length=cyl_length,
    color="black",
    alpha=0.4,
)

try:
    ax.set_box_aspect([1, 1, 1])
except Exception:
    pass

ax.view_init(elev=30, azim=30)
plt.tight_layout()
plt.savefig("squirrel_finger_final_frame.png", dpi=200)
plt.show()