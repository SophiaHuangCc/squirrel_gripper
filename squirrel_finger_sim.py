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
import csv

from scipy.spatial import ConvexHull, Delaunay

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

# Compute contact forces and evaluations
def compute_contact_metrics_frame(
    rod_pos,
    rod_vel,
    cyl_center,
    cyl_axis,
    cyl_radius,
    k,
    mu,
):
    """
    rod_pos   : (3, n_nodes)
    rod_vel   : (3, n_nodes)
    cyl_center: (3,)
    cyl_axis  : (3,)
    cyl_radius: float
    k         : normal stiffness (same as RodCylinderContact.k)
    mu        : friction coefficient (same as RodCylinderContact.friction_coefficient)

    Returns:
        indices          : node indices in contact
        normal_forces    : normal force magnitudes (k * penetration)
        normal_vel       : normal component of velocity (m/s)
        tangential_speed : |v_t| at node (m/s)
        friction_forces  : Coulomb friction magnitudes (mu * normal_force)
    """

    cyl_axis = cyl_axis / np.linalg.norm(cyl_axis)

    # N x 3 arrays
    rel = rod_pos.T - cyl_center[None, :] 
    proj_len = np.dot(rel, cyl_axis)
    proj = np.outer(proj_len, cyl_axis)
    radial = rel - proj
    radial_dist = np.linalg.norm(radial, axis=1)

    eps = 1e-12
    normal_vec = np.zeros_like(radial)
    mask_nonzero = radial_dist > eps
    normal_vec[mask_nonzero] = radial[mask_nonzero] / radial_dist[mask_nonzero, None]

    # penetration (penalty model)
    overlap = cyl_radius - radial_dist
    contact_mask = overlap > 0.0

    normal_force_mag = k * np.clip(overlap, a_min=0.0, a_max=None)

    # velocities
    vel = rod_vel.T 
    normal_vel = np.sum(vel * normal_vec, axis=1) 
    vel_t = vel - normal_vel[:, None] * normal_vec
    tangential_speed = np.linalg.norm(vel_t, axis=1)

    friction_force_mag = mu * normal_force_mag

    # only keep contacting nodes
    idx = np.where(contact_mask)[0]

    return (
        idx,
        normal_force_mag[idx],
        normal_vel[idx],
        tangential_speed[idx],
        friction_force_mag[idx],
    )

def origin_in_hull(points):
    """
    Returns True if the origin [0,0,0] is inside the convex hull of 'points'.
    points: (N,3) array of force vectors
    """
    try:
        hull = ConvexHull(points) # Will error with degenerate points
        delaunay = Delaunay(points[hull.vertices])
        return delaunay.find_simplex([0,0,0]) >= 0
    except:
        return False

def compute_grasp_metrics(F_array, pos_contact, cyl_center, cyl_direction, n_perturb=20, sigma_pos=1e-4):
    """
    Compute grasp metrics for a single frame.
    
    Args:
        F_array       : (n_nodes, 3) total force vectors at contacting nodes
        pos_contact   : (3, n_nodes) positions of contacting nodes
        cyl_center    : (3,) center of cylinder
        cyl_direction : (3,) cylinder axis (unit vector)
        n_perturb     : number of perturbation samples for robust FC
        sigma_pos     : standard deviation of contact position perturbation (m)
        
    Returns:
        FC_boolean    : True if FC_metric > 0
        FC_metric     : Ferrari-Canny metric (min distance from hull to origin)
        RFC_score     : robust force closure score in [0,1]
    """
    n_nodes = F_array.shape[0]
    if n_nodes < 3:
        return False, 0, 0
    
    # Force closure boolean
    force_closure = origin_in_hull(F_array)
    
    # Ferrari Canny
    try:
        hull = ConvexHull(F_array)
        hull_vertices = F_array[hull.vertices]
        ferrari_canny = np.min(np.linalg.norm(hull_vertices, axis=1))
    except:
        ferrari_canny = 0
    
    # Robust force closure
    success_count = 0
    for _ in range(n_perturb):
        perturb = np.random.normal(0, sigma_pos, size=(n_nodes, 3))
        F_array_pert = F_array + perturb
        success_count += origin_in_hull(F_array_pert)
    robust_force_closure = success_count / n_perturb

    # Max torque
    tau_net = np.zeros(3)
    for i in range(n_nodes):
        r_i = pos_contact[:, i] - cyl_center
        f_i = F_array[i]
        tau_net += np.cross(r_i, f_i)
    max_torque = np.linalg.norm(tau_net)

    # Angular fraction
    angular_frac = 0
    # print(pos_contact)
    # cyl_axis_unit = cyl_direction / np.linalg.norm(cyl_direction)
    # vec = pos_contact - cyl_center[:, None]
    # vec_proj = vec - cyl_axis_unit[:, None] * np.sum(vec * cyl_axis_unit[:, None], axis=0)
    # angles = np.arctan2(vec_proj[1], vec_proj[0])
    # angles = np.mod(angles, 2 * np.pi)
    # angles_sorted = np.sort(angles)
    # # print(angles_sorted)
    # gaps = np.diff(np.concatenate([angles_sorted, [angles_sorted[0] + 2 * np.pi]]))
    # max_gap = np.max(gaps)
    # angular_frac = 1.0 - max_gap / (2 * np.pi)
    
    return force_closure, ferrari_canny, robust_force_closure, max_torque, angular_frac

def compute_total_energy(finger, n_forces, k):
    """
    Compute total potential energy for a single frame.

    Args:
        finger     : CosseratRod object
        n_forces   : normal contact forces at those nodes
        k  : contact stiffness

    Returns:
        U_total    : total potential energy (J)
    """
    # Approx elastic energy
    U_elastic = 0.0
    for i in range(finger.n_elems - 1):
        d_i   = finger.director_collection[:, 2, i]
        d_ip1 = finger.director_collection[:, 2, i+1]
        ds = finger.rest_lengths[i]           # element length
        kappa = (d_ip1 - d_i) / ds            # curvature approximation
        U_elastic += 0.5 * np.dot(kappa, finger.bend_matrix[..., i] @ kappa)

    # Contact energy
    if len(n_forces) > 0:
        U_contact = np.sum(0.5 * n_forces**2 / k)
    else:
        U_contact = 0.0

    U_total = U_elastic + U_contact
    return U_total

#####################################
## Simulation parameters and setup
#####################################
# Simulation parameters
final_time = 2.0
time_step = 1.8e-5 # 1.8e-5 original, 5e-6 for stability with joints
rendering_fps = 30.0
total_steps = int(final_time / time_step)
step_skip = int(1.0 / (rendering_fps * time_step))

# Create rod (finger)
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
base_length = 0.1 # in meters for finger
n_elements = 80
base_radius = 0.005
density = 997.7
youngs_modulus = 3e5 # 3e5 original, E
shear_modulus = 1.2e5 # 1.2e5 original, G < E
tension = 0.5 # tendon tension force in Newtons
eps = 0.3 # % softer at the tip
damping_constant = 5e-4 # original 0.002, 0 means no internal damping
k = 1e1
nu = 1 # velocity damping coefficient
mu = 0.5 # friction coefficient
velocity_damping_coefficient = 1e1
vertebra_mass = 0.01 # mass of each vertebra in kg, almost negligible to start with

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
# 1) non-uniform alternative 1
# profile = np.linspace(1.0 + eps, 1.0 - eps, n_elements - 1)
# 2) non-uniform alternative 2
# s = np.linspace(0, 1, n_elements - 1)
# profile = 1.0 + eps * (1.0 - s**2)
# 3) non-uniform alternative 3
# s = np.linspace(0, 1, n_elements - 1)
# profile = 1.0 - eps * (s**2)        # base: 1.0, tip: 0.5

# for i in range(n_elements - 1):
#     finger.bend_matrix[1, 1, i] *= profile[i]
#     finger.bend_matrix[2, 2, i] *= profile[i]
finger.bend_matrix[1, 1, :] *= 1e3
finger.bend_matrix[2, 2, :] *= 1e3

# 4) 3 joints
joint_indices = [
    int(0.25 * (n_elements - 1)),
    int(0.50 * (n_elements - 1)),
    int(0.75 * (n_elements - 1)),
]
# Make joints soft
joint_soft_factor = 1e-6  # smaller => softer joint (start here)
for j in joint_indices:
    finger.bend_matrix[1, 1, j] *= joint_soft_factor
    finger.bend_matrix[2, 2, j] *= joint_soft_factor

# Create cylinder
cyl_radius = 0.03
cyl_length = 0.4
cyl_density = 1200.0
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
    # first_vertebra_node=3,
    # final_vertebra_node=n_elements - 3,
    first_vertebra_node=joint_indices[0],
    final_vertebra_node=joint_indices[-1],
    vertebra_mass=vertebra_mass,
    tension=tension,
    vertebra_height_orientation=np.array([0.0, 0.0, -1.0]),
    n_elements=n_elements,
)

gravity_magnitude = -9.80665
acc_gravity = np.zeros(3)
acc_gravity[2] = gravity_magnitude

# Gravity forces
sim.add_forcing_to(finger).using(
    GravityForces, acc_gravity
)

sim.dampen(finger).using(
    AnalyticalLinearDamper,
    damping_constant=damping_constant, 
    time_step=time_step,
)

# Rod-cylinder contact
contact = sim.detect_contact_between(finger, cylinder).using(
    RodCylinderContact,
    k=k, 
    nu=nu,
    velocity_damping_coefficient=velocity_damping_coefficient, 
    friction_coefficient=mu,
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
            self.callback_params["forces"].append(system.external_forces.copy()) # forces
            self.callback_params["velocity"].append(system.velocity_collection.copy())
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
forces_data    = callback_data_finger["forces"]  # forces
velocity_data  = callback_data_finger["velocity"]

# Cylinder info
cyl_center = cylinder.position_collection[:, 0]

grasp_metrics = []

for frame_idx, (pos, vel) in enumerate(zip(position_data, velocity_data)):
    t = frame_idx * step_skip * time_step

    indices, n_forces, n_vel, t_speed, f_forces = compute_contact_metrics_frame(
        pos, vel, cyl_center, cyl_direction, cyl_radius, k, mu
    )

    # Force array around cylinder
    F_array = np.zeros((len(indices), 3))
    for j, idx in enumerate(indices):
        radial = pos[:, idx] - cyl_center
        radial_norm = np.linalg.norm(radial) + 1e-12
        F_array[j] = n_forces[j] * (radial / radial_norm)
        
        # Add friction
        tan = np.array([-radial[1], radial[0], 0.0])
        tan /= np.linalg.norm(t) + 1e-12
        F_array[j] += mu * n_forces[j] * tan

    metrics = compute_grasp_metrics(F_array, pos[:, indices], cyl_center, cyl_direction) if len(indices) != 0 else (False, 0, 0, 0, 0)

    # Calculate potential energy
    finger.position_collection[:] = pos
    finger.velocity_collection[:] = vel
    U_total = compute_total_energy(finger, n_forces, k)

    grasp_metrics.append([frame_idx, t, len(indices), *metrics, U_total])

with open("grasp_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Time", "Contact Sim Points", "Force Closure", "Ferrari-Canny", "Robust Force Closure", "Max Torque", "Angular Fraction", "Potential Energy"])
    for row in grasp_metrics:
        writer.writerow(row)


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
    final_vertebra_node = n_elements - 2
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
    # count=count+1
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

    forces = forces_data[count]
    pos     = position_data[count]

    # magnitude for each node
    mag = np.linalg.norm(forces, axis=0)
    # print("Max tendon force magnitude:", np.max(mag))

    force_scale = 0.2
    step_nodes  = 4

    for i in range(0, forces.shape[1], step_nodes):
        if mag[i] < 1e-6:
            continue  # skip almost-zero forces

        x, y, z = pos[0, i], pos[1, i], pos[2, i]
        fx, fy, fz = forces[0, i], forces[1, i], forces[2, i]

        # optional normalization so direction is visible, length scaled by magnitude
        ax.quiver(
            x, y, z,
            fx, fy, fz,
            length=force_scale,
            normalize=True,
            color="magenta",  # tendon color
        )

    # Update counter (don’t run past the last saved frame)
    count = min(count + 1, len(position_data) - 1)
    # returning numpy imagedef make_frame(t):
    return mplfig_to_npimage(fig)

# creating animation
clip = VideoClip(make_frame, duration=final_time)

# displaying animation with auto play and looping
clip.write_videofile("squirrel_finger_perching.mp4", codec="libx264", fps=rendering_fps)

###################################
# contact logging to CSV
###################################
cyl_center = cylinder.position_collection[:, 0]
cyl_axis   = cylinder.director_collection[2, :, 0]
k_contact  = k
mu_contact = mu

dt_saved = step_skip * time_step

first_contact_frame = None
max_normal_force_overall = 0.0

with open("contact_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "frame_idx",
            "time",
            "node_idx",
            "normal_force",
            "normal_velocity",
            "tangential_speed",
            "friction_force",
        ]
    )

    for frame_idx, (pos, vel) in enumerate(zip(position_data, velocity_data)):
        t = frame_idx * dt_saved

        (
            indices,
            n_forces,
            n_vel,
            t_speed,
            f_forces,
        ) = compute_contact_metrics_frame(
            pos,
            vel,
            cyl_center,
            cyl_axis,
            cyl_radius,
            k_contact,
            mu_contact,
        )

        if len(indices) > 0:
            frame_max = np.max(n_forces)
            if frame_max > max_normal_force_overall:
                max_normal_force_overall = frame_max

            if first_contact_frame is None:
                first_contact_frame = frame_idx

        # write one row per contacting node
        for j, node_idx in enumerate(indices):
            writer.writerow(
                [
                    frame_idx,
                    t,
                    int(node_idx),
                    float(n_forces[j]),
                    float(n_vel[j]),
                    float(t_speed[j]),
                    float(f_forces[j]),
                ]
            )

if first_contact_frame is None:
    print("No contact in ANY saved frame.")
else:
    print(f"First contact at frame {first_contact_frame}, "
          f"t = {first_contact_frame * dt_saved:.4f} s")
    print(f"Max normal force over all frames: {max_normal_force_overall:.3f} N")
    print("Contact log written to contact_log.csv")


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