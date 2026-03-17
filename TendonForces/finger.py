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
    try:
        plt.switch_backend("TkAgg")
        print("[DEBUG] Matplotlib interactive backend enabled.")
    except ImportError:
        plt.switch_backend("Agg")
        print("[DEBUG] TkAgg unavailable in headless mode; falling back to Agg backend.")
else:
    plt.switch_backend("Agg")

# MoviePy for video rendering
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

# PyElastica imports
from elastica.modules import BaseSystemCollection, Connections, Constraints, Forcing, CallBacks, Damping, Contact
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC, FixedConstraint, ConstraintBase
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


def compute_contact_force_vectors(
    rod_pos,
    rod_vel,
    cyl_center,
    cyl_radius,
    k,
    mu,
    base_radius,
    friction_velocity_override=None,
):
    """
    Compute per-contact force vectors from geometry + Coulomb friction.
    friction_velocity_override: optional 3-vector used to define disturbance slip direction.
    """
    dx = rod_pos[0, :] - cyl_center[0]
    dz = rod_pos[2, :] - cyl_center[2]
    radial_dist = np.sqrt(dx**2 + dz**2)
    contact_threshold = cyl_radius + base_radius
    overlaps = contact_threshold - radial_dist
    contact_mask = overlaps > 0.0

    normal_vec = np.zeros((rod_pos.shape[1], 3))
    normal_vec[:, 0] = dx
    normal_vec[:, 2] = dz
    norms = np.linalg.norm(normal_vec, axis=1)
    mask = norms > 1e-12
    normal_vec[mask] /= norms[mask, None]

    normal_force_mag = k * np.where(contact_mask, overlaps, 0.0)

    if friction_velocity_override is None:
        vel = rod_vel.T
    else:
        v_override = np.asarray(friction_velocity_override, dtype=float).reshape(1, 3)
        vel = np.repeat(v_override, rod_pos.shape[1], axis=0)

    normal_vel = np.sum(vel * normal_vec, axis=1)
    vel_t = vel - normal_vel[:, None] * normal_vec
    tangential_speed = np.linalg.norm(vel_t, axis=1)

    friction_dir = np.zeros_like(vel_t)
    tan_mask = tangential_speed > 1e-12
    friction_dir[tan_mask] = -vel_t[tan_mask] / tangential_speed[tan_mask][:, None]
    friction_force_mag = mu * normal_force_mag

    f_n = normal_force_mag[:, None] * normal_vec
    f_t = friction_force_mag[:, None] * friction_dir
    f_total = f_n + f_t

    idx = np.where(contact_mask)[0]
    return (
        idx,
        f_total[idx],
        f_n[idx],
        f_t[idx],
        normal_force_mag[idx],
        overlaps[idx],
    )


def evaluate_disturbance_contact_response(
    rod_pos,
    rod_vel,
    cyl_center,
    cyl_radius,
    k,
    mu,
    base_radius,
    base_indices,
    node_masses,
    disturbance_force,
    n_steps,
    dt,
):
    """
    Apply a disturbance pulse for a few explicit steps and measure resulting contact wrench.
    This is a lightweight post-process rollout (not a full Cosserat re-simulation).
    """
    disturbance_force = np.asarray(disturbance_force, dtype=float).reshape(3)
    pos = np.asarray(rod_pos, dtype=float).copy()
    vel = np.asarray(rod_vel, dtype=float).copy()
    base_indices = np.asarray(base_indices, dtype=int)
    if base_indices.size == 0:
        raise ValueError("base_indices must have at least one node index")

    node_masses = np.asarray(node_masses, dtype=float).reshape(-1)
    if node_masses.size != pos.shape[1]:
        raise ValueError("node_masses length must equal number of rod nodes")
    node_masses = np.maximum(node_masses, 1e-9)

    dt = max(float(dt), 1e-8)
    n_steps = max(int(n_steps), 1)
    base_force_per_node = disturbance_force / float(base_indices.size)

    idx = np.array([], dtype=int)
    contact_forces = np.zeros((0, 3))
    normal_force_mag = np.zeros((0,), dtype=float)
    net_contact_force = np.zeros(3)
    net_contact_torque = np.zeros(3)
    base_point = pos[:, base_indices].mean(axis=1)

    force_hist = []
    torque_hist = []
    contacts_hist = []

    for _ in range(n_steps):
        (
            idx,
            contact_forces,
            _,
            _,
            normal_force_mag,
            _,
        ) = compute_contact_force_vectors(
            rod_pos=pos,
            rod_vel=vel,
            cyl_center=cyl_center,
            cyl_radius=cyl_radius,
            k=k,
            mu=mu,
            base_radius=base_radius,
        )

        nodal_forces = np.zeros_like(pos)
        if len(idx) > 0:
            nodal_forces[:, idx] += contact_forces.T
        nodal_forces[:, base_indices] += base_force_per_node.reshape(3, 1)

        acc = nodal_forces / node_masses.reshape(1, -1)
        vel = vel + dt * acc
        pos = pos + dt * vel

        base_point = pos[:, base_indices].mean(axis=1)
        if len(idx) > 0:
            cp = pos[:, idx].T
            net_contact_force = np.sum(contact_forces, axis=0)
            net_contact_torque = np.sum(np.cross(cp - base_point[None, :], contact_forces), axis=0)
        else:
            net_contact_force = np.zeros(3)
            net_contact_torque = np.zeros(3)

        force_hist.append(net_contact_force.copy())
        torque_hist.append(net_contact_torque.copy())
        contacts_hist.append(int(len(idx)))

    applied_torque = np.zeros(3)
    fmag = np.linalg.norm(disturbance_force)
    f_dir = disturbance_force / fmag if fmag > 1e-12 else np.zeros(3)
    resist_force = -float(np.dot(net_contact_force, f_dir)) if fmag > 1e-12 else 0.0
    resist_torque = 0.0

    force_alignment = 0.0
    if fmag > 1e-12 and np.linalg.norm(net_contact_force) > 1e-12:
        force_alignment = float(
            np.dot(net_contact_force, disturbance_force)
            / (np.linalg.norm(net_contact_force) * fmag)
        )
    torque_alignment = 0.0

    return {
        "num_contacts": int(len(idx)),
        "applied_force": disturbance_force,
        "applied_torque": applied_torque,
        "net_contact_force": net_contact_force,
        "net_contact_torque": net_contact_torque,
        "resist_force": resist_force,
        "resist_torque": resist_torque,
        "force_alignment": force_alignment,
        "torque_alignment": torque_alignment,
        "force_resisted": bool(resist_force >= fmag and fmag > 0.0),
        "torque_resisted": False,
        "total_normal_force": float(np.sum(normal_force_mag)) if len(idx) else 0.0,
        "contact_positions": pos[:, idx].copy() if len(idx) else np.zeros((3, 0)),
        "contact_forces": contact_forces.copy(),
        "position_final": pos.copy(),
        "velocity_final": vel.copy(),
        "base_point_final": base_point.copy(),
        "force_history": np.array(force_hist),
        "torque_history": np.array(torque_hist),
        "contacts_history": np.array(contacts_hist, dtype=int),
    }


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


class ConstantNodeForce(NoForces):
    def __init__(self, force_vector, node_indices):
        self.force_vector = np.asarray(force_vector, dtype=float)
        self.node_indices = np.asarray(node_indices, dtype=int)

    def apply_forces(self, system, time=0.0):
        if self.node_indices.size == 0:
            return
        force_per_node = self.force_vector / float(self.node_indices.size)
        for idx in self.node_indices:
            system.external_forces[:, idx] += force_per_node


class BaseAxisSpringDamper(NoForces):
    """
    Apply a spring-damper force on a base node along selected axes.
    Useful for preventing unbounded drift in free force-driven runs.
    """

    def __init__(
        self,
        node_idx,
        target_position,
        stiffness,
        damping,
        active_axes=(1.0, 1.0, 0.0),
        max_force=5.0,
    ):
        self.node_idx = int(node_idx)
        self.target_position = np.asarray(target_position, dtype=float).reshape(3)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.active_axes = np.asarray(active_axes, dtype=float).reshape(3)
        self.max_force = max(0.0, float(max_force))

    def apply_forces(self, system, time=0.0):
        x = system.position_collection[:, self.node_idx]
        v = system.velocity_collection[:, self.node_idx]
        dx = (x - self.target_position) * self.active_axes
        dv = v * self.active_axes
        force = -self.stiffness * dx - self.damping * dv
        fmag = np.linalg.norm(force)
        if self.max_force > 0.0 and fmag > self.max_force:
            force *= self.max_force / max(fmag, 1e-12)
        system.external_forces[:, self.node_idx] += force


class BaseOrientationSpringDamper(NoForces):
    """
    Apply a soft restoring torque to the base element director plus angular damping.
    This avoids hard orientation constraints while preventing flip-over in force-driven mode.
    """

    def __init__(self, element_idx, reference_director, stiffness, damping, max_torque=0.02):
        self.element_idx = int(element_idx)
        self.reference_director = np.asarray(reference_director, dtype=float).reshape(3, 3)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.max_torque = max(0.0, float(max_torque))

    def apply_torques(self, system, time=0.0):
        idx = self.element_idx
        current = system.director_collection[:, :, idx]
        omega = system.omega_collection[:, idx]
        # Small-angle attitude error from frame misalignment.
        attitude_error = 0.5 * (
            np.cross(current[:, 0], self.reference_director[:, 0])
            + np.cross(current[:, 1], self.reference_director[:, 1])
            + np.cross(current[:, 2], self.reference_director[:, 2])
        )
        torque = -self.stiffness * attitude_error - self.damping * omega
        tmag = np.linalg.norm(torque)
        if self.max_torque > 0.0 and tmag > self.max_torque:
            torque *= self.max_torque / max(tmag, 1e-12)
        system.external_torques[:, idx] += torque


class BaseAngleMonitor(NoForces):
    """
    Reads the current base angle from the first rod element and stores it in landing_state.
    No force or torque is applied.
    """

    def __init__(self, reference_tangent, landing_state):
        self.reference_tangent = np.asarray(reference_tangent, dtype=float).reshape(3)
        self.landing_state = landing_state

    def apply_forces(self, system, time=0.0):
        current_tangent = system.tangents[:, 0]
        angle = signed_angle_xz(self.reference_tangent, current_tangent)
        self.landing_state["ankle_angle"] = angle


class NodalViscousDrag(NoForces):
    """
    Damps rigid-body motion by applying per-node drag force proportional to velocity.
    """

    def __init__(self, damping_coefficient, axis_weights=(1.0, 1.0, 1.0)):
        self.damping_coefficient = max(0.0, float(damping_coefficient))
        self.axis_weights = np.asarray(axis_weights, dtype=float).reshape(3, 1)

    def apply_forces(self, system, time=0.0):
        if self.damping_coefficient <= 0.0:
            return
        drag = -self.damping_coefficient * system.velocity_collection * self.axis_weights
        system.external_forces += drag


class PrescribedLandingPositionBC(ConstraintBase):
    """
    Prescribe only base position during landing; leave orientation free
    so ankle/base rotational dynamics can happen.
    """

    def __init__(
        self,
        fixed_position,
        landing_distance=0.03,
        landing_speed=0.06,
        landing_direction=(0.0, 0.0, -1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.landing_distance = max(0.0, float(landing_distance))
        self.landing_speed = max(0.0, float(landing_speed))
        d = np.asarray(landing_direction, dtype=float).reshape(3)
        dnorm = np.linalg.norm(d)
        self.landing_direction = d / dnorm if dnorm > 1e-12 else np.array([0.0, 0.0, -1.0])
        self._base_position0 = np.array(fixed_position, dtype=float).reshape(3, -1)[:, 0].copy()

    def constrain_values(self, system, time):
        p_idx = int(self.constrained_position_idx[0])

        drop = min(self.landing_distance, self.landing_speed * max(0.0, float(time)))
        target = self._base_position0 + drop * self.landing_direction
        system.position_collection[:, p_idx] = target

    def constrain_rates(self, system, time):
        p_idx = int(self.constrained_position_idx[0])

        still_dropping = (self.landing_speed > 0.0) and (self.landing_speed * float(time) < self.landing_distance)
        v = self.landing_speed * self.landing_direction if still_dropping else np.zeros(3)
        system.velocity_collection[:, p_idx] = v


class BaseXYLockedBC(ConstraintBase):
    """
    Constrain only base X/Y of node 0; leave Z and orientation unconstrained.
    This creates a guided vertical rail for force-driven landing.
    """

    def __init__(self, fixed_position, **kwargs):
        super().__init__(**kwargs)
        self._base_xy = np.array(fixed_position, dtype=float).reshape(3, -1)[:2, 0].copy()

    def constrain_values(self, system, time):
        p_idx = int(self.constrained_position_idx[0])
        system.position_collection[0, p_idx] = self._base_xy[0]
        system.position_collection[1, p_idx] = self._base_xy[1]

    def constrain_rates(self, system, time):
        p_idx = int(self.constrained_position_idx[0])
        system.velocity_collection[0, p_idx] = 0.0
        system.velocity_collection[1, p_idx] = 0.0


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

def signed_angle_xz(v_ref, v_cur):
    ref = np.array([v_ref[0], v_ref[2]], dtype=float)
    cur = np.array([v_cur[0], v_cur[2]], dtype=float)

    n1 = np.linalg.norm(ref)
    n2 = np.linalg.norm(cur)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0

    ref /= n1
    cur /= n2

    dot = np.clip(np.dot(ref, cur), -1.0, 1.0)
    det = ref[0] * cur[1] - ref[1] * cur[0]
    return np.arctan2(det, dot)

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
    parser.add_argument("--final_time", type=float, default=2.0, help="Total simulation time in seconds")
    # cylinder and contact params
    parser.add_argument("--cyl_rad", type=float, default=0.015, help="Cylinder radius (m)")
    parser.add_argument("--k_contact", type=float, default=1.25e3, help="Contact stiffness")
    parser.add_argument(
        "--auto_contact_stiffness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-raise k_contact to a physically motivated minimum based on E and element size.",
    )
    parser.add_argument(
        "--max_penetration_warn",
        type=float,
        default=0.002,
        help="Warn if measured overlap exceeds this value (m).",
    )
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
    parser.add_argument("--joint_softness", type=str, default="0.001,0.0009, 0.0008", help="Multiplier for bending stiffness at vertebrae (0.01 = 1% of original stiffness)")
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
    parser.add_argument(
        "--landing_motion",
        action="store_true",
        help="Enable landing setup (start above branch and use selected landing mode).",
    )
    parser.add_argument(
        "--landing_mode",
        type=str,
        default="prescribed",
        choices=["prescribed", "force_driven"],
        help="Landing dynamics: 'prescribed' enforces base Z trajectory; 'force_driven' uses only forces.",
    )
    parser.add_argument(
        "--landing_speed",
        type=float,
        default=0.0,
        help="Approach speed magnitude (m/s) when landing_motion is enabled.",
    )
    parser.add_argument(
        "--landing_height",
        type=float,
        default=0.03,
        help="Initial extra height above the nominal start pose for landing_motion (m).",
    )
    parser.add_argument(
        "--initial_x_gap",
        type=float,
        default=None,
        help=(
            "Initial horizontal X distance (m) from finger base to cylinder center. "
            "If unset, keeps mode-specific default placement."
        ),
    )
    parser.add_argument(
        "--landing_approach_deg",
        type=float,
        default=30.0,
        help="Approach direction angle from horizontal in XZ plane (deg). 0=horizontal, 90=vertical down.",
    )
    parser.add_argument(
        "--prescribed_stop_at_contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In prescribed landing mode, shorten commanded drop so base does not cross cylinder contact boundary.",
    )
    parser.add_argument(
        "--prescribed_contact_margin",
        type=float,
        default=0.0,
        help="Optional radial margin (m) added to cylinder radius for prescribed base stop condition.",
    )
    parser.add_argument(
        "--base_force_mag",
        type=float,
        default=0.0,
        help="External force magnitude (N) applied to base nodes.",
    )
    parser.add_argument(
        "--base_force_dir",
        type=str,
        default="0,0,-1",
        help="External base force direction as 'x,y,z' (normalized internally).",
    )
    parser.add_argument(
        "--base_force_nodes",
        type=int,
        default=1,
        help="Number of proximal nodes sharing the external base force.",
    )
    parser.add_argument(
        "--full_visualization",
        action="store_true",
        help="Render with fixed global axis limits (entire trajectory + cylinder) instead of auto-follow view.",
    )
    parser.add_argument(
        "--force_driven_stabilize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In force_driven mode, add XY base spring-damper stabilization to prevent runaway drift.",
    )
    parser.add_argument(
        "--force_driven_xy_k",
        type=float,
        default=120.0,
        help="XY spring stiffness (N/m) for force_driven stabilization.",
    )
    parser.add_argument(
        "--force_driven_xy_c",
        type=float,
        default=3.0,
        help="XY damping (N*s/m) for force_driven stabilization.",
    )
    parser.add_argument(
        "--force_driven_tendon_ramp",
        type=float,
        default=1.0,
        help="Ramp-up time (s) for tendon actuation in force_driven mode.",
    )
    parser.add_argument(
        "--force_driven_xy_fmax",
        type=float,
        default=5.0,
        help="Max XY stabilization force magnitude (N) for force_driven mode.",
    )
    parser.add_argument(
        "--force_driven_lock_base_xy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In force_driven mode, hard-lock base X/Y (orientation still free, Z force-driven).",
    )
    parser.add_argument(
        "--force_driven_z_stabilize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In force_driven mode, add base Z spring-damper toward landing target to improve final settling.",
    )
    parser.add_argument(
        "--force_driven_z_k",
        type=float,
        default=120.0,
        help="Base Z stabilization stiffness (N/m) in force_driven mode.",
    )
    parser.add_argument(
        "--force_driven_z_c",
        type=float,
        default=12.0,
        help="Base Z stabilization damping (N*s/m) in force_driven mode.",
    )
    parser.add_argument(
        "--force_driven_z_fmax",
        type=float,
        default=4.0,
        help="Max base Z stabilization force magnitude (N) in force_driven mode.",
    )
    parser.add_argument(
        "--force_driven_z_target",
        type=str,
        default="cylinder",
        choices=["cylinder", "start"],
        help="Z-stabilization target: 'cylinder' uses branch center z, 'start' uses nominal start z.",
    )
    parser.add_argument(
        "--force_driven_z_target_offset",
        type=float,
        default=-0.01,
        help="Offset (m) added to the chosen force-driven Z target.",
    )
    parser.add_argument(
        "--force_driven_min_damping",
        type=float,
        default=5.0,
        help="Minimum internal damping used in force_driven mode for better settling.",
    )
    parser.add_argument(
        "--force_driven_node_drag",
        type=float,
        default=4.0,
        help="External nodal viscous drag coefficient for force_driven mode.",
    )
    parser.add_argument(
        "--force_driven_node_drag_axes",
        type=str,
        default="1,1,1",
        help="Axis weights for nodal drag as 'x,y,z'.",
    )
    parser.add_argument(
        "--force_driven_rot_stabilize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In force_driven mode, add soft rotational spring-damper at base element.",
    )
    parser.add_argument(
        "--force_driven_rot_k",
        type=float,
        default=0.03,
        help="Rotational spring gain for force_driven stabilization.",
    )
    parser.add_argument(
        "--force_driven_rot_c",
        type=float,
        default=0.02,
        help="Rotational damping gain for force_driven stabilization.",
    )
    parser.add_argument(
        "--force_driven_rot_tmax",
        type=float,
        default=0.02,
        help="Max rotational stabilization torque magnitude.",
    )
    parser.add_argument(
        "--disturbance_force_mag",
        type=float,
        default=1.0,
        help="Disturbance force magnitude (N) for post-run contact stability checks.",
    )
    parser.add_argument(
        "--disturbance_base_nodes",
        type=int,
        default=5,
        help="Number of proximal nodes used to define base point for disturbance torque calculation.",
    )
    parser.add_argument(
        "--disturbance_steps",
        type=int,
        default=40,
        help="Number of short rollout steps used to apply disturbance and measure contact response.",
    )
    parser.add_argument(
        "--disturbance_dt_scale",
        type=float,
        default=1.0,
        help="Multiplier on simulation time_step for disturbance rollout integration dt.",
    )

    parser.add_argument(
        '--confirm_tendon', 
        action='store_true', 
        help='Enable tendon force verification mode')
    
    parser.add_argument(
        "--angle_force_mode",
        type=str,
        default="constant",
        choices=["constant", "linear", "sin", "cos"],
        help="How tendon tension varies with ankle angle.",
    )
    parser.add_argument(
        "--angle_gain",
        type=float,
        default=0.0,
        help="Gain for angle-dependent tendon tension.",
    )
    parser.add_argument(
        "--angle_offset",
        type=float,
        default=0.0,
        help="Angle offset (rad) before tension modulation starts.",
    )
    parser.add_argument(
        "--min_tension",
        type=float,
        default=0.0,
        help="Minimum tendon tension.",
    )
    parser.add_argument(
        "--max_tension",
        type=float,
        default=None,
        help="Maximum tendon tension.",
    )

    args = parser.parse_args()

    # Parameters from args (key variables to tune)
    E = args.E
    nu = args.poisson_nu  # Poisson's ratio (0.3~0.4 for TPU)
    G = E / (2 * (1 + nu))  # Shear modulus in Pa
    tension = args.tension
    n_elements = args.n_elements
    damping_constant = args.damping
    landing_motion = args.landing_motion
    landing_mode = args.landing_mode
    landing_speed = abs(args.landing_speed)
    landing_height = args.landing_height
    initial_x_gap = None if args.initial_x_gap is None else max(0.0, float(args.initial_x_gap))
    prescribed_stop_at_contact = bool(args.prescribed_stop_at_contact)
    prescribed_contact_margin = max(0.0, float(args.prescribed_contact_margin))
    force_driven_stabilize = bool(args.force_driven_stabilize)
    force_driven_lock_base_xy = bool(args.force_driven_lock_base_xy)
    force_driven_z_stabilize = bool(args.force_driven_z_stabilize)
    force_driven_xy_k = max(0.0, float(args.force_driven_xy_k))
    force_driven_xy_c = max(0.0, float(args.force_driven_xy_c))
    force_driven_tendon_ramp = max(0.0, float(args.force_driven_tendon_ramp))
    force_driven_xy_fmax = max(0.0, float(args.force_driven_xy_fmax))
    force_driven_z_k = max(0.0, float(args.force_driven_z_k))
    force_driven_z_c = max(0.0, float(args.force_driven_z_c))
    force_driven_z_fmax = max(0.0, float(args.force_driven_z_fmax))
    force_driven_z_target = args.force_driven_z_target
    force_driven_z_target_offset = float(args.force_driven_z_target_offset)
    force_driven_min_damping = max(0.0, float(args.force_driven_min_damping))
    force_driven_node_drag = max(0.0, float(args.force_driven_node_drag))
    force_driven_node_drag_axes = parse_vec3_from_csv(args.force_driven_node_drag_axes, "--force_driven_node_drag_axes")
    force_driven_rot_stabilize = bool(args.force_driven_rot_stabilize)
    force_driven_rot_k = max(0.0, float(args.force_driven_rot_k))
    force_driven_rot_c = max(0.0, float(args.force_driven_rot_c))
    force_driven_rot_tmax = max(0.0, float(args.force_driven_rot_tmax))
    disturbance_force_mag = max(0.0, float(args.disturbance_force_mag))
    disturbance_base_nodes = max(1, int(args.disturbance_base_nodes))
    disturbance_steps = max(1, int(args.disturbance_steps))
    disturbance_dt_scale = max(0.01, float(args.disturbance_dt_scale))
    k_contact = args.k_contact
    auto_contact_stiffness = bool(args.auto_contact_stiffness)
    max_penetration_warn = max(0.0, float(args.max_penetration_warn))
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
    cyl_x = 0.025 # 0.025 default
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

    joint_softness_list = parse_float_list_with_resize(
        args.joint_softness,
        target_len=len(vertebra_nodes),
        arg_name="--joint_softness",
    )

    # Mass of squirrel body
    body_mass = args.body_mass
    gravity = 9.80665
    body_weight_force = np.array([0.0, 0.0, -body_mass * gravity])
    base_force_mag = abs(args.base_force_mag)
    base_force_dir = parse_vec3_from_csv(args.base_force_dir, "--base_force_dir")
    base_force_nodes = max(1, int(args.base_force_nodes))
    base_force_dir_norm = np.linalg.norm(base_force_dir)
    if base_force_mag > 0.0 and base_force_dir_norm < 1e-12:
        raise ValueError("--base_force_dir cannot be zero vector when --base_force_mag > 0")
    base_force_unit = (
        base_force_dir / base_force_dir_norm if base_force_dir_norm > 1e-12 else np.array([0.0, 0.0, 0.0])
    )
    base_external_force = base_force_mag * base_force_unit
    
    # Time stepping parameters
    wave_speed = np.sqrt(E / density)
    dx = base_length / n_elements
    # Physical estimate: k ~ E*A/L where A~(radius*dx), L~(2*radius)
    # This avoids unrealistically deep penetration with low penalty stiffness.
    contact_width = base_radius
    area_per_element = contact_width * dx
    k_contact_physical = (E * area_per_element) / max(2 * base_radius, 1e-12)
    if auto_contact_stiffness and k_contact < k_contact_physical:
        print(
            f"[CONTACT] Auto-raising k_contact from {k_contact:.2e} to physical estimate "
            f"{k_contact_physical:.2e} N/m"
        )
        k_contact = float(k_contact_physical)

    dt_critical = dx / wave_speed
    final_time = float(args.final_time)
    if final_time <= 0.0:
        raise ValueError("--final_time must be > 0")
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

    landing_state = {
        "ankle_angle": 0.0,         # current ankle/base angle in radians
        "pull_scale": 1.0,          # optional debug
        "current_tension": tension, # actual tendon tension used this step
    }

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
        x_gap = (2.0 * cyl_radius) if initial_x_gap is None else initial_x_gap
        start_x = cyl_start[0] - x_gap
        start_z = cyl_z - (cyl_radius) * np.sin(angle_rad)
        print(
            f"[APPROACH ANGLE] Calculated start position: x={start_x:.3f}, z={start_z:.3f} "
            f"(x_gap={x_gap:.3f})"
        )
        
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

    if initial_x_gap is not None and args.sol != "approach_angle":
        start_pos = start_pos.copy()
        start_pos[0] = cyl_x - initial_x_gap
        print(
            f">>> INITIAL GAP OVERRIDE: base x set to {start_pos[0]:.3f} m "
            f"(x_gap={initial_x_gap:.3f} m from cylinder center)"
        )

    start_pos_nominal = start_pos.copy()

    if landing_motion:
        # Start above the branch and let gravity produce the landing.
        start_pos = start_pos.copy()
        start_pos[2] += landing_height
        print(f">>> LANDING MOTION: start height offset = +{landing_height:.3f} m")

    landing_angle_rad = np.deg2rad(float(args.landing_approach_deg))
    landing_drop_dir = np.array(
        [np.cos(landing_angle_rad), 0.0, -np.sin(landing_angle_rad)],
        dtype=float,
    )
    landing_drop_dir /= max(np.linalg.norm(landing_drop_dir), 1e-12)

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

    base_tangent_ref = finger.tangents[:, 0].copy()
    base_director_ref = finger.director_collection[:, :, 0].copy()

    if landing_motion:
        sim.add_forcing_to(finger).using(
            BaseAngleMonitor,
            reference_tangent=base_tangent_ref,
            landing_state=landing_state,
        )

    if landing_motion and landing_speed > 0.0:
        # Give whole finger an initial approach velocity along angled drop direction.
        finger.velocity_collection[:, :] = landing_speed * landing_drop_dir.reshape(3, 1)
        print(
            ">>> LANDING MOTION: initial approach velocity = "
            f"{landing_speed * landing_drop_dir} m/s "
            f"(angle={args.landing_approach_deg:.1f} deg from horizontal)"
        )

    # Apply soft joints at vertebrae locations
    print(f"[SOFT JOINTS] using mode '{args.v_mode}' at nodes: {vertebra_nodes}")
    print(f"[SOFT JOINTS] multipliers: {joint_softness_list}")  

    for i, j in enumerate(vertebra_nodes):

        idx = int(np.clip(j, 0, finger.bend_matrix.shape[2] - 1))
        joint_mult = joint_softness_list[i]
        # progressively softer joints toward tip
        finger.bend_matrix[1, 1, idx] *= joint_mult
        finger.bend_matrix[2, 2, idx] *= joint_mult
        print(f"  node {idx}: softness multiplier = {joint_mult:.6f}")

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
    if not landing_motion:
        sim.constrain(finger).using(
            OneEndFixedBC,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )
    else:
        if landing_mode == "prescribed":
            drop_speed = landing_speed if landing_speed > 0.0 else (landing_height / 0.5)
            landing_distance_cmd = landing_height
            if prescribed_stop_at_contact:
                p0_xz = start_pos[[0, 2]].copy()
                c_xz = np.array([cyl_x, cyl_z], dtype=float)
                d_xz = landing_drop_dir[[0, 2]].copy()
                d_norm = np.linalg.norm(d_xz)
                if d_norm > 1e-12:
                    d_xz /= d_norm
                    # Stop when base-center distance to cylinder center reaches cylinder radius.
                    r_guard = cyl_radius + prescribed_contact_margin
                    rel = p0_xz - c_xz
                    a = np.dot(d_xz, d_xz)
                    b = 2.0 * np.dot(rel, d_xz)
                    c_quad = np.dot(rel, rel) - r_guard**2
                    disc = b * b - 4.0 * a * c_quad
                    if disc >= 0.0:
                        sqrt_disc = np.sqrt(disc)
                        roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
                        roots = [r for r in roots if r >= 0.0]
                        if roots:
                            hit_dist = min(roots)
                            if hit_dist < landing_distance_cmd:
                                landing_distance_cmd = max(0.0, hit_dist)
                                print(
                                    ">>> LANDING MOTION (prescribed): stop-at-contact active, "
                                    f"landing_distance reduced to {landing_distance_cmd:.4f} m "
                                    f"(guard radius={r_guard:.4f} m)"
                                )
            print(
                f">>> LANDING MOTION (prescribed): base downward drop speed={drop_speed:.3f} m/s "
                f"over {landing_distance_cmd:.3f} m"
            )
            sim.constrain(finger).using(
                PrescribedLandingPositionBC,
                constrained_position_idx=(0,),
                landing_distance=landing_distance_cmd,
                landing_speed=drop_speed,
                landing_direction=landing_drop_dir,
            )
            print(">>> PRESCRIBED LANDING: position prescribed, orientation free for ankle motion")

        else:
            print(">>> LANDING MOTION (force_driven): orientation unconstrained, drop driven by forces")
            if force_driven_lock_base_xy:
                sim.constrain(finger).using(
                    BaseXYLockedBC,
                    constrained_position_idx=(0,),
                )
                print(">>> FORCE_DRIVEN: base X/Y locked (Z and orientation remain free)")
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
        if landing_motion and landing_mode == "force_driven":
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
                ramp_up_time=force_driven_tendon_ramp,
                use_gradient=False,
                center_seek=False,
                vertebra_nodes_list=vertebra_nodes,
            )
            print(f">>> FORCE_DRIVEN: tendon ramp-up enabled (ramp_up_time={force_driven_tendon_ramp:.3f}s)")
        else:
            tendon_debug = {
                "stretch_prev": np.zeros((len(vertebra_nodes), 3)),
                "stretch_next": np.zeros((len(vertebra_nodes), 3)),
            }
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
                debug_store=tendon_debug,
                landing_state=landing_state,
                angle_force_mode=args.angle_force_mode,
                angle_gain=args.angle_gain,
                angle_offset=args.angle_offset,
                min_tension=args.min_tension,
                max_tension=args.max_tension,
            )

    # Gravity
    sim.add_forcing_to(finger).using(GravityForces, np.array([0.0, 0.0, -9.80665]))

    # Apply body load only for base-fixed modes; in landing mode this can over-accelerate free flight.
    if not landing_motion:
        sim.add_forcing_to(finger).using(
            BodyWeightForcing,
            force_vector=body_weight_force,
            node_indices=np.arange(0, 5)
        )
    else:
        print(">>> LANDING MOTION: skipping BodyWeightForcing on proximal nodes")

    if base_force_mag > 0.0:
        base_nodes = np.arange(0, min(base_force_nodes, n_elements + 1))
        sim.add_forcing_to(finger).using(
            ConstantNodeForce,
            force_vector=base_external_force,
            node_indices=base_nodes,
        )
        print(
            f">>> BASE EXTERNAL FORCE: |F|={base_force_mag:.3f} N dir={base_force_unit} "
            f"distributed on {len(base_nodes)} node(s)"
        )

    if landing_motion and landing_mode == "force_driven" and force_driven_stabilize and not force_driven_lock_base_xy:
        base_target = finger.position_collection[:, 0].copy()
        sim.add_forcing_to(finger).using(
            BaseAxisSpringDamper,
            node_idx=0,
            target_position=base_target,
            stiffness=force_driven_xy_k,
            damping=force_driven_xy_c,
            active_axes=np.array([1.0, 1.0, 0.0]),
            max_force=force_driven_xy_fmax,
        )
        print(
            f">>> FORCE_DRIVEN STABILIZATION: XY spring-damper on base "
            f"(k={force_driven_xy_k:.2f}, c={force_driven_xy_c:.2f}, fmax={force_driven_xy_fmax:.2f})"
        )
    elif landing_motion and landing_mode == "force_driven" and force_driven_lock_base_xy:
        print(">>> FORCE_DRIVEN STABILIZATION: XY spring-damper skipped (hard XY lock enabled)")
    if landing_motion and landing_mode == "force_driven" and force_driven_rot_stabilize:
        base_director_ref = finger.director_collection[:, :, 0].copy()
        sim.add_forcing_to(finger).using(
            BaseOrientationSpringDamper,
            element_idx=0,
            reference_director=base_director_ref,
            stiffness=force_driven_rot_k,
            damping=force_driven_rot_c,
            max_torque=force_driven_rot_tmax,
        )
        print(
            f">>> FORCE_DRIVEN STABILIZATION: rotational spring-damper at base "
            f"(k={force_driven_rot_k:.3f}, c={force_driven_rot_c:.3f}, tmax={force_driven_rot_tmax:.3f})"
        )

    if landing_motion and landing_mode == "force_driven" and force_driven_z_stabilize:
        base_target_z = finger.position_collection[:, 0].copy()
        if force_driven_z_target == "start":
            z_target = start_pos_nominal[2]
        else:
            z_target = cyl_z
        z_target += force_driven_z_target_offset
        base_target_z[2] = z_target
        sim.add_forcing_to(finger).using(
            BaseAxisSpringDamper,
            node_idx=0,
            target_position=base_target_z,
            stiffness=force_driven_z_k,
            damping=force_driven_z_c,
            active_axes=np.array([0.0, 0.0, 1.0]),
            max_force=force_driven_z_fmax,
        )
        print(
            f">>> FORCE_DRIVEN STABILIZATION: Z spring-damper to landing target "
            f"(target_z={z_target:.4f}, k={force_driven_z_k:.2f}, c={force_driven_z_c:.2f}, fmax={force_driven_z_fmax:.2f})"
        )
    if landing_motion and landing_mode == "force_driven" and force_driven_node_drag > 0.0:
        sim.add_forcing_to(finger).using(
            NodalViscousDrag,
            damping_coefficient=force_driven_node_drag,
            axis_weights=force_driven_node_drag_axes,
        )
        print(
            f">>> FORCE_DRIVEN DAMPING: nodal viscous drag "
            f"(c={force_driven_node_drag:.2f}, axes={force_driven_node_drag_axes})"
        )

    # Damping
    damping_to_apply = damping_constant
    if landing_motion and landing_mode == "force_driven":
        damping_to_apply = max(damping_to_apply, force_driven_min_damping)
    if damping_to_apply > 0.0:
        sim.dampen(finger).using(
            AnalyticalLinearDamper,
            damping_constant=damping_to_apply,
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

    # --- PHYSICAL CONTACT STIFFNESS DIAGNOSTIC ---
    print(f"[PHYSICS] Calculated k_contact: {k_contact_physical:.2e} N/m")


    # Callbacks for data collection
    class CB(CallBackBaseClass):
        def __init__(self, step_skip, callback_params, tendon_debug, landing_state):
            super().__init__()
            self.every = step_skip
            self.callback_params = callback_params
            self.tendon_debug = tendon_debug
            self.landing_state = landing_state

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

                self.callback_params["stretch_prev"].append(self.tendon_debug["stretch_prev"].copy())
                self.callback_params["stretch_next"].append(self.tendon_debug["stretch_next"].copy())

                self.callback_params["ankle_angle"].append(self.landing_state["ankle_angle"])
                self.callback_params["current_tension"].append(self.landing_state["current_tension"])

    data = defaultdict(list)
    sim.collect_diagnostics(finger).using(CB, step_skip=step_skip, callback_params=data, 
                                         tendon_debug=tendon_debug, landing_state=landing_state)

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
    data_to_save["base_radius"] = np.array([base_radius])
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
    # Use geometry-based contact to keep this consistent with contact logging.
    final_pos = data["position"][-1]      # (3, N)
    final_vel = data["velocity"][-1]      # (3, N)
    (
        contact_idx,
        normal_forces,
        _,
        _,
        _,
        _,
        _,
    ) = compute_contact_metrics_frame(
        final_pos,
        final_vel,
        cylinder.position_collection[:, 0],
        cylinder.director_collection[2, :, 0],
        cyl_radius,
        k_contact,
        mu_contact,
        base_radius,
        cyl_length,
    )
    contact_vertices = final_pos[:, contact_idx].T
    cyl_axis_pos = cylinder.position_collection[:, 0]

    ext_force = np.array([0.0, 0.0, -body_mass * gravity])
    ext_torque = np.array([0.0, 0.0, 0.0])
    body_wrench = np.concatenate([ext_force, ext_torque])
    if contact_vertices.shape[0] == 0:
        is_force_closure = False
        energy_score = 0.0
        print("[METRICS] No final contact vertices detected; setting force closure=False and energy=0.")
    else:
        contact_normals = contact_vertices - cyl_axis_pos
        normal_norm = np.linalg.norm(contact_normals, axis=1)
        valid_mask = normal_norm > 1e-12
        if not np.any(valid_mask):
            is_force_closure = False
            energy_score = 0.0
            print("[METRICS] Contact normals degenerate; setting force closure=False and energy=0.")
        else:
            contact_vertices_valid = contact_vertices[valid_mask]
            contact_normals_valid = contact_normals[valid_mask] / normal_norm[valid_mask][:, None]
            is_force_closure = check_force_closure(
                contact_vertices_valid,
                contact_normals_valid,
                mu_contact,
                external_wrench=body_wrench,
            )
            energy_score = compute_total_energy(finger, normal_forces[valid_mask], k_contact)

    data_to_save["metric_is_force_closure"] = np.array([is_force_closure])
    data_to_save["metric_energy_total"] = np.array([energy_score])
    data_to_save["metric_contact_count"] = np.array([len(contact_idx)])
    data_to_save["body_mass"] = np.array([body_mass])
    print(f"[METRICS] Force Closure Stable: {is_force_closure} under {body_mass}kg load  Total Energy: {energy_score:.6f} J  Contact Points: {len(contact_idx)}")

    pos_data = data["position"]
    vel_data = data["velocity"]
    if len(pos_data) < 2:
        raise RuntimeError("No saved frames")

    if args.confirm_tendon:
        print("\n" + "="*70)
        print("TENDON FORCE VALIDATION (LAST FRAME)")
        print("="*70)

        ext_force_data = data["external_forces"]
        stretch_prev_data = data["stretch_prev"]
        stretch_next_data = data["stretch_next"]

        last_idx = min(
            len(pos_data),
            len(stretch_prev_data),
            len(stretch_next_data),
            len(ext_force_data),
        ) - 1

        F_last = ext_force_data[last_idx]
        SP_last = stretch_prev_data[last_idx]
        SN_last = stretch_next_data[last_idx]

        for i, v_idx in enumerate(vertebra_nodes):
            v_idx = int(v_idx)

            sp_vec = SP_last[i]
            sn_vec = SN_last[i]
            tendon_sum_vec = sp_vec + sn_vec
            external_vec = F_last[:, v_idx]

            sp_mag = np.linalg.norm(sp_vec)
            sn_mag = np.linalg.norm(sn_vec)
            sum_mag = np.linalg.norm(tendon_sum_vec)
            ext_mag = np.linalg.norm(external_vec)

            sum_dir = tendon_sum_vec / sum_mag if sum_mag > 1e-12 else np.zeros(3)
            ext_dir = external_vec / ext_mag if ext_mag > 1e-12 else np.zeros(3)

            diff_vec = external_vec - tendon_sum_vec
            diff_mag = np.linalg.norm(diff_vec)

            print(f"\nNode {v_idx:3}")
            print(f"  stretch_prev mag = {sp_mag:.6f} | vec = [{sp_vec[0]: .6f}, {sp_vec[1]: .6f}, {sp_vec[2]: .6f}]")
            print(f"  stretch_next mag = {sn_mag:.6f} | vec = [{sn_vec[0]: .6f}, {sn_vec[1]: .6f}, {sn_vec[2]: .6f}]")
            print(f"  tendon_sum   mag = {sum_mag:.6f} | dir = [{sum_dir[0]: .3f}, {sum_dir[1]: .3f}, {sum_dir[2]: .3f}]")
            print(f"  external     mag = {ext_mag:.6f} | dir = [{ext_dir[0]: .3f}, {ext_dir[1]: .3f}, {ext_dir[2]: .3f}]")
            print(f"  |external - tendon_sum| = {diff_mag:.6e}")

    csv_path = os.path.join(base_outdir, f"contact_log_{run_id}_{suffix}.csv")

    print("\n" + "="*60)
    print("ANGLE-DEPENDENT PULLING SUMMARY")
    print("="*60)
    print(f"Final ankle angle (rad): {data['ankle_angle'][-1]:.6f}")
    print(f"Final ankle angle (deg): {np.degrees(data['ankle_angle'][-1]):.3f}")
    last_idx = min(
        len(data["stretch_prev"]),
        len(data["stretch_next"]),
        len(data["current_tension"]),
        len(data["ankle_angle"]),
    ) - 1

    SP_last = data["stretch_prev"][last_idx]
    SN_last = data["stretch_next"][last_idx]
    current_tension_last = data["current_tension"][last_idx]

    for i, v_idx in enumerate(vertebra_nodes):
        sp_vec = SP_last[i]
        sn_vec = SN_last[i]
        tendon_sum_vec = sp_vec + sn_vec

        sp_mag = np.linalg.norm(sp_vec)
        sn_mag = np.linalg.norm(sn_vec)
        sum_mag = np.linalg.norm(tendon_sum_vec)

        print(f"\nNode {int(v_idx):3}")
        print(f"  stretch_prev      = [{sp_vec[0]: .6f}, {sp_vec[1]: .6f}, {sp_vec[2]: .6f}] |mag|={sp_mag:.6f}")
        print(f"  stretch_next      = [{sn_vec[0]: .6f}, {sn_vec[1]: .6f}, {sn_vec[2]: .6f}] |mag|={sn_mag:.6f}")
        print(f"  tendon_resultant  = [{tendon_sum_vec[0]: .6f}, {tendon_sum_vec[1]: .6f}, {tendon_sum_vec[2]: .6f}] |mag|={sum_mag:.6f}")
        print(f"  current tension   = {current_tension_last:.6f}")
    print("\n" + "="*60)
    
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
            # Keep this threshold consistent with compute_contact_metrics_frame.
            total_contacts += int(np.sum((cyl_radius + base_radius - radial_dist) > 0.0))
        best = max(best, (total_contacts, min_rad, a), key=lambda x: x[0]) if best else (total_contacts, min_rad, a)

    total_contacts, min_rad, cyl_axis = best
    print(f"[CONTACT DEBUG] total_contact_node_hits_over_all_frames={total_contacts}")
    print(f"[CONTACT DEBUG] min_distance_to_cylinder_axis={min_rad:.6f} (radius={cyl_radius:.6f})")
    if total_contacts == 0:
        print("[CONTACT DEBUG] No penetration detected => rod never actually reaches the cylinder (or axis dir wrong).")

    first_contact_frame = None
    max_normal_force_overall = 0.0
    max_overlap_overall = 0.0
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
                frame_overlap_max = float(np.max(overlap_all[idx]))
                max_overlap_overall = max(max_overlap_overall, frame_overlap_max)
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
        if landing_motion and landing_mode == "prescribed" and prescribed_stop_at_contact:
            print(
                "[CONTACT HINT] prescribed_stop_at_contact may be too conservative for this setup. "
                "Try --no-prescribed_stop_at_contact to allow full landing motion."
            )
    else:
        print(f"[CONTACT] First contact frame={first_contact_frame}  t={first_contact_frame*dt_saved:.4f}s")
        print(f"[CONTACT] Max normal force over all frames: {max_normal_force_overall:.6f} N")
        print(f"[CONTACT] Max overlap over all frames: {max_overlap_overall:.6f} m")
        if max_overlap_overall > max_penetration_warn:
            print(
                "[CONTACT WARNING] Large penetration detected. "
                f"overlap={max_overlap_overall:.6f} m > warn={max_penetration_warn:.6f} m. "
                "Consider increasing --k_contact/--nu_contact, reducing --tension, or using --landing_mode force_driven."
            )
            if landing_motion and landing_mode == "prescribed":
                print(
                    "[CONTACT WARNING] Prescribed landing enforces a kinematic base path; "
                    "this can still drive penetration under high tendon load."
                )
        print(f"[CONTACT] Wrote {os.path.join(base_outdir, f'contact_log_{run_id}_{suffix}.csv')}")

    data_to_save["metric_contact_max_overlap"] = np.array([max_overlap_overall])

    print(f"\n[CONTACT] Starting geometric metrics check for: {csv_path}")
    is_fc = False
    is_stable = False
    metrics = {
        "num_contacts": 0,
        "angular_span": 0.0,
        "total_normal_force": 0.0,
        "total_friction_force": 0.0,
        "max_normal_force": 0.0
    }
    final_score = float("nan")
    breakdown = {}
    disturbance_results = {}
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

        # Disturbance stability checks from final settled contact state.
        base_count = min(disturbance_base_nodes, final_pos.shape[1])
        base_point = final_pos[:, :base_count].mean(axis=1)
        base_indices = np.arange(base_count, dtype=int)
        total_rod_mass = float(np.sum(finger.mass))
        node_mass_uniform = np.full(final_pos.shape[1], total_rod_mass / max(1, final_pos.shape[1]))
        disturbance_dt = time_step * disturbance_dt_scale
        disturbance_cases = {
            "drag_left": np.array([-1.0, 0.0, 0.0]),
            "drag_right": np.array([1.0, 0.0, 0.0]),
            "drag_down": np.array([0.0, 0.0, -1.0]),
        }

        resistance_count = 0
        total_cases = len(disturbance_cases)
        for name, dvec in disturbance_cases.items():
            applied_force = disturbance_force_mag * dvec
            resp = evaluate_disturbance_contact_response(
                rod_pos=final_pos,
                rod_vel=final_vel,
                cyl_center=cylinder.position_collection[:, 0],
                cyl_radius=cyl_radius,
                k=k_contact,
                mu=mu_contact,
                base_radius=base_radius,
                base_indices=base_indices,
                node_masses=node_mass_uniform,
                disturbance_force=applied_force,
                n_steps=disturbance_steps,
                dt=disturbance_dt,
            )
            disturbance_results[name] = resp

            if resp["force_resisted"] and resp["torque_resisted"]:
                resistance_count += 1

            data_to_save[f"disturbance_{name}_applied_force"] = resp["applied_force"]
            data_to_save[f"disturbance_{name}_applied_torque"] = resp["applied_torque"]
            data_to_save[f"disturbance_{name}_net_contact_force"] = resp["net_contact_force"]
            data_to_save[f"disturbance_{name}_net_contact_torque"] = resp["net_contact_torque"]
            data_to_save[f"disturbance_{name}_resist_force"] = np.array([resp["resist_force"]])
            data_to_save[f"disturbance_{name}_resist_torque"] = np.array([resp["resist_torque"]])
            data_to_save[f"disturbance_{name}_force_alignment"] = np.array([resp["force_alignment"]])
            data_to_save[f"disturbance_{name}_torque_alignment"] = np.array([resp["torque_alignment"]])
            data_to_save[f"disturbance_{name}_force_resisted"] = np.array([resp["force_resisted"]])
            data_to_save[f"disturbance_{name}_torque_resisted"] = np.array([resp["torque_resisted"]])
            data_to_save[f"disturbance_{name}_contacts_history"] = resp["contacts_history"]
            data_to_save[f"disturbance_{name}_force_history"] = resp["force_history"]

        data_to_save["disturbance_base_point"] = base_point
        data_to_save["disturbance_force_application_point"] = base_point # TODO
        data_to_save["disturbance_resistance_score"] = np.array([resistance_count / total_cases])

        print("\n[DISTURBANCE STABILITY CHECK]")
        for name, resp in disturbance_results.items():
            print(
                f"  {name}: contacts={resp['num_contacts']} "
                f"|F_app|={np.linalg.norm(resp['applied_force']):.3f} "
                f"|F_contact|={np.linalg.norm(resp['net_contact_force']):.3f} "
                f"align_F={resp['force_alignment']:.3f} "
                f"resist_F={resp['resist_force']:.3f} "
                f"force_resisted={resp['force_resisted']}"
            )
            print(
                f"    |tau_app|={np.linalg.norm(resp['applied_torque']):.3f} "
                f"|tau_contact|={np.linalg.norm(resp['net_contact_torque']):.3f} "
                f"align_tau={resp['torque_alignment']:.3f} "
                f"resist_tau={resp['resist_torque']:.3f} "
                f"torque_resisted={resp['torque_resisted']}"
            )

        # Save one force-visualization image per disturbance case.
        for name, resp in disturbance_results.items():
            fig_dist = plt.figure(figsize=(13, 6))
            ax_dist = fig_dist.add_subplot(121, projection="3d")
            ax_contact = fig_dist.add_subplot(122, projection="3d")
            P_dist = resp["position_final"]
            ax_dist.scatter(P_dist[0], P_dist[1], P_dist[2], s=8, alpha=0.8)
            center_dist = cylinder.position_collection[:, 0]
            axis_dist = cylinder.director_collection[2, :, 0]
            draw_cylinder(ax_dist, center_dist, axis_dist, cyl_radius, cyl_length, color="black", alpha=0.25)

            # Mark base point used for disturbance application and wrench reference.
            ax_dist.scatter(
                base_point[0], base_point[1], base_point[2],
                color="black", s=60, marker="x", depthshade=False,
            )

            f_app = resp["applied_force"]
            f_net = resp["net_contact_force"]
            if np.linalg.norm(f_app) > 1e-12:
                ax_dist.quiver(
                    base_point[0], base_point[1], base_point[2],
                    f_app[0], f_app[1], f_app[2],
                    length=0.03, normalize=True, color="tab:red",
                )
            if np.linalg.norm(f_net) > 1e-12:
                ax_dist.quiver(
                    base_point[0], base_point[1], base_point[2],
                    f_net[0], f_net[1], f_net[2],
                    length=0.03, normalize=True, color="tab:green",
                )

            ax_dist.set_xlim(-0.02, 0.12)
            ax_dist.set_ylim(-0.12, 0.12)
            ax_dist.set_zlim(-0.10, 0.10)
            ax_dist.set_title(
                f"Disturbance: {name} |F_app|={np.linalg.norm(f_app):.3f} "
                f"|F_contact|={np.linalg.norm(f_net):.3f}"
            )
            ax_dist.view_init(elev=0, azim=-90)
            ax_dist.grid(True, alpha=0.3)

            # Subplot 2: all per-contact force vectors on contact elements.
            ax_contact.scatter(P_dist[0], P_dist[1], P_dist[2], s=6, alpha=0.25, color="gray")
            draw_cylinder(ax_contact, center_dist, axis_dist, cyl_radius, cyl_length, color="black", alpha=0.2)
            cp = resp["contact_positions"]
            cf = resp["contact_forces"]
            if cp.shape[1] > 0:
                ax_contact.scatter(cp[0], cp[1], cp[2], s=28, color="tab:blue", alpha=0.9)
                for j in range(cp.shape[1]):
                    if np.linalg.norm(cf[j]) < 1e-10:
                        continue
                    ax_contact.quiver(
                        cp[0, j], cp[1, j], cp[2, j],
                        cf[j, 0], cf[j, 1], cf[j, 2],
                        length=0.02, normalize=True, color="tab:purple", alpha=0.9,
                    )
            ax_contact.scatter(
                base_point[0], base_point[1], base_point[2],
                color="black", s=60, marker="x", depthshade=False,
            )
            ax_contact.set_xlim(-0.02, 0.12)
            ax_contact.set_ylim(-0.12, 0.12)
            ax_contact.set_zlim(-0.10, 0.10)
            ax_contact.set_title(f"Per-contact force vectors ({cp.shape[1]} contacts)")
            ax_contact.view_init(elev=0, azim=-90)
            ax_contact.grid(True, alpha=0.3)
            plt.tight_layout()
            disturbance_plot_path = os.path.join(
                base_outdir, f"disturbance_force_{name}_{run_id}_{suffix}.png"
            )
            plt.savefig(disturbance_plot_path, dpi=180, bbox_inches="tight")
            plt.close(fig_dist)
            print(f"[DISTURBANCE] Force visualization saved: {disturbance_plot_path}")

        cycle_time, strength, slip_res = calculate_nist_scores(
            data, cyl_radius, base_radius, k_contact, mu_contact, args.body_mass, gravity
        )
        normal_forces_arr = np.asarray(normal_forces, dtype=float)
        max_normal_force_for_score = float(np.max(normal_forces_arr)) if normal_forces_arr.size > 0 else 0.0
        total_normal_force_for_score = float(np.sum(normal_forces_arr)) if normal_forces_arr.size > 0 else 0.0

        grasp_data = {
            "angular_span": metrics['angular_span'], # Calculate from your contact log
            "vertical_support": stab_metrics['total_support_n'],
            "body_weight": args.body_mass * gravity,
            "total_energy": energy_score,
            "contact_count": metrics['num_contacts'],
            "max_normal_force": max_normal_force_for_score,
            "total_normal_force": total_normal_force_for_score,
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

    full_limits = None
    if args.full_visualization:
        pos_stack = np.stack(pos_data, axis=0)  # (n_saved, 3, n_nodes)
        rod_min = np.min(pos_stack, axis=(0, 2))
        rod_max = np.max(pos_stack, axis=(0, 2))
        cyl_center_vis = cylinder.position_collection[:, 0].copy()
        cyl_pad = np.array([cyl_radius, cyl_length * 0.5, cyl_radius])
        vis_min = np.minimum(rod_min, cyl_center_vis - cyl_pad)
        vis_max = np.maximum(rod_max, cyl_center_vis + cyl_pad)
        margin = np.array([0.02, 0.02, 0.02])
        full_limits = (vis_min - margin, vis_max + margin)

    def make_frame(t):
        ax.clear()
        rod_radius = base_radius
        frame_idx = min(int(max(0.0, t) / dt_saved), len(pos_data) - 1)
        P = pos_data[frame_idx]
        V = vel_data[frame_idx]
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

        if full_limits is not None:
            vis_min, vis_max = full_limits
            ax.set_xlim(vis_min[0], vis_max[0])
            ax.set_ylim(vis_min[1], vis_max[1])
            ax.set_zlim(vis_min[2], vis_max[2])
        elif landing_motion:
            # Keep both finger and branch visible in guided landing mode.
            center_p = np.mean(P, axis=1)
            center_c = cylinder.position_collection[:, 0]
            center_mix = 0.5 * (center_p + center_c)
            x_half = max(0.08, 0.5 * abs(center_p[0] - center_c[0]) + 0.04)
            z_half = max(0.08, 0.5 * abs(center_p[2] - center_c[2]) + 0.04)
            ax.set_xlim(center_mix[0] - x_half, center_mix[0] + x_half)
            ax.set_ylim(-0.12, 0.12)
            ax.set_zlim(center_mix[2] - z_half, center_mix[2] + z_half)
        else:
            ax.set_xlim(-0.02, 0.12)
            ax.set_ylim(-0.12, 0.12)
            ax.set_zlim(-0.10, 0.10)

        # --- Force visualization of forces ---
        if args.confirm_tendon or args.debug:
            F = data["external_forces"][frame_idx]          # (3, n_nodes)
            mag = np.linalg.norm(F, axis=0)  # (n_nodes,)

            if args.confirm_tendon:
                sp_frame = data["stretch_prev"][frame_idx]  # Red Arrow 1
                sn_frame = data["stretch_next"][frame_idx]  # Red Arrow 2

                for i, v_idx in enumerate(vertebra_nodes):

                    pos = P[:, v_idx]

                    stretch_prev_vec = sp_frame[i]
                    stretch_next_vec = sn_frame[i]

                    tendon_sum_vec = stretch_prev_vec + stretch_next_vec
                    external_vec = F[:, v_idx]

                    # red = stretch_prev
                    ax.quiver(
                        pos[0], pos[1], pos[2],
                        stretch_prev_vec[0], stretch_prev_vec[1], stretch_prev_vec[2],
                        color="red", length=0.02
                    )

                    # blue = stretch_next
                    ax.quiver(
                        pos[0], pos[1], pos[2],
                        stretch_next_vec[0], stretch_next_vec[1], stretch_next_vec[2],
                        color="blue", length=0.02
                    )

                    # green = SUM (actual tendon resultant)
                    ax.quiver(
                        pos[0], pos[1], pos[2],
                        tendon_sum_vec[0], tendon_sum_vec[1], tendon_sum_vec[2],
                        color="green", length=0.02
                    )

                    # magenta = simulation external force
                    ax.quiver(
                        pos[0], pos[1], pos[2],
                        external_vec[0], external_vec[1], external_vec[2],
                        color="magenta", length=0.02
                    )
                
                # for i, v_idx in enumerate(vertebra_nodes):
                #     v_idx = int(v_idx)
                #     pos = P[:, v_idx]
                    
                #     # Plot Red Arrows (The Stretch)
                #     ax.quiver(pos[0], pos[1], pos[2], sp_frame[i,0], sp_frame[i,1], sp_frame[i,2], 
                #             color='red', length=0.02, normalize=True)
                #     ax.quiver(pos[0], pos[1], pos[2], sn_frame[i,0], sn_frame[i,1], sn_frame[i,2], 
                #             color='red', length=0.02, normalize=True)
                    
                #     # Plot Green Arrow (The Resultant)
                #     f_res = F_last[:, v_idx]
                #     ax.quiver(pos[0], pos[1], pos[2], f_res[0], f_res[1], f_res[2], 
                #             color='green', length=0.02)


            else: 
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

                # Draw configured base external force only when enabled.
                if base_force_mag > 0.0:
                    base_pos = data["position"][-1][:, :5].mean(axis=1)
                    f_vec = base_external_force
                    ax.quiver(
                        base_pos[0], base_pos[1], base_pos[2],
                        f_vec[0], f_vec[1], f_vec[2],
                        color='red', linewidth=3, label='Base external force',
                        arrow_length_ratio=0.3
                    )

                # Contact force vectors and net contact wrench (about base).
                (
                    cidx,
                    contact_forces,
                    _,
                    _,
                    _,
                    _,
                ) = compute_contact_force_vectors(
                    rod_pos=P,
                    rod_vel=V,
                    cyl_center=cylinder.position_collection[:, 0],
                    cyl_radius=cyl_radius,
                    k=k_contact,
                    mu=mu_contact,
                    base_radius=base_radius,
                )
                base_count_vis = min(disturbance_base_nodes, P.shape[1])
                base_point_vis = P[:, :base_count_vis].mean(axis=1)
                ax.scatter(
                    base_point_vis[0], base_point_vis[1], base_point_vis[2],
                    color="black", s=45, marker="x", depthshade=False,
                )
                if len(cidx) > 0:
                    cscale = 0.01
                    for j in range(len(cidx)):
                        cf = contact_forces[j]
                        if np.linalg.norm(cf) < 1e-8:
                            continue
                        ax.quiver(
                            base_point_vis[0], base_point_vis[1], base_point_vis[2],
                            cf[0], cf[1], cf[2],
                            length=cscale,
                            normalize=True,
                            color="yellow",
                        )
                    net_cf = np.sum(contact_forces, axis=0)
                    if np.linalg.norm(net_cf) > 1e-8:
                        ax.quiver(
                            base_point_vis[0], base_point_vis[1], base_point_vis[2],
                            net_cf[0], net_cf[1], net_cf[2],
                            length=0.03,
                            normalize=True,
                            color="lime",
                        )
                    cp = P[:, cidx]
                    r = cp.T - base_point_vis[None, :]
                    net_tau = np.sum(np.cross(r, contact_forces), axis=0)
                    if np.linalg.norm(net_tau) > 1e-8:
                        ax.quiver(
                            base_point_vis[0], base_point_vis[1], base_point_vis[2],
                            net_tau[0], net_tau[1], net_tau[2],
                            length=0.03,
                            normalize=True,
                            color="orange",
                        )
            
        center = cylinder.position_collection[:, 0]
        axis_dir = cylinder.director_collection[2, :, 0]
        draw_cylinder(ax, center, axis_dir, cyl_radius, cyl_length, color="black", alpha=0.35)

        # front view: from -Y looking toward +Y
        ax.view_init(elev=0, azim=-90)

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

        if disturbance_results:
            base_count_live = min(disturbance_base_nodes, P.shape[1])
            base_point_live = P[:, :base_count_live].mean(axis=1)
            ax_live.scatter(
                base_point_live[0], base_point_live[1], base_point_live[2],
                color="black", s=70, marker="x", depthshade=False,
            )
            disturbance_point_live = base_point_live
            colors = {
                "drag_left": "tab:blue",
                "drag_right": "tab:green",
                "drag_down": "tab:red",
            }
            for key, resp in disturbance_results.items():
                col = colors.get(key, "black")
                f_app = resp["applied_force"]
                f_net = resp["net_contact_force"]
                t_net = resp["net_contact_torque"]
                if np.linalg.norm(f_app) > 1e-12:
                    ax_live.quiver(
                        disturbance_point_live[0], disturbance_point_live[1], disturbance_point_live[2],
                        f_app[0], f_app[1], f_app[2],
                        length=0.03, normalize=True, color=col,
                    )
                if np.linalg.norm(f_net) > 1e-12:
                    ax_live.quiver(
                        base_point_live[0], base_point_live[1], base_point_live[2],
                        f_net[0], f_net[1], f_net[2],
                        length=0.03, normalize=True, color=col, alpha=0.6,
                    )
                if np.linalg.norm(t_net) > 1e-12:
                    ax_live.quiver(
                        base_point_live[0], base_point_live[1], base_point_live[2],
                        t_net[0], t_net[1], t_net[2],
                        length=0.03, normalize=True, color="orange", alpha=0.7,
                    )

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