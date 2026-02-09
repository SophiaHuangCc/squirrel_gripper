import numpy as np
from TendonForces import TendonForces
from elastica.external_forces import NoForces
from elastica.modules import BaseSystemCollection, Connections, Constraints, Forcing, CallBacks, Damping, Contact


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