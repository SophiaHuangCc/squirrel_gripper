import numpy as np
from elastica.typing import SystemType, RodType

# Normally NoForces would be included in the forcing module, but PyElastica requires it to be from the original elastica.external_forces module
from elastica.external_forces import NoForces
from numba import njit

class TendonForces(NoForces):
    """
    This class applies tendon forcing along the length of the rod.

        Attributes
        ----------
        vertebra_height: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
        num_vertebrae: int
            Amount of vertebrae to be used in the system.
        vertebra_height_vector: numpy.ndarray
            1D (dim) numpy array. Describes the orientation and height in space of the vertebrae in the system.
        tension: float
            Tension applied to the tendon in the system.
        n_elements: int
            Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
        vertebra_weight_vector: numpy.ndarray
            1D (dim) numpy array. Vector which specifies the orientation and magnitude of the weight of the vertebrae (By default it is in the global -Z direction).
        vertebra_nodes: list
            1D (dim) list. Contains the node numbers of every node with vertebrae. The vertebrae are assumed to be uniformly spaced through the intervals specified by 
            first_vertebra_node and final_vertebra_node, with an amount equal to num_vertebrae.
        force_data: numpy.ndarray
            2D (dim,3) numpy array. Contains the force vectors caused by tendon forcing for each of the nodes with vertebrae.
        

    """

    def __init__(self, vertebra_height, num_vertebrae, first_vertebra_node, final_vertebra_node, vertebra_mass, tension, vertebra_height_orientation, n_elements,
                 vertebra_nodes_list=None, debug_store=None, landing_state=None, ankle_wrap_radius=0.0, ankle_stiffness=0.0, ankle_rest_angle=0.0, min_tension=0.0, max_tension=100.0,
                 ankle_contact_gated=False, distal_anchor_node=-1):
        """

        Parameters 
        ----------
        vertebra_height: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
        num_vertebrae: int
            Amount of vertebrae to be used in the system.
        first_vertebra_node: int
            The first node to have a vertebra, from the base of the rod to the tip.
        final_vertebra_node: int
            The last node to have a vertebra, from the base of the rod to the tip.
        vertebra_mass: float
            Total mass of a single vertebra.
        tension: float
            Tension applied to the tendon in the system.
        vertebra_height_orientation: numpy.ndarray
            1D (dim) numpy array. Describes the orientatation of the vertebrae in the system.
        n_elements: int
            Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
        vertebra_nodes_list: list, optional
            1D (dim) list. If provided, this list will be used as the vertebra_nodes instead of calculating them uniformly.
        """
        super(TendonForces, self).__init__()

        # Initializing class attributes to be used in other methods
        self.vertebra_height = vertebra_height
        self.num_vertebrae = num_vertebrae
        self.vertebra_height_vector = vertebra_height_orientation * vertebra_height
        self.tension = tension
        self.n_elements = n_elements
        self.vertebra_weight_vector = np.array([0.0, 0.0, -vertebra_mass * 9.80665])
        self.debug_store = debug_store
        
        self.debug_store = debug_store
        self.landing_state = landing_state
        self.ankle_wrap_radius = float(ankle_wrap_radius)
        self.ankle_stiffness = float(ankle_stiffness)
        self.ankle_rest_angle = float(ankle_rest_angle)
        self.min_tension = float(min_tension)
        self.max_tension = None if max_tension is None else float(max_tension)
        self.nominal_tension = float(tension)
        self.ankle_contact_gated = bool(ankle_contact_gated)
        self.distal_anchor_node = int(distal_anchor_node)

        # self.stretch_next = np.zeros((self.num_vertebrae, 3))
        # self.stretch_prev = np.zeros((self.num_vertebrae, 3))

        ##### Start of modified section #####
        # If a manual list of vertebra nodes is provided, use it
        if vertebra_nodes_list is not None:
            self.vertebra_nodes = list(vertebra_nodes_list)
            self.num_vertebrae = len(self.vertebra_nodes)
        else: # linear interpolation of vertebra nodes
            self.vertebra_nodes = []
            vertebra_increment = (final_vertebra_node - first_vertebra_node)/(num_vertebrae - 1)
            for i in range(num_vertebrae):
                self.vertebra_nodes.append(round(i * vertebra_increment + first_vertebra_node))

        # Creating vector containing the node numbers with the vertebras for this instance of TendonForces
        # self.vertebra_nodes = []
        # vertebra_increment = (final_vertebra_node - first_vertebra_node)/(num_vertebrae - 1)
        # for i in range(num_vertebrae):
        #     self.vertebra_nodes.append(round(i * vertebra_increment + first_vertebra_node))
        ##### End of modified section #####

    def _get_current_tension(self):
        ls = self.landing_state
        if ls is None:
            current_tension = self.nominal_tension
            if current_tension < self.min_tension:
                current_tension = self.min_tension
            if self.max_tension is not None:
                current_tension = min(current_tension, self.max_tension)
            return current_tension

        angle_abs = float(ls.get("ankle_angle", 0.0))

        # Pre-contact: no wrap stretch; T = T_0 (see ContactAnkleRestMonitor in finger.py).
        if self.ankle_contact_gated and not ls.get("rod_cylinder_contact", False):
            current_tension = self.nominal_tension
            if current_tension < self.min_tension:
                current_tension = self.min_tension
            if self.max_tension is not None:
                current_tension = min(current_tension, self.max_tension)
            ls["current_tension"] = current_tension
            ls["delta_theta"] = 0.0
            ls["delta_L"] = 0.0
            ls["minus_delta_L"] = 0.0
            ls["theta_rest_used"] = angle_abs
            return current_tension

        if self.ankle_contact_gated:
            theta_rest = ls.get("ankle_rest_angle_effective")
            if theta_rest is None:
                theta_rest = self.ankle_rest_angle
        else:
            theta_rest = self.ankle_rest_angle

        delta_theta = angle_abs - theta_rest
        # Wrap kinematics (straight segments fixed): delta_L = -r * delta_theta.
        delta_L = -self.ankle_wrap_radius * delta_theta
        # Elastic law: T = T_0 + k_ankle * (-delta_L).
        minus_delta_L = -delta_L
        current_tension = self.nominal_tension + self.ankle_stiffness * minus_delta_L

        if current_tension < self.min_tension:
            current_tension = self.min_tension
        if self.max_tension is not None:
            current_tension = min(current_tension, self.max_tension)

        ls["current_tension"] = current_tension
        ls["delta_theta"] = delta_theta
        ls["delta_L"] = delta_L
        ls["minus_delta_L"] = minus_delta_L
        ls["theta_rest_used"] = theta_rest

        return current_tension

    def apply_forces(self, system: SystemType, time: np.float64 = 0.0):
        # The application of the force data is done outside of the @njit decorated function because self.force_data needs to be referenced in self.compute_torques()

        # Retrieves relative position unit norm vectors between each vertebra top (where the tendon contacts the vertebra)
        unit_norm_vector_array = self.get_rotations(
            np.array(system.position_collection),
            np.array(system.director_collection),
            np.array(self.vertebra_nodes),
            self.vertebra_height_vector,
            self.distal_anchor_node,
        )


        current_tension = self._get_current_tension()

        # Modified: add stretch forces
        self.stretch_next = unit_norm_vector_array[1:] * current_tension
        self.stretch_prev = unit_norm_vector_array[:-1] * -current_tension

        # Computes the forces in each vertebra
        self.force_data = self.compute_forces(current_tension, np.array(self.vertebra_nodes), unit_norm_vector_array)
        distal_anchor_force = np.zeros(3)
        if self.distal_anchor_node >= 0:
            distal_anchor_force = -unit_norm_vector_array[-1] * current_tension

        # Creating the force data set to apply to the rod
        apply_force = np.zeros((3,self.n_elements+1))

        # PyElastica handles forces in GLOBAL coord. system, so they are applied directly. Also, the vertebra weights are added to each vertebra
        for i in range (len(self.vertebra_nodes)):
            apply_force[:,self.vertebra_nodes[i]] = self.force_data[i] + self.vertebra_weight_vector
        if self.distal_anchor_node >= 0:
            anchor_node = min(max(self.distal_anchor_node, 0), self.n_elements)
            apply_force[:, anchor_node] += distal_anchor_force

        # Applies forces to the rod
        system.external_forces += apply_force

        if self.debug_store is not None:
            self.debug_store["stretch_next"] = self.stretch_next.copy()
            self.debug_store["stretch_prev"] = self.stretch_prev.copy()
            # The proximal force acts toward the previous tendon anchor.
            self.debug_store["tendon_direction_prev"] = (-unit_norm_vector_array[:-1]).copy()
            self.debug_store["tendon_direction_next"] = unit_norm_vector_array[1:].copy()
            self.debug_store["resultant_force_global"] = self.force_data.copy()
            self.debug_store["tendon_path_length"] = self.get_path_length(
                np.array(system.position_collection),
                np.array(system.director_collection),
                np.array(self.vertebra_nodes),
                self.vertebra_height_vector,
                self.distal_anchor_node,
            )

    @staticmethod
    @njit(cache=True)
    def get_path_length(position_collection, director_collection, vertebra_nodes,
                        vertebra_height_vector, distal_anchor_node):
        """Length of the modeled tendon polyline through its routing points."""
        n_nodes = position_collection.shape[1]
        n_directors = director_collection.shape[2]
        route_nodes = np.empty(len(vertebra_nodes) + 2, dtype=np.int64)
        route_nodes[0] = 0
        for i in range(len(vertebra_nodes)):
            route_nodes[i + 1] = vertebra_nodes[i]
        route_count = len(vertebra_nodes) + 1
        if distal_anchor_node >= 0:
            route_nodes[route_count] = min(max(distal_anchor_node, 0), n_nodes - 1)
            route_count += 1

        total = 0.0
        previous = np.zeros(3, dtype=np.float64)
        for i in range(route_count):
            node = min(max(route_nodes[i], 0), n_nodes - 1)
            director_idx = min(max(node, 0), n_directors - 1)
            point = (
                position_collection[:, node]
                + np.ascontiguousarray(director_collection[..., director_idx].T)
                @ np.ascontiguousarray(vertebra_height_vector)
            )
            if i > 0:
                total += np.linalg.norm(point - previous)
            previous = point
        return total


    def apply_torques(self, system: SystemType, time: np.float64 = 0.0):
        # The force_data set and vertebra_weight_vector are expressed in the global coordinate frame and must be changed to local reference frames for torque application
        # Creating the array which will contain the transformed force vectors
        transformed_force_data = np.zeros((len(self.vertebra_nodes), 3), dtype=np.float64)

        # Transforming the force vectors calculated in the compute_forces method from the global reference frame to the local reference frame
        for i in range(len(self.vertebra_nodes)):
            transformed_force_data[i] = system.director_collection[...,(self.vertebra_nodes[i]-1)] @ self.force_data[i]
        transformed_anchor_force = np.zeros(3, dtype=np.float64)
        anchor_element_idx = -1
        if self.distal_anchor_node >= 0:
            anchor_node = min(max(self.distal_anchor_node, 0), self.n_elements)
            anchor_element_idx = min(max(anchor_node - 1, 0), self.n_elements - 1)
            distal_unit = self.get_rotations(
                np.array(system.position_collection),
                np.array(system.director_collection),
                np.array(self.vertebra_nodes),
                self.vertebra_height_vector,
                self.distal_anchor_node,
            )[-1]
            distal_anchor_force = -distal_unit * self._get_current_tension()
            transformed_anchor_force = system.director_collection[..., anchor_element_idx] @ distal_anchor_force

        self.compute_torques(
            self.vertebra_height_vector, np.array(self.vertebra_nodes), transformed_force_data,
            self.n_elements, system.external_torques,
            anchor_element_idx, transformed_anchor_force,
        )

        if self.debug_store is not None:
            torque_local = np.cross(
                self.vertebra_height_vector.reshape(1, 3),
                transformed_force_data,
            )
            torque_global = np.zeros_like(torque_local)
            for i, node in enumerate(self.vertebra_nodes):
                element_idx = min(max(int(node) - 1, 0), system.director_collection.shape[2] - 1)
                torque_global[i] = system.director_collection[..., element_idx].T @ torque_local[i]
            self.debug_store["computed_torque_local"] = torque_local
            self.debug_store["computed_torque_global"] = torque_global


    @staticmethod
    @njit(cache=True)
    def get_rotations(position_collection, director_collection, vertebra_nodes, vertebra_height_vector, distal_anchor_node):
        # Returns an array containing the unit norm vector which describes the orientation of each segment of tendon between vertebrae

        # Initializing unit_norm_vector_array to store the unit normed vectors that describe the global orientation of the forces in each vertebra
        unit_norm_vector_array = np.zeros((len(vertebra_nodes) + 1, 3), dtype=np.float64) # +1 modified
        n_nodes = position_collection.shape[1]
        n_directors = director_collection.shape[2]

        for i in range(len(vertebra_nodes)+1):
            # There is a +1 in the for loop to account for the force between the first vertebra and the fixed node

            # If statement, used for the case when i = 0 and thus there is no vertebra before this one, same for the final vertebra (no vertebra after that one)
            if i==0:
                current_vertebra = 0
                next_vertebra = vertebra_nodes[i]
            elif i==len(vertebra_nodes):
                current_vertebra = vertebra_nodes[i-1]
                if distal_anchor_node >= 0:
                    next_vertebra = min(max(distal_anchor_node, 0), n_nodes - 1)
                else:
                    next_vertebra = vertebra_nodes[i-1]
            else:
                current_vertebra = vertebra_nodes[i-1]
                next_vertebra = vertebra_nodes[i]

            # Setting up values to be used iteratively
            x_current = position_collection[0, current_vertebra]
            y_current = position_collection[1, current_vertebra]
            z_current = position_collection[2, current_vertebra]

            x_next = position_collection[0, next_vertebra]
            y_next = position_collection[1, next_vertebra]
            z_next = position_collection[2, next_vertebra]

            current_director_idx = min(max(current_vertebra, 0), n_directors - 1)
            next_director_idx = min(max(next_vertebra, 0), n_directors - 1)
            current_rotation_matrix = director_collection[...,current_director_idx]
            next_rotation_matrix = director_collection[...,next_director_idx]

            current_node = np.array([x_current, y_current, z_current])
            next_node = np.array([x_next, y_next, z_next])

            # Calculating relative position vector between vertebrae, considering the vertebra height
            # Continguous arrays to help with computation speed
            delta_vector = (next_node + np.ascontiguousarray(next_rotation_matrix.T) @ np.ascontiguousarray(vertebra_height_vector)) - (current_node + np.ascontiguousarray(current_rotation_matrix.T) @ np.ascontiguousarray(vertebra_height_vector))

            # Calculating the unit-normed vector based on the differences calculated in the previous step
            delta_vector_norm = np.linalg.norm(delta_vector)
            if delta_vector_norm < 1e-12:
                unit_norm_delta_vector = np.zeros(3)
            else:
                unit_norm_delta_vector = delta_vector / delta_vector_norm

            # Legacy behavior: if no distal anchor is requested, the final
            # tendon segment is zero. With distal_anchor_node>=0, the tendon
            # continues from the last vertebra to that anchor node.
            if i==len(vertebra_nodes) and distal_anchor_node < 0:
                unit_norm_delta_vector = np.zeros(3)

            # Storing the unit normed vector to be later used in the compute_forces method
            unit_norm_vector_array[i] = unit_norm_delta_vector

        return unit_norm_vector_array

    @staticmethod
    @njit(cache=True)
    def compute_forces(tension, vertebra_nodes, unit_norm_vector_array):

        # Creating array to store forces in vertebrae
        force_data = np.zeros((len(vertebra_nodes), 3), dtype=np.float64)

        for i in range(len(vertebra_nodes)):
            # This for loop multiplies the unit normed vectors calculated previously, with the tension of the tendon, thus creating the force vector for each vertebra
            # Contiguous array to increase speed in njit decorator
            force_current_prev = unit_norm_vector_array[i] * -tension
            force_current_next = unit_norm_vector_array[i+1] * tension

            # Summing the components of both force vectors to get the final force vector, which is then stored for use in the apply_forces and compute_torques methods
            force_data[i] = force_current_prev + force_current_next

        return force_data


    @staticmethod
    @njit(cache=True)
    def compute_torques(vertebra_height_vector, vertebra_nodes, transformed_force_data, n_elements, external_torques, anchor_element_idx, transformed_anchor_force):

        # Creating torque data set for storage
        torque_data = np.zeros((len(vertebra_nodes), 3),dtype=np.float64)

        # Goes through vertebra nodes to calculate torques for them
        for i in range(len(vertebra_nodes)):

            # Cross product between the vertebra height vector and the local force vector due to the tendons, to obtain the tendon torque for that vertebra
            torque_vector = np.cross(vertebra_height_vector, transformed_force_data[i])

            # Sum of the vectors, and storage into the torque_data array
            torque_data[i] = torque_vector

        # Appending the computed torque vector to the final torque data set
        apply_torque = np.zeros((3,n_elements+1))

        k = 0
        for i in range(n_elements):
            if i in vertebra_nodes:
                apply_torque[:,i] = torque_data[k]
                k += 1
        apply_torque = apply_torque[:,1:]

        # Applying the torque data set to the rod (torque on the final vertebra)
        external_torques += apply_torque
        if anchor_element_idx >= 0:
            external_torques[:, anchor_element_idx] += np.cross(vertebra_height_vector, transformed_anchor_force)
