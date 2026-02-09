import numpy as np
import sys
from elastica.callback_functions import CallBackBaseClass

class SquirrelCallback(CallBackBaseClass):
    def __init__(self, step_skip, callback_params):
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step):
        if np.isnan(system.position_collection).any():
            print(f"[ERROR] NaN at step {current_step}, time {time:.4f}")
            sys.exit(1)
        
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["acceleration"].append(system.acceleration_collection.copy())
            self.callback_params["omega"].append(system.omega_collection.copy())
            self.callback_params["alpha"].append(system.alpha_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["lengths"].append(system.lengths.copy())
            self.callback_params["tangents"].append(system.tangents.copy())
            self.callback_params["internal_forces"].append(system.internal_forces.copy())
            self.callback_params["internal_torques"].append(system.internal_torques.copy())
            self.callback_params["external_forces"].append(system.external_forces.copy())
            self.callback_params["external_torques"].append(system.external_torques.copy())
            self.callback_params["sigma"].append(system.sigma.copy())
            self.callback_params["kappa"].append(system.kappa.copy())
            self.callback_params["internal_stress"].append(system.internal_stress.copy())
            self.callback_params["internal_couple"].append(system.internal_couple.copy())
            self.callback_params["dilatation"].append(system.dilatation.copy())
            self.callback_params["dilatation_rate"].append(system.dilatation_rate.copy())
            self.callback_params["voronoi_dilatation"].append(system.voronoi_dilatation.copy())