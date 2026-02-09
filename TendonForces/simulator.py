import numpy as np
from elastica.modules import (BaseSystemCollection, Connections, Constraints, 
                              Forcing, CallBacks, Damping, Contact)
from elastica.rod.cosserat_rod import CosseratRod
from elastica.rigidbody.cylinder import Cylinder

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

def setup_rod_and_cylinder(args, start_pos, direction, normal, cyl_params):
    # Material properties
    E = args.E
    nu = args.poisson_nu
    G = E / (2 * (1 + nu))
    density = 1200
    mass_second_moment_of_inertia = 0.25 * np.pi * args.base_rad**4

    # Create Rod
    finger = CosseratRod.straight_rod(
        n_elements=args.n_elements,
        start=start_pos,
        direction=direction,
        normal=normal,
        base_length=args.base_len,
        base_radius=args.base_rad,
        density=density,
        youngs_modulus=E,
        shear_modulus=G,
        mass_second_moment_of_inertia=mass_second_moment_of_inertia,
    )

    # Apply soft joints
    if args.v_mode == "uniform":
        vertebra_nodes = np.linspace(args.v_start, args.v_end, args.num_v, dtype=int)
    else:
        vertebra_nodes = np.array([int(x) for x in args.v_list.split(",")], dtype=int)
        
    for j in vertebra_nodes:
        idx = int(np.clip(j, 0, finger.bend_matrix.shape[2] - 1))
        finger.bend_matrix[1, 1, idx] *= args.joint_softness
        finger.bend_matrix[2, 2, idx] *= args.joint_softness

    # Create Cylinder
    cylinder = Cylinder(
        start=cyl_params['start'],
        direction=cyl_params['direction'],
        normal=cyl_params['normal'],
        base_length=cyl_params['length'],
        base_radius=args.cyl_rad,
        density=1200.0,
    )

    return finger, cylinder, vertebra_nodes