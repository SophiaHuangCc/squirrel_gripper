import numpy as np
from grasp_utils import contact_forces_exist, random_pose_perturbation

def check_force_closure(vertices, normals, mu, num_facets=10):
    """
    Check whether a grasp is in force closure.
    
    Parameters
    ----------
    vertices : (N,3) np.ndarray
        Contact points.
    normals : (N,3) np.ndarray
        Surface normals at contacts.
    num_facets : int
        Number of vectors to approximate the friction cone.
    mu : float
        Friction coefficient.
    
    Returns
    -------
    is_fc : bool
        True if grasp is in force closure, False otherwise.
    """
    status, _ = contact_forces_exist(vertices, normals, mu, np.zeros(6), num_facets)
    return status

def compute_ferrari_canny(vertices, normals, mu, num_facets=10, num_samples=100):
    """
    Should return a score for the grasp according to the Ferrari Canny metric.

    Parameters
    ----------
    vertices (Nx3 np.ndarray): obj mesh vertices on which the fingers will be placed
    normals (Nx3 np.ndarray): obj mesh normals at the contact points
    num_facets (int): number of vectors to use to approximate the friction cone, vectors 
        will be along the friction cone boundary
    mu (float): coefficient of friction

    Returns
    -------
    (float): quality of the grasp
    """
    best_score = None
    for _ in range(num_samples):
        theta = np.random.rand() * 2 * np.pi
        w = np.array([0, np.cos(theta), np.sin(theta), 0, 0, 0])
    
        status, LQ = contact_forces_exist(vertices, normals, mu, w, num_facets)
        score = 1 / np.sqrt(LQ)
        if status:
            if best_score == None or score < best_score:
                best_score = score

    return best_score

def compute_robust_force_closure(vertices, normals, mu, num_facets=10, num_samples=100, std_trans=100, std_rot=100):
    """
    Returns a score for the grasp according to the robust force closure metric.

    Parameters
    ----------
    vertices (Nx3 np.ndarray): obj mesh vertices on which the fingers will be placed
    normals (Nx3 np.ndarray): obj mesh normals at the contact points
    num_facets (int): number of vectors to use to approximate the friction cone, vectors 
        will be along the friction cone boundary
    mu (float): coefficient of friction

    Returns
    -------
    (float): quality of the grasp
    """
    N = vertices.shape[0]
    success_count = 0

    for _ in range(num_samples):
        R, t = random_pose_perturbation(std_trans, std_rot)
        perturbed_vertices = (R @ vertices.T).T + t
        perturbed_normals = (R @ normals.T).T
        perturbed_normals /= np.linalg.norm(perturbed_normals, axis=1)[:, None]

        success_count += check_force_closure(perturbed_vertices, perturbed_normals, mu, num_facets)

    rfc_prob = success_count / num_samples
    return rfc_prob

def compute_max_torque(contact_positions, normals, forces, mu, rod_axis, rod_center):
    """
    Compute maximum torque along the rod axis, accounting for friction.
    
    Parameters
    ----------
    contact_positions : Nx3 array of contact points
    normals : Nx3 array of contact normals (unit vectors)
    f_n : array of normal forces at each contact
    mu : coefficient of friction
    rod_axis : 3-element unit vector along the rod
    rod_center : 3-element vector, pivot point for torque (center of rod)
    
    Returns
    -------
    scalar : maximum torque along rod axis
    """
    tau_total = np.zeros(3)
    
    for p, n, fn in zip(contact_positions, normals, forces):
        r = p - rod_center
        # Tangential direction perpendicular to lever arm and normal
        t_dir = np.cross(r, n)
        if np.linalg.norm(t_dir) > 1e-12:
            t_dir /= np.linalg.norm(t_dir)
        # Total force including friction-limited tangential component
        f_total = fn * n + mu * fn * t_dir
        tau_total += np.cross(r, f_total)
    
    # Project total torque onto rod axis
    rod_axis = rod_axis / np.linalg.norm(rod_axis)
    torque_along_rod = np.abs(tau_total @ rod_axis)
    
    return torque_along_rod

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
    if n_forces is not None and len(n_forces) > 0:
        U_contact = np.sum(0.5 * n_forces**2 / k)
    else:
        U_contact = 0.0

    U_total = U_elastic + U_contact
    return U_total