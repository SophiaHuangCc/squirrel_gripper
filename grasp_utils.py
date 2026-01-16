import numpy as np
import cvxpy as cvx

def normalize(vec):
    """
    Returns a normalized version of a numpy vector

    Parameters
    ----------
    vec (nx np.ndarray): vector to normalize

    Returns
    -------
    (nx np.ndarray): normalized vector
    """
    return vec / np.linalg.norm(vec)

def hat(v):
    """
    See https://en.wikipedia.org/wiki/Hat_operator or the MLS book

    Parameters
    ----------
    v (3x, 3x1, 6x, or 6x1 np.ndarray): vector to create hat matrix for

    Returns
    -------
    (3x3 or 6x6 np.ndarray): the hat version of the v
    """
    if v.shape == (3, 1) or v.shape == (3,):
        return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
    elif v.shape == (6, 1) or v.shape == (6,):
        return np.array([
                [0, -v[5], v[4], v[0]],
                [v[5], 0, -v[3], v[1]],
                [-v[4], v[3], 0, v[2]],
                [0, 0, 0, 0]
            ])
    else:
        raise ValueError

def adj(g):
    """
    Adjoint of a rotation matrix. See the MLS book.

    Parameters
    ----------
    g (4x4 np.ndarray): homogenous transform matrix

    Returns
    -------
    (6x6 np.ndarray): adjoint matrix
    """
    if g.shape != (4, 4):
        raise ValueError

    R = g[0:3,0:3]
    p = g[0:3,3]
    result = np.zeros((6, 6))
    result[0:3,0:3] = R
    result[0:3,3:6] = np.matmul(hat(p), R)
    result[3:6,3:6] = R
    return result

def look_at_general(origin, direction):
    """
    Creates a homogenous transformation matrix at the origin such that the 
    z axis is the same as the direction specified. There are infinitely 
    many of such matrices, but we choose the one where the y axis is as 
    vertical as possible.  

    Parameters
    ----------
    origin (3x np.ndarray): origin coordinates
    direction (3x np.ndarray): direction vector

    Returns
    -------
    (4x4 np.ndarray): homogenous transform matrix
    """
    up = np.array([0, 0, 1])
    z = normalize(direction)
    if np.allclose(z, up) or np.allclose(z, -up):
        x = np.array([1,0,0])
    else:
        x = normalize(np.cross(up, z))
    y = np.cross(z, x)

    result = np.eye(4)

    # set rotation part of matrix
    result[0:3,0] = x
    result[0:3,1] = y
    result[0:3,2] = z

    # set translation part of matrix to origin
    result[0:3,3] = origin

    return result

def random_pose_perturbation(std_trans=0.01, std_rot=0.04):
    """
    Generate a small random 3D pose perturbation.
    """
    t = np.random.randn(3) * std_trans
    
    # Rotation only around x-axis
    axis = np.array([1.0, 0.0, 0.0])
    angle = np.random.randn() * std_rot
    
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)
    
    return R, t

def get_grasp_map(vertices, normals):
    """ 
    Compute the grasp map given the contact points and their surface normals

    Parameters
    ----------
    vertices (2x3 np.ndarray): obj mesh vertices on which the fingers will be placed
    normals (2x3 np.ndarray): obj mesh normals at the contact points
    num_facets (int): number of vectors to use to approximate the friction cone, vectors 
        will be along the friction cone boundary
    mu (float): coefficient of friction

    Returns
    -------
    (np.ndarray): grasp map
    """
    N = vertices.shape[0]
    B = np.eye(6, 3)
    G_list = []

    for i in range(N):
        g = look_at_general(vertices[i], -normals[i])
        G_i = adj(np.linalg.inv(g)).T @ B
        G_list.append(G_i)

    G = np.hstack(G_list)
    return G

def contact_forces_exist(vertices, normals, mu, desired_wrench, num_facets=10):
    """
    Compute whether the given grasp (at contacts with surface normals) can produce 
    the desired_wrench. Will be used for gravity resistance.

    Parameters
    ----------
    vertices (Nx3 np.ndarray): obj mesh vertices on which the fingers will be placed
    normals (Nx3 np.ndarray): obj mesh normals at the contact points
    num_facets (int): number of vectors to use to approximate the friction cone, vectors 
        will be along the friction cone boundary
    mu (float): coefficient of friction
    desired_wrench (np.ndarray): potential wrench to be produced

    Returns
    -------
    (bool): whether contact forces can produce the desired_wrench on the object
    """
    assert vertices.shape[1] == 3
    assert normals.shape[1] == 3
    N = vertices.shape[0]

    # Friction cone vectors
    edges = np.array([[np.cos(2*np.pi*i/num_facets), 
                       np.sin(2*np.pi*i/num_facets), 
                       1/mu] for i in range(num_facets)]).T
    edges /= np.linalg.norm(edges, axis=0)  # normalize each column
    cone_vectors = edges           # 3 x num_facets

    G = get_grasp_map(vertices, normals)

    f = cvx.Variable(3*N)
    b = [cvx.Variable(num_facets) for _ in range(N)]

    objective = cvx.Minimize(cvx.norm2(G @ f - desired_wrench))
    constraints = []

    for i in range(N):
        idx_base = i*3
        constraints += [
            b[i] >= 0,
            f[idx_base:idx_base+3] == cone_vectors @ b[i],
            f[idx_base+2] >= 0
        ]

    prob = cvx.Problem(objective, constraints)
    prob.solve()

    if prob.status == 'optimal':
        return True, np.sum(f.value**2)
    else:
        return False, None