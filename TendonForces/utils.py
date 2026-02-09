import numpy as np

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
    dx = rod_pos[0, :] - cyl_center[0]
    dz = rod_pos[2, :] - cyl_center[2]
    radial_dist = np.sqrt(dx**2 + dz**2)

    contact_threshold = cyl_radius + base_radius # 0.020
    overlaps = contact_threshold - radial_dist

    contact_mask = overlaps > 0.0
    
    normal_vec = np.zeros((rod_pos.shape[1], 3))
    normal_vec[:, 0] = dx
    normal_vec[:, 2] = dz
    
    norms = np.linalg.norm(normal_vec, axis=1)
    mask = norms > 1e-12
    normal_vec[mask] /= norms[mask, None]

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


