"""Direction-only contact-support metrics independent of the simulator CLI."""

import numpy as np


def contact_normals_from_geometry(rod_pos, cyl_center, cyl_radius, base_radius):
    dx = rod_pos[0, :] - cyl_center[0]
    dz = rod_pos[2, :] - cyl_center[2]
    radial = np.sqrt(dx**2 + dz**2)
    contact = radial < (cyl_radius + base_radius)
    valid = contact & (radial > 1e-12)
    normals = np.zeros((int(np.sum(valid)), 3), dtype=float)
    normals[:, 0] = dx[valid] / radial[valid]
    normals[:, 2] = dz[valid] / radial[valid]
    return normals


def directional_contact_support_score(contact_normals, disturbance_direction, friction_mu=0.0):
    """Score whether contact-normal/friction rays can oppose a fixed direction."""
    normals = np.asarray(contact_normals, dtype=float)
    if normals.size == 0:
        return 0.0
    normals = normals.reshape(-1, 3)
    disturbance = np.asarray(disturbance_direction, dtype=float).reshape(3)
    target = -disturbance[[0, 2]]
    target_norm = np.linalg.norm(target)
    if target_norm <= 1e-12:
        return 0.0
    target /= target_norm
    radial = normals[:, [0, 2]]
    lengths = np.linalg.norm(radial, axis=1)
    radial = radial[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    if len(radial) == 0:
        return 0.0
    tangent = np.stack((-radial[:, 1], radial[:, 0]), axis=1)
    mu = max(0.0, float(friction_mu))
    rays = np.concatenate((radial, radial + mu * tangent, radial - mu * tangent), axis=0)
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    best_residual = 1.0
    for ray in rays:
        coefficient = max(0.0, float(np.dot(ray, target)))
        best_residual = min(best_residual, float(np.linalg.norm(coefficient * ray - target)))
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            matrix = np.stack((rays[i], rays[j]), axis=1)
            if abs(float(np.linalg.det(matrix))) <= 1e-10:
                continue
            if np.all(np.linalg.solve(matrix, target) >= -1e-10):
                best_residual = 0.0
                break
        if best_residual == 0.0:
            break
    return float(np.clip(1.0 - best_residual, 0.0, 1.0))


def directional_projected_force_stats(contact_forces, disturbance_direction, reference_force):
    """Return usable force opposing one disturbance and a smooth normalized score."""
    forces = np.asarray(contact_forces, dtype=float)
    if forces.size == 0:
        return {
            "opposing_force": 0.0, "mean_opposing_force": 0.0,
            "total_force": 0.0, "force_score": 0.0, "num_contacts": 0,
        }
    forces = forces.reshape(-1, 3)
    disturbance = np.asarray(disturbance_direction, dtype=float).reshape(3)
    magnitude = float(np.linalg.norm(disturbance))
    if magnitude <= 1e-12:
        raise ValueError("disturbance_direction must be nonzero")
    target = -disturbance / magnitude
    # Sum useful per-contact projections before transverse components cancel.
    opposing = np.maximum(forces @ target, 0.0)
    opposing_force = float(np.sum(opposing))
    reference = max(float(reference_force), 1e-12)
    return {
        "opposing_force": opposing_force,
        "mean_opposing_force": float(np.mean(opposing)),
        "total_force": float(np.linalg.norm(forces, axis=1).sum()),
        "force_score": opposing_force / (opposing_force + reference),
        "num_contacts": int(len(forces)),
    }
