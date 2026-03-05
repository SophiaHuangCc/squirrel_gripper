"""
Metrics framework for analyzing grasp quality from contact logs.

This module provides functions to:
- Read contact log CSV files
- Extract contact data for specific frames
- Compute force closure metrics
"""

import numpy as np
import csv
from pathlib import Path
from typing import Optional, Tuple, Dict, Union, List
import matplotlib.pyplot as plt


def read_contact_log(csv_path: Union[str, Path]) -> List[Dict]:
    """
    Read contact log CSV file.
    
    Args:
        csv_path: Path to contact_log.csv file
        
    Returns:
        List of dictionaries with keys including: frame_idx, time, node_idx, 
        node_x, node_y, node_z, radial_dist, overlap, normal_force, 
        normal_velocity, tangential_speed, friction_force,
        cyl_center_x, cyl_center_y, cyl_center_z, cyl_axis_x, cyl_axis_y, 
        cyl_axis_z, cyl_radius
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Contact log not found: {csv_path}")
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['frame_idx'] = int(row['frame_idx'])
            row['time'] = float(row['time'])
            row['node_idx'] = int(row['node_idx'])
            
            # Contact node position (new fields)
            if 'node_x' in row:
                row['node_x'] = float(row['node_x'])
                row['node_y'] = float(row['node_y'])
                row['node_z'] = float(row['node_z'])
            
            row['radial_dist'] = float(row['radial_dist'])
            row['overlap'] = float(row['overlap'])
            row['normal_force'] = float(row['normal_force'])
            row['normal_velocity'] = float(row['normal_velocity'])
            row['tangential_speed'] = float(row['tangential_speed'])
            row['friction_force'] = float(row['friction_force'])
            
            # Cylinder geometry (new fields)
            if 'cyl_center_x' in row:
                row['cyl_center_x'] = float(row['cyl_center_x'])
                row['cyl_center_y'] = float(row['cyl_center_y'])
                row['cyl_center_z'] = float(row['cyl_center_z'])
                row['cyl_axis_x'] = float(row['cyl_axis_x'])
                row['cyl_axis_y'] = float(row['cyl_axis_y'])
                row['cyl_axis_z'] = float(row['cyl_axis_z'])
                row['cyl_radius'] = float(row['cyl_radius'])
            
            data.append(row)
    
    return data


def get_frame_contacts(data: List[Dict], frame_idx: Optional[int] = None) -> List[Dict]:
    """
    Extract contact data for a specific frame.
    
    Args:
        data: List of contact dictionaries from read_contact_log()
        frame_idx: Frame index to extract. If None, uses the last frame.
        
    Returns:
        List of contact dictionaries filtered to the specified frame
    """
    if frame_idx is None:
        frame_idx = max(row['frame_idx'] for row in data)
    
    frame_data = [row for row in data if row['frame_idx'] == frame_idx]
    return frame_data


def check_force_closure_from_csv(
    contact_data: List[Dict],
    min_angular_span: float = 180.0,
    min_contacts: int = 3,
) -> Tuple[bool, Dict]:
    """
    Check force closure from contact log data.
    
    Requires CSV data with node positions and cylinder geometry (new format).
    
    Args:
        contact_data: List of contact dictionaries for a single frame
        min_angular_span: Minimum angular span in degrees for force closure
        min_contacts: Minimum number of contacts required
        
    Returns:
        (is_force_closure, metrics_dict) where metrics_dict contains:
        - num_contacts: number of contact points
        - angular_span: angular span of contacts in degrees
        - total_normal_force: sum of normal forces
        - max_normal_force: maximum normal force
        - total_friction_force: sum of friction forces
        - contact_node_indices: array of node indices in contact
        - normal_forces: array of normal force magnitudes
        
    Raises:
        ValueError: If required fields (node positions, cylinder geometry) are missing
    """
    if len(contact_data) == 0:
        return False, {
            "num_contacts": 0,
            "angular_span": 0.0,
            "total_normal_force": 0.0,
            "max_normal_force": 0.0,
            "total_friction_force": 0.0,
            "contact_node_indices": np.array([]),
            "normal_forces": np.array([]),
        }
    
    # Validate required fields are present
    required_fields = ['node_x', 'node_y', 'node_z', 
                       'cyl_center_x', 'cyl_center_y', 'cyl_center_z',
                       'cyl_axis_x', 'cyl_axis_y', 'cyl_axis_z', 'cyl_radius']
    missing_fields = [field for field in required_fields if field not in contact_data[0]]
    if missing_fields:
        raise ValueError(
            f"Missing required fields in contact log: {missing_fields}. "
            "Please use the new CSV format with node positions and cylinder geometry."
        )
    
    # Extract basic contact data
    contact_node_indices = np.array([row['node_idx'] for row in contact_data])
    normal_forces = np.array([row['normal_force'] for row in contact_data])
    friction_forces = np.array([row['friction_force'] for row in contact_data])
    
    num_contacts = len(contact_node_indices)
    total_normal_force = np.sum(normal_forces)
    max_normal_force = np.max(normal_forces) if len(normal_forces) > 0 else 0.0
    total_friction_force = np.sum(friction_forces)
    
    # Basic checks
    has_sufficient_contacts = num_contacts >= min_contacts
    has_sufficient_force = total_normal_force > 0.0
    
    # Extract contact positions from CSV
    contact_positions = np.array([
        [row['node_x'], row['node_y'], row['node_z']]
        for row in contact_data
    ]).T  # (3, n_contacts)
    
    # Extract cylinder geometry from CSV (all rows should have the same values)
    first_row = contact_data[0]
    cyl_center = np.array([
        first_row['cyl_center_x'],
        first_row['cyl_center_y'],
        first_row['cyl_center_z']
    ])
    cyl_axis = np.array([
        first_row['cyl_axis_x'],
        first_row['cyl_axis_y'],
        first_row['cyl_axis_z']
    ])
    
    # Compute angular span
    angular_span = compute_angular_span_from_positions(
        contact_positions,
        cyl_center,
        cyl_axis,
    )
    
    has_sufficient_span = angular_span >= min_angular_span
    is_force_closure = has_sufficient_contacts and has_sufficient_force and has_sufficient_span
    
    metrics = {
        "num_contacts": num_contacts,
        "angular_span": angular_span,
        "total_normal_force": float(total_normal_force),
        "max_normal_force": float(max_normal_force),
        "total_friction_force": float(total_friction_force),
        "contact_node_indices": contact_node_indices,
        "normal_forces": normal_forces,
    }
    
    return is_force_closure, metrics


def compute_angular_span_from_positions(
    contact_positions: np.ndarray,  # (3, n_contacts)
    cyl_center: np.ndarray,         # (3,)
    cyl_axis: np.ndarray,           # (3,)
) -> float:
    """
    Compute the angular span of contacts around the cylinder from contact positions.
    
    Args:
        contact_positions: (3, n_contacts) array of contact node positions
        cyl_center: (3,) array of cylinder center position
        cyl_axis: (3,) array of cylinder axis direction
        
    Returns:
        Angular span in degrees (0-360)
    """
    if contact_positions.shape[1] < 2:
        return 0.0
    
    # Compute vectors from cylinder center to contact points
    rel_vecs = contact_positions.T - cyl_center[None, :]  # (n_contacts, 3)
    
    # Project onto plane perpendicular to cylinder axis
    cyl_axis_norm = cyl_axis / (np.linalg.norm(cyl_axis) + 1e-12)
    proj_lengths = np.dot(rel_vecs, cyl_axis_norm)  # (n_contacts,)
    proj_vecs = np.outer(proj_lengths, cyl_axis_norm)  # (n_contacts, 3)
    radial_vecs = rel_vecs - proj_vecs  # (n_contacts, 3)
    
    # Compute radial distances and unit vectors
    radial_dists = np.linalg.norm(radial_vecs, axis=1)  # (n_contacts,)
    mask = radial_dists > 1e-12
    if not np.any(mask):
        return 0.0
    
    radial_unit = np.zeros_like(radial_vecs)
    radial_unit[mask] = radial_vecs[mask] / radial_dists[mask, None]
    
    # Choose a reference direction perpendicular to cylinder axis
    ref_idx = np.where(mask)[0][0]
    ref_dir = radial_unit[ref_idx]
    
    # Build a coordinate system in the plane perpendicular to cylinder axis
    y_axis = np.cross(cyl_axis_norm, ref_dir)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    x_axis = np.cross(y_axis, cyl_axis_norm)
    
    # Compute angles of all contact points around the cylinder
    n_contacts = contact_positions.shape[1]
    contact_angles = np.zeros(n_contacts)
    for i in range(n_contacts):
        if mask[i]:
            x_component = np.dot(radial_unit[i], x_axis)
            y_component = np.dot(radial_unit[i], y_axis)
            angle_rad = np.arctan2(y_component, x_component)
            contact_angles[i] = np.degrees(angle_rad)
        else:
            contact_angles[i] = np.nan
    
    # Remove NaN angles and sort
    valid_angles = contact_angles[mask]
    if len(valid_angles) < 2:
        return 0.0
    
    # Sort angles and compute span
    sorted_angles = np.sort(valid_angles)
    
    # Compute angular span (accounting for wrap-around)
    gaps = np.diff(sorted_angles)
    wrap_gap = (360.0 + sorted_angles[0]) - sorted_angles[-1]
    gaps = np.append(gaps, wrap_gap)
    
    # Angular span is 360 minus the largest gap
    max_gap = np.max(gaps)
    angular_span = 360.0 - max_gap
    
    return float(angular_span)


def compute_angular_span(
    contact_node_indices: np.ndarray,
    rod_positions: np.ndarray,  # (3, n_nodes)
    cyl_center: np.ndarray,     # (3,)
    cyl_axis: np.ndarray,       # (3,)
) -> float:
    """
    Compute the angular span of contacts around the cylinder.
    
    Args:
        contact_node_indices: Array of node indices in contact
        rod_positions: (3, n_nodes) array of rod node positions
        cyl_center: (3,) array of cylinder center position
        cyl_axis: (3,) array of cylinder axis direction
        
    Returns:
        Angular span in degrees (0-360)
    """
    if len(contact_node_indices) < 2:
        return 0.0
    
    # Get contact positions
    contact_positions = rod_positions[:, contact_node_indices]  # (3, n_contacts)
    
    # Compute vectors from cylinder center to contact points
    rel_vecs = contact_positions.T - cyl_center[None, :]  # (n_contacts, 3)
    
    # Project onto plane perpendicular to cylinder axis
    cyl_axis_norm = cyl_axis / (np.linalg.norm(cyl_axis) + 1e-12)
    proj_lengths = np.dot(rel_vecs, cyl_axis_norm)  # (n_contacts,)
    proj_vecs = np.outer(proj_lengths, cyl_axis_norm)  # (n_contacts, 3)
    radial_vecs = rel_vecs - proj_vecs  # (n_contacts, 3)
    
    # Compute radial distances and unit vectors
    radial_dists = np.linalg.norm(radial_vecs, axis=1)  # (n_contacts,)
    mask = radial_dists > 1e-12
    if not np.any(mask):
        return 0.0
    
    radial_unit = np.zeros_like(radial_vecs)
    radial_unit[mask] = radial_vecs[mask] / radial_dists[mask, None]
    
    # Choose a reference direction perpendicular to cylinder axis
    ref_idx = np.where(mask)[0][0]
    ref_dir = radial_unit[ref_idx]
    
    # Build a coordinate system in the plane perpendicular to cylinder axis
    y_axis = np.cross(cyl_axis_norm, ref_dir)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    x_axis = np.cross(y_axis, cyl_axis_norm)
    
    # Compute angles of all contact points around the cylinder
    contact_angles = np.zeros(len(contact_node_indices))
    for i in range(len(contact_node_indices)):
        if mask[i]:
            x_component = np.dot(radial_unit[i], x_axis)
            y_component = np.dot(radial_unit[i], y_axis)
            angle_rad = np.arctan2(y_component, x_component)
            contact_angles[i] = np.degrees(angle_rad)
        else:
            contact_angles[i] = np.nan
    
    # Remove NaN angles and sort
    valid_angles = contact_angles[mask]
    if len(valid_angles) < 2:
        return 0.0
    
    # Sort angles and compute span
    sorted_angles = np.sort(valid_angles)
    
    # Compute angular span (accounting for wrap-around)
    gaps = np.diff(sorted_angles)
    wrap_gap = (360.0 + sorted_angles[0]) - sorted_angles[-1]
    gaps = np.append(gaps, wrap_gap)
    
    # Angular span is 360 minus the largest gap
    max_gap = np.max(gaps)
    angular_span = 360.0 - max_gap
    
    return float(angular_span)


def analyze_grasp_from_log(
    csv_path: Union[str, Path],
    frame_idx: Optional[int] = None,
    min_angular_span: float = 180.0,
    min_contacts: int = 3,
) -> Tuple[bool, Dict]:
    """
    Complete workflow: read contact log and compute force closure.
    
    Requires CSV with node positions and cylinder geometry (new format).
    
    Args:
        csv_path: Path to contact_log.csv (must contain node positions and cylinder geometry)
        frame_idx: Frame to analyze (None = last frame)
        min_angular_span: Minimum angular span in degrees
        min_contacts: Minimum number of contacts required
        
    Returns:
        (is_force_closure, metrics_dict)
        
    Raises:
        ValueError: If required fields are missing from the CSV
    """
    data = read_contact_log(csv_path)
    frame_data = get_frame_contacts(data, frame_idx)
    
    is_fc, metrics = check_force_closure_from_csv(
        frame_data,
        min_angular_span=min_angular_span,
        min_contacts=min_contacts,
    )
    
    return is_fc, metrics


def analyze_stable_closure_from_log(
    csv_path: Union[str, Path],
    frame_idx: Optional[int] = None,
    rod_mass: float = 0.1,      # in kg
    v_mass_total: float = 0.2,  # in kg
    args: Optional[object] = None
) -> Tuple[bool, Dict]:
    data = read_contact_log(csv_path)
    frame_data = get_frame_contacts(data, frame_idx)
    
    is_stable, stab_metrics = check_stable_closure_hanging(
        frame_data, 
        finger_mass=(rod_mass + v_mass_total),
        body_mass=args.body_mass
    )
    
    return is_stable, stab_metrics


def plot_contacts_2d(
    contact_data: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    figsize: Tuple[float, float] = (8, 8),
) -> None:
    """
    Generate a 2D plot of contact points projected onto a plane perpendicular to the cylinder axis.
    
    Args:
        contact_data: List of contact dictionaries for a single frame
        output_path: Optional path to save the plot (e.g., "contacts_2d.png")
        show_plot: Whether to display the plot
        figsize: Figure size (width, height) in inches
    """
    if len(contact_data) == 0:
        print("No contact data to plot")
        return
    
    # Validate required fields
    required_fields = ['node_x', 'node_y', 'node_z', 
                       'cyl_center_x', 'cyl_center_y', 'cyl_center_z',
                       'cyl_axis_x', 'cyl_axis_y', 'cyl_axis_z', 'cyl_radius']
    missing_fields = [field for field in required_fields if field not in contact_data[0]]
    if missing_fields:
        raise ValueError(
            f"Missing required fields in contact log: {missing_fields}. "
            "Please use the new CSV format with node positions and cylinder geometry."
        )
    
    # Extract contact positions
    contact_positions = np.array([
        [row['node_x'], row['node_y'], row['node_z']]
        for row in contact_data
    ]).T  # (3, n_contacts)
    
    # Extract cylinder geometry
    first_row = contact_data[0]
    cyl_center = np.array([
        first_row['cyl_center_x'],
        first_row['cyl_center_y'],
        first_row['cyl_center_z']
    ])
    cyl_axis = np.array([
        first_row['cyl_axis_x'],
        first_row['cyl_axis_y'],
        first_row['cyl_axis_z']
    ])
    cyl_radius = first_row['cyl_radius']
    
    # Extract normal forces for color coding
    normal_forces = np.array([row['normal_force'] for row in contact_data])
    
    # Align 2D plot with video front view (azim=-90, elev=0): horizontal=X, vertical=Z.
    # Use coordinates relative to cylinder center so the circle cross-section remains centered at (0,0).
    rel_vecs = contact_positions.T - cyl_center[None, :]  # (n_contacts, 3)
    x_coords = rel_vecs[:, 0]  # world X
    y_coords = rel_vecs[:, 2]  # world Z
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw cylinder cross-section as a circle
    circle = plt.Circle((0, 0), cyl_radius, fill=False, color='black', 
                        linestyle='--', linewidth=2, label='Cylinder cross-section')
    ax.add_patch(circle)
    
    # Plot contact points, color-coded by normal force
    if len(normal_forces) > 0 and np.max(normal_forces) > 0:
        scatter = ax.scatter(x_coords, y_coords, c=normal_forces, s=100, 
                            cmap='viridis', alpha=0.7, edgecolors='black', 
                            linewidths=1, label='Contact points')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Normal Force (N)', rotation=270, labelpad=20)
    else:
        ax.scatter(x_coords, y_coords, s=100, c='red', alpha=0.7, 
                  edgecolors='black', linewidths=1, label='Contact points')
    
    # Draw lines from center to contact points
    for i in range(len(x_coords)):
        ax.plot([0, x_coords[i]], [0, y_coords[i]], 'gray', alpha=0.3, linewidth=0.5)
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('Contact Points (Front View: X-Z)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add text with summary info
    num_contacts = len(contact_data)
    total_force = np.sum(normal_forces) if len(normal_forces) > 0 else 0.0
    info_text = f'Contacts: {num_contacts}\nTotal Force: {total_force:.3f} N'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set reasonable axis limits
    max_dist = max(np.max(np.abs(x_coords)), np.max(np.abs(y_coords)), cyl_radius * 1.2) if len(x_coords) > 0 else cyl_radius * 1.2
    ax.set_xlim(-max_dist * 1.1, max_dist * 1.1)
    ax.set_ylim(-max_dist * 1.1, max_dist * 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 2D contact plot to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_contacts_2d_from_log(
    csv_path: Union[str, Path],
    frame_idx: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> None:
    """
    Convenience function to plot contacts from a CSV file.
    
    Args:
        csv_path: Path to contact_log.csv
        frame_idx: Frame to plot (None = last frame)
        output_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    data = read_contact_log(csv_path)
    frame_data = get_frame_contacts(data, frame_idx)
    plot_contacts_2d(frame_data, output_path=output_path, show_plot=show_plot)


if __name__ == "__main__":
    # Example usage
    import sys
    
    csv_path = "contact_log.csv"
    plot_output = None
    show_plot = False
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        plot_output = sys.argv[2]  # Optional: output path for plot
        show_plot = False
    if len(sys.argv) > 3 and sys.argv[3].lower() == 'show':
        show_plot = True
    
    print(f"Analyzing contact log: {csv_path}")
    is_fc, metrics = analyze_grasp_from_log(csv_path)
    
    print(f"\n[FORCE CLOSURE] {'ACHIEVED' if is_fc else 'NOT ACHIEVED'}")
    print(f"  Contacts: {metrics['num_contacts']}")
    print(f"  Angular span: {metrics['angular_span']:.1f}°")
    print(f"  Total normal force: {metrics['total_normal_force']:.6f} N")
    print(f"  Max normal force: {metrics['max_normal_force']:.6f} N")
    print(f"  Total friction force: {metrics['total_friction_force']:.6f} N")
    
    # Generate 2D plot
    if plot_output or show_plot:
        print(f"\nGenerating 2D contact plot...")
        plot_contacts_2d_from_log(csv_path, output_path=plot_output, show_plot=show_plot)

    
def check_stable_closure_hanging(
    contact_data: List[Dict],
    finger_mass: float,  # in kg
    body_mass: float = 0.5 # Squirrel body in kg
) -> Tuple[bool, Dict]:
    """
    Evaluates if the vertical upward forces (Normal_z + Friction_z) 
    exceed the total weight of the squirrel.
    """
    if len(contact_data) == 0:
        return False, {"margin": 0.0, "support": 0.0, "load": 0.0}

    # 1. Calculate the Load (Gravity pulling down)
    g = 9.80665
    total_weight = (finger_mass + body_mass) * g
    
    # 2. Calculate Vertical Support
    # Note: In the simulation, friction_force is a magnitude. 
    # To get the vertical COMPONENT of friction, we assume it acts 
    # purely against gravity in the +Z direction if sliding occurs.
    
    total_upward_support = 0.0
    
    for row in contact_data:
        # Normal Force Component:
        # Direction is from cylinder center to node
        dz = row['node_z'] - row['cyl_center_z']
        radial_dist = row['radial_dist']
        
        # Unit vector z-component
        nz = dz / (radial_dist + 1e-12)
        normal_z_contribution = row['normal_force'] * nz
        
        # Friction Component:
        # Friction is tangential. In a static hanging case, friction 
        # is primarily fighting gravity, so we take its full magnitude 
        # as a potential upward force (+Z).
        friction_z_contribution = row['friction_force']
        
        total_upward_support += (normal_z_contribution + friction_z_contribution)

    is_stable = total_upward_support >= total_weight
    margin = total_upward_support / total_weight if total_weight > 0 else 0.0
    
    return is_stable, {
        "is_stable": is_stable,
        "margin": margin,
        "total_support_n": total_upward_support,
        "total_load_n": total_weight
    }