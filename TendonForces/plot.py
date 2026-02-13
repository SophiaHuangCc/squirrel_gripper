import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip
from .utils import draw_cylinder # Ensure draw_cylinder is available in your utils

def generate_simulation_video(video_path, data, cylinder, args, v_nodes, fps, final_time, body_weight_force):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    pos_data = data["position"]
    force_data = data["external_forces"]

    def make_frame(t):
        # Correctly map time to frame index
        frame_idx = min(int(t * fps), len(pos_data) - 1)
        ax.clear()
        
        P = pos_data[frame_idx]
        F = force_data[frame_idx]
        
        # 1. Draw the Rod (Main nodes)
        ax.scatter(P[0], P[1], P[2], s=6, alpha=0.8, color='blue', label='Rod Nodes')
        
        # 2. Draw Vertebrae (Red dots)
        # Use clipping to ensure v_nodes are within bounds of current P
        v_idx_clean = np.clip(v_nodes, 0, P.shape[1] - 1).astype(int)
        ax.scatter(
            P[0, v_idx_clean], P[1, v_idx_clean], P[2, v_idx_clean],
            color="red",
            s=25,
            depthshade=False,
            zorder=10,
            label='Vertebrae'
        )

        # 3. Force Visualizations (External Forces)
        if args.debug:
            mag = np.linalg.norm(F, axis=0)
            force_scale = 0.02
            step_nodes = 5 # Visual spacing for arrows

            # General magenta arrows for all contact/external forces
            for i in range(0, F.shape[1], step_nodes):
                if mag[i] > 1e-6:
                    ax.quiver(
                        P[0, i], P[1, i], P[2, i],
                        F[0, i], F[1, i], F[2, i],
                        length=force_scale,
                        normalize=True,
                        color="magenta",
                        alpha=0.6
                    )

            # Specific Cyan arrows for forces acting directly on vertebrae
            for v_idx in v_idx_clean:
                f_vec = F[:, v_idx]
                if np.linalg.norm(f_vec) > 1e-5:
                    ax.quiver(
                        P[0, v_idx], P[1, v_idx], P[2, v_idx],
                        f_vec[0], f_vec[1], f_vec[2],
                        length=force_scale,
                        color="cyan",
                        linewidth=1.5,
                        zorder=11
                    )

        # 4. Body Weight Arrow (at the base/tensioning point)
        # We calculate base position by averaging the first 5 nodes
        base_pos = P[:, :5].mean(axis=1)
        scale_factor = 0.01 
        bw_vec = body_weight_force * scale_factor
        
        ax.quiver(
            base_pos[0], base_pos[1], base_pos[2],
            bw_vec[0], bw_vec[1], bw_vec[2],
            color='red', 
            linewidth=3, 
            label=f'Weight ({args.body_mass}kg)',
            arrow_length_ratio=0.3
        )

        # 5. Cylinder Visualization
        cyl_center = cylinder.position_collection[:, 0]
        cyl_axis_dir = cylinder.director_collection[2, :, 0]
        # Using args.cyl_rad and the cylinder length from simulation
        draw_cylinder(ax, cyl_center, cyl_axis_dir, args.cyl_rad, cylinder.length, color="black", alpha=0.3)

        # 6. Axis and View Limits
        ax.set_xlim(-0.02, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_zlim(-0.10, 0.10)
        
        # Labels for clarity in debug mode
        if args.debug:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')

        # Front view: from -Y looking toward +Y
        ax.view_init(elev=0, azim=-90)
        
        return mplfig_to_npimage(fig)

    # Generate the clip
    clip = VideoClip(make_frame, duration=final_time)
    clip.write_videofile(video_path, codec="libx264", fps=fps, logger=None)
    plt.close(fig)