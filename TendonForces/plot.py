import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip
from utils import draw_cylinder

def generate_simulation_video(video_path, data, cylinder, args, v_nodes, fps, final_time, body_weight_force):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    pos_data = data["position"]
    force_data = data["external_forces"]

    def make_frame(t):
        frame_idx = min(int(t * fps), len(pos_data) - 1)
        ax.clear()
        P = pos_data[frame_idx]
        F = force_data[frame_idx]
        
        # Draw Rod with simple spheres
        ax.scatter(P[0], P[1], P[2], s=6)
        
        # Draw Vertebrae
        ax.scatter(P[0, v_nodes], P[1, v_nodes], P[2, v_nodes], color="red", s=20, zorder=10)

        # Force Quivers
        if args.debug:
            mag = np.linalg.norm(F, axis=0)
            for i in range(0, F.shape[1], 5):
                if mag[i] > 1e-6:
                    ax.quiver(P[0,i], P[1,i], P[2,i], F[0,i], F[1,i], F[2,i], 
                              length=0.02, color="magenta", normalize=True)

        # Body Weight Arrow
        base_pos = P[:, :5].mean(axis=1)
        ax.quiver(base_pos[0], base_pos[1], base_pos[2], 
                  body_weight_force[0]*0.01, body_weight_force[1]*0.01, body_weight_force[2]*0.01, 
                  color='red', linewidth=3)

        # Cylinder
        center = cylinder.position_collection[:, 0]
        axis_dir = cylinder.director_collection[2, :, 0]
        draw_cylinder(ax, center, axis_dir, args.cyl_rad, 0.20, color="black", alpha=0.35)

        ax.set_xlim(-0.02, 0.12); ax.set_ylim(-0.12, 0.12); ax.set_zlim(-0.10, 0.10)
        ax.view_init(elev=0, azim=-90)
        return mplfig_to_npimage(fig)

    clip = VideoClip(make_frame, duration=final_time)
    clip.write_videofile(video_path, codec="libx264", fps=fps, logger=None)
    plt.close(fig)