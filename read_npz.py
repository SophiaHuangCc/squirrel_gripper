import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Load your saved data
# Replace with your actual filename
data = np.load("squirrel_paw_results/master_log_20260206_015209_default.npz")

# 2. Extract the positions (Frames, 3, Nodes)
pos = data["position"]
c_pos = data["cyl_position"]    # Cylinder center (3, 1)
c_rad = data["cyl_radius"]
tension = data["tension"]
cyl_rad = data["cyl_radius"]
# angle = data["approach_deg"]
print("Tension used in this simulation:", tension)
print("Cylinder radius:", cyl_rad)
# print("Approach angle (deg):", angle)

# 3. Create the interactive 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Let's look at the very last frame (the grip)
# pos[-1] means the last time step
# pos[-1, 0, :] is all X-coords, [ -1, 1, :] is Y, [ -1, 2, :] is Z
ax.plot(pos[-1, 0, :], pos[-1, 1, :], pos[-1, 2, :], 'b-o', markersize=4, label="Finger")
ax.scatter(c_pos[0], c_pos[1], c_pos[2], s=c_rad*1000, color='gray', alpha=0.5, label="Cylinder")

# 4. Formatting the view
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title("Interactive Finger Visualization (Last Frame)")
ax.legend()

# Set equal limits so the finger doesn't look stretched
ax.set_xlim(-0.02, 0.12)
ax.set_ylim(-0.05, 0.05)
ax.set_zlim(-0.10, 0.10)

plt.show() # This opens the window where you can rotate with your mouse!