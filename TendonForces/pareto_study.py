import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ====== Load data ======
file_path = "/Users/sophiahuang/Desktop/SquirrelGripper/sg_ws/TendonForces/pareto_optimal_study.xlsx"   # or full path
df = pd.read_excel(file_path)

df = df.rename(columns={
    "Tension (N)": "tension",
    "Finger Radius (mm)": "radius",
    "Joint Softness (%)": "softness"
})

# ====== Pareto computation (per radius) ======
# Objective: minimize tension, maximize softness
def pareto_mask_min_max(tension: np.ndarray, softness: np.ndarray) -> np.ndarray:
    """
    Pareto-optimal mask where tension is minimized and softness is maximized.
    i is dominated if exists j with:
      tension_j <= tension_i AND softness_j >= softness_i
      and at least one strict.
    """
    n = len(tension)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominates_i = (tension <= tension[i]) & (softness >= softness[i]) & (
            (tension < tension[i]) | (softness > softness[i])
        )
        if np.any(dominates_i):
            is_pareto[i] = False
    return is_pareto

df["pareto"] = False
for r, g in df.groupby("radius", sort=False):
    mask = pareto_mask_min_max(g["tension"].to_numpy(), g["softness"].to_numpy())
    df.loc[g.index, "pareto"] = mask

pareto_pts = df[df["pareto"]].copy()

# ====== Best-fit line through Pareto points (3D PCA / TLS) ======
P = pareto_pts[["radius", "softness", "tension"]].to_numpy(dtype=float)

line_xyz = None
if P.shape[0] >= 2:
    mu = P.mean(axis=0)
    X = P - mu
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    direction = vt[0]                 # principal direction
    t = X @ direction
    t_min, t_max = t.min(), t.max()
    line_xyz = np.vstack([mu + t_min * direction, mu + t_max * direction])

# ====== Plot: 3D surface + points + Pareto + best-fit line ======
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

x = df["radius"].to_numpy()
y = df["softness"].to_numpy()
z = df["tension"].to_numpy()

# Triangulated surface (works for non-gridded points)
try:
    ax.plot_trisurf(x, y, z, alpha=0.25, linewidth=0.2)
except Exception:
    pass

ax.scatter(x, y, z, s=45, alpha=0.7, label="All samples")
ax.scatter(
    pareto_pts["radius"], pareto_pts["softness"], pareto_pts["tension"],
    s=90, label="Pareto-optimal"
)

if line_xyz is not None:
    ax.plot(line_xyz[:, 0], line_xyz[:, 1], line_xyz[:, 2],
            linewidth=3, label="Best-fit line (PCA)")
    
# ====== Line equation ======
if P.shape[0] >= 2:
    r0, s0, T0 = mu
    a, b, c = direction

    eq_text = (
        "Best-fit line (parametric):\n"
        f"r = {r0:.3f} + {a:.3f}·t\n"
        f"s = {s0:.3f} + {b:.3f}·t\n"
        f"T = {T0:.3f} + {c:.3f}·t"
    )

    print("\n=== Best-fit Pareto line ===")
    print(eq_text)

if line_xyz is not None:
    # Place text near center of line
    mid = line_xyz.mean(axis=0)

    ax.text(
        mid[0], mid[1], mid[2],
        eq_text,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

ax.set_xlabel("Finger Radius (mm)")
ax.set_ylabel("Joint Softness (%)")
ax.set_zlabel("Tension (N)")
ax.set_title("3D Pareto Study: Radius vs Softness vs Tension\n(Pareto points + best-fit line)")
ax.view_init(elev=20, azim=45)
ax.legend(loc="best")

plt.tight_layout()
plt.savefig("pareto_plot.png", dpi=200)
plt.show()