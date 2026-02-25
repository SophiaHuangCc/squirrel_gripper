# Squirrel Gripper

---

## Project Structure

* **TendonForces**: Core simulation logic. Contains `finger.py` and the `PyElastica` implementation for tendon-driven rods.
* **dynamics**: Parallel research folder for kinematics and dynamic modeling.


---

## TendonForces Simulation

### 1. Environment Setup (for TendonForces)
Recreate the virtual environment and install dependencies:

```bash
cd TendonForces
python3 -m venv .venv
source .venv/bin/activate
pip install requirements.txt # for python 3.12
pip install req.txt # for python 3.10
```

#### Alternative: Environment Setup with Mamba (https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos-linux--wsl)
```bash
cd TendonForces
mamba create -n soft python=3.12
mamba activate soft
pip install -r requirements.txt # for python 3.12
# or, for Python 3.10:
# mamba create -n soft python=3.10
# mamba activate soft
# pip install -r req.txt
```

---

### 2. Tendon Force Simulation for Squirrel Finger

1. **Running the Sweep (Large Scale):**
To execute the large sample sweep across the optimizable parameters:

```bash
cd TendonForces
python3 parallel_runner.py # for ubuntu machine
python3 parallel_local.py # for mac
python3 parallel_runner_ray.py # Ray parallel (use --num_cpus N to override)
```

2. **Running a single set of parameters:**
Can add the --debug flag for force visualization
```bash
python ./finger.py --tension 10.0 --joint_softness 0.002 --sol approach_angle --base_rad 0.005 --approach_deg 45
```

simulate approaching motion
```bash
python ./finger.py --tension 10.0 --joint_softness 0.002 --sol approach_angle --base_rad 0.005 --approach_deg 45 --landing_motion --landing_height 0.03 --landing_speed 0.0 --full_visualization
```

---

### 3. Configuration Parameters Reference

The simulation supports a wide range of command-line arguments to tune the physics and environment.

### 1. Simulation & Material Properties
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--E` | `2e7` | Young's Modulus (Pa). Controls overall rod stiffness. |
| `--poisson_nu` | `0.4` | Poisson's ratio for the material. |
| `--damping` | `0.8` | Internal damping constant for the rod dynamics. |
| `--n_elements` | `80` | Number of discretization elements in the rod. |
| `--base_len` | `0.10` | Total finger length (m). |
| `--base_rad` | `0.005` | Finger cross-section radius (m). |

### 2. Tendon & Approach Logic
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--sol` | `approach_angle` | Choice of solver logic (e.g., `standard`, `approach_angle`, `nonuniform_tendon`). |
| `--approach_deg` | `45.0` | Angle of approach in degrees (0 = horizontal, 90 = vertical). |
| `--tension` | `0.4` | Pulling force applied to the tendon (N). |



### 3. Cylinder & Contact Physics
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--cyl_rad` | `0.015` | Radius of the target cylinder (m). |
| `--k_contact` | `1.25e3` | Contact stiffness between rod and cylinder. |
| `--nu_contact` | `5.0` | Contact damping to prevent jitter. |
| `--mu_contact` | `0.6` | Friction coefficient between rod and cylinder. |
| `--vel_damp_contact` | `10` | Numerical stability parameter for contact velocity. |

### 4. Vertebrae & Hinge System
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--v_mode` | `uniform` | Vertebrae placement strategy: `uniform` or `manual`. |
| `--v_list` | `30,46,62` | Comma-separated node indices used when `v_mode` is `manual`. |
| `--v_start` / `--v_end` | `30` / `62` | Range of nodes for vertebrae when `v_mode` is `uniform`. |
| `--num_v` | `3` | Total number of vertebrae to place. |
| `--v_mass` | `0.002` | Mass (kg) of each individual vertebra. |
| `--v_height` | `0.005` | Offset distance (moment arm) of the tendon from the rod center. |
| `--joint_softness` | `0.01` | Multiplier for bending stiffness at hinge points (e.g., `0.01` = 1% stiffness). |



### 5. Output & Stability
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--body_mass` | `0.5` | Mass of the squirrel body (kg) used for stability calculations. |
| `--output_dir` | `squirrel_paw_results` | Directory where data and videos are saved. |
| `--suffix` | `default` | Filename suffix to differentiate experimental runs. |
| `--debug` | `False` | Flag to enable force magnitude and direction plots. |


---

### 3. Data Analysis & Maintenance
```markdown
---

## Data & Ranking
The simulation uses NIST Grasping Metrics and Gravity Margin calculations to score outputs. 

* Use the generated `.csv` files to rank results by performance scores.
* `.npz` files contain full coordinate data for reconstruction and analysis.

## Offline Ranking
Given a directory that contains simulation results, list out the best designs for each metric and visualize in a wandb chart
```bash
python offline_analysis.py
```

---
```

## Dymanics Model