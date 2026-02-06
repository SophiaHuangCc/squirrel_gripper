# Squirrel Gripper

---

## Project Structure

* **TendonForces**: Core simulation logic. Contains `finger.py` and the `PyElastica` implementation for tendon-driven rods.
* **dynamics**: Parallel research folder for kinematics and dynamic modeling.


---

## Getting Started

### 1. Environment Setup
Recreate the virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requirements.txt # for python 3.12
pip install req.txt # for python 3.10
```

---

### 2. Tendon Force Simulation for Squirrel Finger

1. **Running the Sweep (Large Scale)**
To execute the large sample sweep across the optimizable parameters:

```bash
cd TendonForces
python3 parallel_runner.py # for ubuntu machine
python3 parallel_local.py # for mac
```

2. **Running a single set of parameters**
Can add the --debug flag for force visualization
```bash
python ./finger.py --tension 3.0 --joint_softness 0.001 --sol approach_angle --base_rad 0.005 --approach_deg 45
```

---

### 3. Data Analysis & Maintenance
```markdown
---

## Data & Ranking
The simulation uses NIST Grasping Metrics and Gravity Margin calculations to score outputs. 

* Use the generated `.csv` files to rank results by performance scores.
* `.npz` files contain full coordinate data for reconstruction and analysis.

---
```