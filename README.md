# Squirrel Gripper Workspace (sg_ws)

An integrated simulation environment for studying the mechanics of a tendon-driven squirrel-inspired soft gripper. This workspace combines rod dynamics, tendon force optimization, and large-scale parametric sweeps.

---

## 📂 Project Structure

* **TendonForces/**: Core simulation logic. Contains `finger.py` and the `PyElastica` implementation for tendon-driven rods.
* **dynamics/**: Parallel research folder for kinematics and dynamic modeling.
* **runs/**: (Local Only) Directory for large-scale experiment outputs (.npz, .csv, .mp4).
* **.venv/**: Shared Python virtual environment for the workspace.

---

## 🚀 Getting Started

### 1. Environment Setup
Recreate the virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pyelastica tqdm numba matplotlib pandas

---

### 3. Execution Commands
```markdown
### 2. Running the Sweep (Large Scale)
To execute the 4,000+ sample sweep across Tension, Radius, and Vertebrae placement:

```bash
cd TendonForces
python3 sweep_runner.py

python3 TendonForces/finger.py --tension 10.0 --joint_softness 0.005 --approach_deg 45

---

### 4. Data Analysis & Maintenance
```markdown
---

## 📊 Data & Ranking
The simulation uses NIST Grasping Metrics and Gravity Margin calculations to score outputs. 

* Use the generated `.csv` files to rank results by performance scores.
* `.npz` files contain full coordinate data for reconstruction and analysis.

---

## 🛠 Maintenance
* **Git Management:** This repo ignores all large data files via `.gitignore`. 
* **Adding Data:** If you add new parallel directories, ensure they are added to the `.gitignore` to prevent tracking heavy local datasets.