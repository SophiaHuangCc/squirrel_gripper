# Tendon-Driven Squirrel-Inspired Gripper Design

45-minute research presentation starter deck

## Core story

This project develops a simulation-to-learning-to-design pipeline for a tendon-driven soft robotic finger inspired by squirrel grasping. The work begins with a PyElastica tendon-force simulator, generates a structured design dataset, trains a neural surrogate dynamics model, and uses the surrogate for design optimization and diffusion-based generation. The final validated result is not “the optimizer magically works”; the research contribution is a complete pipeline plus the discovery that physically valid design generation depends strongly on matching the joint model used in the dataset. In the current validated path, the dataset and final verification use `bending_only` joint softening, which produces stable curling behavior.

Recommended presentation tone:

- Emphasize engineering/research iteration, not just a shiny final answer.
- Show failed optimization instability as an important diagnostic result.
- Frame `full_material` vs `bending_only` as a model-assumption ablation.
- End with validated pipeline + future work: add stability-aware objectives and retrain with full-material labels.

---

## Timing plan

| Section | Time | Slides |
|---|---:|---:|
| Motivation and problem | 5 min | 1–5 |
| Physical simulator | 9 min | 6–13 |
| Dataset and metrics | 6 min | 14–18 |
| Dynamics surrogate | 7 min | 19–24 |
| Optimization | 7 min | 25–31 |
| Diffusion generation | 5 min | 32–36 |
| Validation, lessons, future work | 6 min | 37–42 |

Target: ~40–42 slides. For 45 minutes, average about 1 minute per slide, with some technical slides faster and result videos slower.

---

## Slide-by-slide starter outline

### 1. Title

Title: Learning and Optimizing Tendon-Driven Squirrel-Inspired Soft Grippers

Subtitle ideas:

- Simulation, surrogate modeling, optimization, and generative design
- From tendon-force physics to stable curling behavior

Visual:

- Best final stable curling video or final pose image.
- Suggested assets:
  - `sg_ws/TendonForces/squirrel_paw_results/output_*.mp4`
  - final optimization/diffusion verification video once selected.

Speaker note:

- Open with the project arc: building a controllable finger model, learning a fast dynamics approximation, and using it for design.

---

### 2. Biological / application motivation

Message:

- Squirrels are able to grasp branches quickly, robustly, and passively stabilize under disturbance.
- This motivates compliant, tendon-driven robotic fingers that can curl around cylindrical objects.

Visual:

- Photo/diagram of squirrel grasping if available.
- Otherwise use a simple branch + curling finger schematic.

---

### 3. Engineering problem

Question:

How can we automatically design a tendon-driven soft finger that curls around a branch quickly and remains stable under disturbance?

Key requirements:

- Fast curling
- Large contact span
- Many contacts around cylinder
- Resistance to disturbance
- Stable simulation/physical plausibility

---

### 4. Why this is hard

Key points:

- Soft-body simulation is slow and nonlinear.
- Tendon actuation couples geometry, stiffness, contact, and damping.
- Design parameters interact: joint softness, vertebra spacing, tension, ankle stiffness.
- Optimizers exploit surrogate weaknesses unless constrained.

Suggested phrase:

“The same parameter that improves curl can also destroy physical stability.”

---

### 5. Pipeline overview

Diagram:

```text
PyElastica tendon simulation
        ↓
Dataset of finger designs and metrics
        ↓
Neural dynamics surrogate
        ↓
Optimization / diffusion generation
        ↓
Full simulator verification
```

Code mapping:

- `TendonForces/finger.py`
- `dynamics/dataloader.py`
- `dynamics/profile_forward_2d.py`
- `optimization/profile_optimization.py`
- `generator/sample.py`
- `dynamics/sim_test_mj.py`

---

## Section 1: Physical simulation

### 6. Simulator foundation

Message:

- Built on PyElastica rod dynamics.
- Tendon forces are applied through vertebra-like routing points.
- Contact with a cylindrical branch is modeled with penalty contact and damping.

Code:

- `sg_ws/TendonForces/finger.py`
- `sg_ws/TendonForces/TendonForces.py`

Visual:

- Basic rod + tendon + cylinder schematic.

---

### 7. Finger design parameterization

Design vector:

```text
[joint softness ×3,
 link lengths ×4,
 base radius,
 base length,
 tendon tension,
 ankle wrap radius,
 ankle stiffness]
```

Task parameters:

```text
[approach angle, cylinder radius]
```

Initial condition:

```text
[landing height, landing speed, initial x gap]
```

Visual:

- Label finger base, three vertebrae/joints, link lengths, cylinder.

---

### 8. Tendon actuation model

Key points:

- Tendon tension produces distributed forces through vertebra contact/routing.
- Ankle wrap model changes effective tendon tension during curl.
- Tension is bounded by min/max tension in stable runs.

Suggested code reference:

- `finger.py`: ankle wrap radius and ankle stiffness arguments.

Visual:

- Tendon path over vertebrae; show tension direction.

---

### 9. Contact and stability parameters

Important final stable simulator parameters:

| Parameter | Stable value | Role |
|---|---:|---|
| Internal damping | `1.0` | dissipates whipping |
| Contact damping `nu` | `30` | absorbs contact energy |
| Contact velocity damping | `90` | reduces post-contact bounce |
| Contact stiffness `k` | `4000` | limits penetration |
| Simulation time | `5.0 s` | allows damped system to settle |

Important note:

- These stabilize reasonable designs but do not rescue physically invalid optimized designs.

---

### 10. Joint modeling: the critical ablation

Two modes:

| Mode | What softens | Behavior |
|---|---|---|
| `bending_only` | bending terms only | hinge-like curling, stable for current dataset |
| `full_material` | bending, twist, shear, axial | more physically complete, but can become floppy |

Main finding:

- The dataset used `bending_only`.
- Optimization/diffusion verification must match this mode for apples-to-apples evaluation.

Speaker note:

- This became one of the most important debugging discoveries.

---

### 11. Example single simulation

Command summary:

```bash
cd TendonForces
python finger.py \
  --v_mode manual \
  --v_list 38,58,80 \
  --joint_softness 0.003,0.002,0.001 \
  --joint_stiffness_mode bending_only \
  --damping 1.0 \
  --nu_contact 30 \
  --vel_damp_contact 90 \
  --k_contact 4000 \
  --final_time 5.0
```

Visual:

- Show MP4 or contact plot.

---

### 12. Trend studies

Message:

- Before learning, manually sweeping parameters revealed interpretable trends.

Sweeps:

- Joint softness
- Vertebra/joint position
- Tendon tension

Suggested assets:

- `sg_ws/TendonForces/trend_test_results/softness_sweep_final_pose.png`
- `sg_ws/TendonForces/trend_test_results/joint_position_sweep_final_pose.png`
- `sg_ws/TendonForces/trend_test_results/tension_sweep_final_pose.png`
- `sg_ws/TendonForces/trend_test_results/all_trend_tests_final_pose.png`

---

### 13. Simulation metrics

Metrics extracted from each run:

- Number of contacts
- Disturbance resistance score
- Angular contact span

Visual:

- Contact plot or disturbance force plot.
- Suggested assets:
  - `sg_ws/TendonForces/squirrel_paw_results/contact_plot_*.png`
  - `sg_ws/TendonForces/squirrel_paw_results/disturbance_force_*.png`

---

## Section 2: Dataset generation

### 14. Dataset generation goal

Message:

- Generate many simulated fingers across a controlled design space.
- Use full simulator labels as training data for a faster surrogate model.

Dataset example:

- `exp3`: 2500 `.npz` samples reported in training log.
- Train/test split used by `DynamicsDataset`.

Code:

- `TendonForces/parallel_runner_ray.py`
- `TendonForces/runs/exp3`

---

### 15. Sampled design ranges

From current workflow:

| Parameter | Sampling |
|---|---|
| Base radius | `0.01025–0.013 m` |
| Base length | `0.15–0.30 m` |
| Tension | `1–6 N` |
| Approach angle | `45–75°` |
| Ankle wrap radius | `0.015–0.025 m` |
| Ankle stiffness | `300–700` |
| Joint positions | ordered, separated |
| Joint softness | discrete profiles |

Speaker note:

- These ranges define what the surrogate can reliably understand.

---

### 16. Dataset outputs stored in `.npz`

Each simulation archive contains:

- Physical inputs
- Rod positions/directors over time
- Contact history
- Scalar performance metrics

Code:

- `TendonForces/finger.py`
- `dynamics/dataloader.py`

---

- `curl_hold_time = 0.2 s`
- `curl_min_contacts = 3`

---

### 18. Why dataset consistency matters

Lesson:

- The surrogate learns the behavior of the simulator configuration used to generate the data.
- If evaluation uses a different joint model, contact model, or damping setup, predictions and verified behavior diverge.

Specific discovery:

- Current dataset behavior matches `bending_only`.
- `full_material` caused instability and incomplete curling for soft optimized designs.

---

## Section 3: Dynamics surrogate

### 19. Why train a surrogate?

Motivation:

- Full PyElastica simulation is expensive.
- Optimization needs many candidate evaluations.
- A neural model approximates the simulator’s performance metrics.

Input:

```text
task_params + design_params + init_config
```

Output:

```text
[contact_norm, disturbance_score, angular_span_norm]
```

---

### 20. Surrogate architecture

Model:

- MLP with approach-angle sinusoidal embedding.
- Timestep embedding included for design-noise training compatibility.

Code:

- `dynamics/profile_forward_2d.py`
- `dynamics/trainer.py`

Visual:

```text
task params ─┐
design params ├─ MLP surrogate ─ four metrics
init config ─┘
```

---

### 21. Input normalization

Examples:

| Quantity | Normalization |
|---|---|
| Approach angle | `angle / 90` |
| Cylinder radius | `radius / 0.05` |
| Joint softness | `value / 0.001` |
| Link length | `length / 0.3` |
| Tension | `tension / 10` |
| Ankle stiffness | `stiffness / 1000` |

Why this matters:

- Keeps training numerically stable.
- Defines interface shared by optimization and diffusion.

---

### 22. Training objective

Loss:

- Mean squared error over three metrics.

Targets:

```text
contact_norm
disturbance_score
angular_span_norm
```

Code:

- `dynamics/main.py`
- `dynamics/trainer.py`

Visual:

- Use prediction-vs-truth scatter plots if available from training output.

---

### 23. Validation: ranking matters

Message:

- For optimization, exact pointwise prediction is useful, but ranking high-quality candidates is especially important.

Metrics to show:

- MAE/RMSE/R² if available.
- Top-k overlap / true score of predicted top-k if logged.

Code:

- `dynamics/main.py`: `evaluate_prediction_quality`

---

### 24. Surrogate limitations

Key points:

- It only learns within the dataset distribution.
- It predicts scalar metrics, not full physical stability.
- It can be exploited by optimizers if design bounds are too loose.

Transition:

- This motivates careful constrained optimization and full-simulator verification.

---

## Section 4: Optimization

### 25. Optimization goal

Objective examples:

```text
disturbance + 0.1 contact + 0.5 angular_span
```

Methods:

- Adam optimization of continuous design parameters
- CMA-ES as derivative-free comparison

Code:

- `optimization/profile_optimizer.py`
- `optimization/profile_optimization.py`

---

### 26. Optimized variables

Variables:

- Joint softness
- Link lengths / vertebra spacing
- Base radius and length
- Tension
- Ankle wrap radius
- Ankle stiffness
- Approach angle

Important:

- Link lengths are normalized to sum to base length.
- Vertebra nodes are recovered from cumulative link lengths.

---

### 27. Initial success and first failure mode

What worked:

- Surrogate optimization found high-scoring candidate designs.

What failed:

- Some optimized candidates were unstable in full simulation.
- They exploited very soft joints and unusual vertebra spacing.

Example red-flag optimized design:

```text
joint_softness ≈ [0.00058, 0.00058, 0.00119]
v_list ≈ [13, 26, 87]
tension ≈ 5.91 N
ankle_stiffness ≈ 693
```

---

### 28. Debugging simulator mismatch

Issue:

- At first, verification did not pass the same damping/contact/final-time parameters as stable manual simulations.

Fix:

- Explicitly passed and logged simulator parameters through `sim_test_mj.py`.

Verification files:

- `finger_command.json`
- `finger_stdout.txt`
- `finger_used_params.json`

---

### 29. Critical result: joint mode mismatch

Finding:

- `full_material` mode made optimized soft designs unstable.
- `bending_only` mode matched the dataset and stabilized verification.

Interpretation:

- `bending_only`: soft hinge behavior
- `full_material`: soft bending + soft shear/axial/twist, which can become floppy

This is a strong research result:

- Model assumptions determine whether optimized designs are physically meaningful.

---

### 30. Final optimization protocol

Validated verification setup:

- `joint_stiffness_mode = bending_only`
- stable damping/contact parameters
- top-k full simulation verification

Recommended stricter optimizer constraints:

```text
joint_soft_min ≥ 0.001 or 0.0015
link_min ≥ 0.035–0.04
tension_max ≤ 4.5–5.0 for safer designs
```

Speaker note:

- These constraints prevent the optimizer from exploiting unphysical soft/floppy regions.

---

### 31. Optimization result slide

Show:

- Best verified candidate video or final pose.
- Predicted vs verified metrics table.

Suggested table:

| Candidate | Contacts | Disturbance | Angular span | Stable? |
|---|---:|---:|---:|---|
| top-1 | ... | ... | ... | yes |
| top-2 | ... | ... | ... | yes |

---

## Section 5: Diffusion generation

### 32. Why diffusion?

Motivation:

- Optimization finds one/few local high-scoring designs.
- Diffusion can learn a distribution of feasible designs from the dataset.
- It can generate diverse candidates conditioned on desired metrics.

Reference style:

- Adapted from DGDM generator pattern.

Code:

- `generator/diffusion.py`
- `generator/diffusion_utils.py`
- `generator/train.py`
- `generator/sample.py`

---

### 33. Diffusion model setup

Training data:

```text
condition = task params + init config + target metrics
sample = 12D design vector
```

Model:

- 1D conditional UNet
- DDIM scheduler
- EMA sampling weights

Output:

- Candidate physical design vectors
- Optional dynamics-model reranking/guidance

---

### 34. Diffusion sampling workflow

Diagram:

```text
desired metrics
    ↓
conditional diffusion model
    ↓
many generated designs
    ↓
dynamics surrogate reranking
    ↓
top-k full simulation verification
```

Commands:

- `generator/train.py`
- `generator/sample.py`
- `generator/evaluate_generated_candidates.py`

---

### 35. Diffusion debugging

Issues discovered:

- EMA CPU/GPU mismatch during training.
- Evaluation needed explicit simulator parameter passthrough.
- Generated designs must be evaluated in the same `bending_only` mode as the dataset.

Why include this:

- It shows the method is sensitive but now reproducible.
- It demonstrates research maturity: we verified the source of instability.

---

### 36. Diffusion result slide

Show:

- Top-k generated candidates.
- Compare diversity vs optimization.
- Stable verified examples under `bending_only`.

Suggested table:

| Method | Diversity | Needs surrogate? | Needs full sim verify? | Best use |
|---|---|---|---|---|
| Adam | low/medium | yes | yes | local refinement |
| CMA-ES | medium | yes | yes | robust search |
| Diffusion | high | optional rerank | yes | design generation |

---

## Section 6: Synthesis and future work

### 37. Final pipeline result

Main claim:

The project now has a complete closed-loop design pipeline:

```text
simulation → dataset → surrogate → optimizer/generator → simulator verification
```

Final validated setting:

- `bending_only` joint mode
- stable damping/contact parameters
- full simulator top-k verification

---

### 38. What worked

Working components:

- Tendon-driven PyElastica finger simulation
- Contact/curl/disturbance metrics
- Dataset generation pipeline
- Four-output dynamics surrogate
- Adam/CMA-ES optimization
- Diffusion design generator
- Verified top-k simulation workflow

---

### 39. What did not work immediately

Frame this positively:

- Full-material joint model made very soft optimized designs unstable.
- Optimizer discovered unrealistic soft/clustered-joint regions.
- Diffusion needed evaluation constraints and mode matching.

Research lesson:

- Learned surrogate optimization must include physical validity constraints.

---

### 40. Design insight

Key insight:

- Curl quality depends on compliant bending.
- Stability depends on preserving shear/axial/twist stiffness and reasonable joint spacing.

Practical rule:

```text
soft enough to bend, not so soft that the rod loses material integrity
```

For `full_material`:

- Use stricter lower bound on joint softness.
- Add explicit stability labels or penalties.

---

### 41. Future work

Near-term:

- Retrain with consistent `bending_only` or full-material labels.
- Add stability metric to target vector.
- Add constraints on joint spacing and minimum link length.
- Penalize extreme tension/ankle stiffness combinations.

Long-term:

- Physical prototype validation
- Multi-finger/gripper assembly
- Real-time control rather than open-loop tension
- Sim-to-real calibration

---

### 42. Closing slide

Takeaway:

- A soft gripper design pipeline can discover high-performing tendon-driven fingers, but only when simulation assumptions, dataset generation, and verification are aligned.

Final one-liner:

“The final result is not just a curled finger — it is a reproducible pipeline for generating, scoring, optimizing, and physically verifying tendon-driven soft gripper designs.”

Visual:

- Best stable video or side-by-side manual vs optimized/diffusion final.

---

## Recommended backup slides

### Backup A. Exact stable verification command

Include your final known-good command for reproducibility.

### Backup B. `bending_only` vs `full_material` implementation

Show code snippet or matrix explanation:

- `bending_only`: scales bending entries only.
- `full_material`: scales bending, twist, shear, axial entries.

### Backup C. Optimized unstable example

Use the design:

```json
{
  "joint_softness": [0.00058, 0.00058, 0.00119],
  "v_list": [13, 26, 87],
  "tension": 5.91,
  "ankle_stiffness": 692.76
}
```

Message:

- This is an instructive surrogate-exploitation example.

### Backup D. Diffusion architecture

Show conditional UNet and DDIM denoising loop.

### Backup E. Dataset distribution plots

Use histograms if generated by:

- `TendonForces/dataset_summary.py`

---

## Suggested assets to gather

### Videos

- Stable manual reference:
  - `sg_ws/TendonForces/squirrel_paw_results/output_*.mp4`
- Trend videos:
  - `sg_ws/TendonForces/trend_test_results/output_*.mp4`
- Final optimization/diffusion verification:
  - generated under `optimization/runs/.../sim_verification...`
  - generated under `generator/runs/sample_exp3/sim_verification...`

### Figures

- `sg_ws/TendonForces/trend_test_results/all_trend_tests_final_pose.png`
- `sg_ws/TendonForces/trend_test_results/softness_sweep_final_pose.png`
- `sg_ws/TendonForces/trend_test_results/joint_position_sweep_final_pose.png`
- `sg_ws/TendonForces/trend_test_results/tension_sweep_final_pose.png`
- `sg_ws/TendonForces/pareto_plot.png`
- `docs/pareto_plot.png`
- `docs/Stress-Strain (1).png`

### Code snippets

- `TendonForces/finger.py`: joint mode and metric extraction
- `dynamics/dataloader.py`: input/target construction
- `dynamics/profile_forward_2d.py`: surrogate architecture
- `optimization/profile_optimizer.py`: objective and bounds
- `generator/diffusion.py`: conditional sampling

---

## Presentation framing advice

Avoid saying:

- “Some parts failed.”
- “The model was wrong.”

Say instead:

- “This exposed a simulator-assumption mismatch.”
- “The optimizer revealed an unmodeled physical validity constraint.”
- “We identified that the dataset corresponds to a bending-only joint abstraction.”
- “The final validated pipeline uses matched simulation assumptions.”

The clean research narrative is:

1. Build simulator.
2. Define measurable grasp metrics.
3. Generate dataset.
4. Learn surrogate.
5. Optimize/generate designs.
6. Verify physically.
7. Diagnose mismatch.
8. Establish validated mode and future stability-aware training.
