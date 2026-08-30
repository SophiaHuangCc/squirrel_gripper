# Squirrel Gripper

This repository simulates a tendon-driven squirrel finger, trains a neural
surrogate of the simulation, optimizes finger designs through that surrogate,
and verifies the strongest predicted candidates in the full simulator.

For the controlled multi-scenario evaluation protocol and a plain-language
explanation of designs, candidates, methods, and scenarios, see
[`benchmarks/README.md`](benchmarks/README.md).

The current workflow is:

```text
PyElastica simulations
        ↓
train/test .npz dataset
        ↓
3-output surrogate model
        ↓
Adam or CMA-ES profile optimization
        ↓
top-k full-simulation verification
```

The active grasp metrics are contact count, disturbance resistance, and angular
contact span.

## Repository layout

- `TendonForces/finger.py`: full PyElastica finger simulation and metric
  extraction.
- `TendonForces/parallel_runner_ray.py`: current randomized train/test dataset
  generator.
- `dynamics/dataloader.py`: converts simulation archives into normalized model
  inputs and three training targets.
- `dynamics/main.py`: surrogate training and validation.
- `dynamics/profile_forward_2d.py`: forward surrogate network.
- `optimization/profile_optimization.py`: optimizes finger parameters against
  the frozen surrogate.
- `optimization/profile_optimizer.py`: Adam and CMA-ES implementations and
  objective definitions.
- `optimization/evaluate_optimized_candidates.py`: selects predicted top-k
  designs and reruns them in `finger.py`.
- `benchmarks/`: evaluates every fixed candidate design on the same versioned
  scenario suite and produces candidate- and method-level comparisons.

Run the commands below from the repository root, the directory containing
`TendonForces`, `dynamics`, and `optimization`.

## Environment

The current runs use Python 3.10 and the repository-level virtual environment:

```bash
cd /home/real/Desktop/SquirrelGripper/ws/squirrel_gripper
source .venv/bin/activate
```

To create a fresh environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r TendonForces/requirements_py310.txt
```

The requirements file includes PyTorch, Diffusers, W&B, SciPy, and `cma`.
Install Ray separately if it is not already available:

```bash
pip install ray
```

A CUDA GPU is currently expected by `profile_optimization.py`.

## Recommended end-to-end benchmark workflow

Run this workflow from the repository root. These paths match the Ubuntu lab
machine; change only `PROJECT_DIR` and `DATASET_DIR` when using another machine
or experiment.

### 1. Export paths

```bash
export PROJECT_DIR=/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper
export DATASET_DIR="$PROJECT_DIR/TendonForces/runs/exp1"
export DYNAMICS_DIR="$PROJECT_DIR/outputs/from_links_v2/dynamics"
export DIFFUSION_DIR="$PROJECT_DIR/outputs/from_links_v2/diffusion"
export BENCHMARK_DIR="$PROJECT_DIR/outputs/from_links_v2/benchmark"

cd "$PROJECT_DIR"
source .venv/bin/activate
```

Dynamics training logs to W&B automatically. Run `wandb login` once on a new
machine, or set `WANDB_MODE=offline` if the machine temporarily has no network.

### 2. Train the 3-task, 16-design dynamics model

```bash
python dynamics/main.py \
  --mode train \
  --device cuda \
  --data_dir "$DATASET_DIR/train" \
  --test_data_dir "$DATASET_DIR/test" \
  --save_dir "$DYNAMICS_DIR" \
  --batch_size 64 \
  --num_workers 8 \
  --lr 1e-3 \
  --num_epochs 300 \
  --patience 20 \
  --val_step 5 \
  --save_ckpt_step 500 \
  --output_dim 3
```

Use `$DYNAMICS_DIR/best.pt` for candidate selection. Checkpoints from the old
2-task/15-design contract are incompatible and must not be reused.

### 3. Train conditional diffusion

```bash
python generator/train.py \
  --device cuda \
  --data_dir "$DATASET_DIR/train" \
  --save_dir "$DIFFUSION_DIR" \
  --batch_size 256 \
  --num_workers 8 \
  --num_epochs 500 \
  --num_train_timesteps 100 \
  --num_inference_steps 20 \
  --learning_rate 1e-4 \
  --val_ratio 0.05 \
  --patience 30 \
  --min_delta 1e-5 \
  --save_every 25 \
  --wandb_project squirrel-gripper-diffusion \
  --wandb_mode online \
  --seed 0
```

Both `conditional_diffusion` and `dgdm` use this same diffusion checkpoint.
DGDM additionally uses the dynamics checkpoint during denoising.
Diffusion training logs train/validation loss, learning rate, best validation
loss, and early-stopping state to the separate `squirrel-gripper-diffusion`
W&B project. Here, 500 is a maximum epoch budget; training stops earlier after
30 epochs without a validation-loss decrease greater than `1e-5`. Set
`--patience 0` to disable early stopping or `--wandb_mode offline` when the lab
machine has no network.

### 4. Candidate-generation smoke test

This quick command exercises every method and writes candidate files, but does
not start the expensive PyElastica scenario rollouts:

```bash
python -m benchmarks.run_baselines \
  --output_dir "${BENCHMARK_DIR}_smoke" \
  --methods reference,random,random_search,retrieval,adam,cma_es,conditional_diffusion,dgdm \
  --candidate_budget 2 \
  --seeds 0 \
  --retrieval_data_dir "$DATASET_DIR/train" \
  --dynamics_checkpoint "$DYNAMICS_DIR/best.pt" \
  --diffusion_checkpoint "$DIFFUSION_DIR/best.pt" \
  --device cuda \
  --random_pool_size 32 \
  --adam_steps 10 \
  --cma_generations 3 \
  --cma_popsize 8 \
  --diffusion_num_samples 8 \
  --diffusion_batch_size 8 \
  --diffusion_inference_steps 5 \
  --dgdm_guidance_scale 0.1
```

Successful completion produces one candidate NPZ per requested method under
`${BENCHMARK_DIR}_smoke/candidates/`. To test simulator command construction
without executing PyElastica, add `--run_benchmark --dry_run`; `--dry_run` is
an option on this command, not a separate benchmark protocol.

### 5. Choose the design-selection protocol

Set `TARGET_ARGS` once before the full command:

| Protocol | Setting | Meaning |
| --- | --- | --- |
| Default nominal specialist | `TARGET_ARGS=()` | Select for `nominal:00` |
| Exact-cell specialist | `TARGET_ARGS=(--target_scenario_id orientation:08)` | Select for one named cell |
| Family specialist | `TARGET_ARGS=(--target_family orientation)` | Select for mean utility over one family |
| Generalist | `TARGET_ARGS=(--generalist)` | Select for mean utility over all 28 labeled cells |

These options are mutually exclusive. In every case, `--run_benchmark`
evaluates the selected frozen top-1 design on the common 28-cell suite. Thus a
specialist is selected on its target but still receives the full transfer test.

### 6. Run the full selected protocol

After setting `TARGET_ARGS`, use the same command for nominal, exact-cell,
family, or generalist experiments:

```bash
python -m benchmarks.run_baselines \
  --output_dir "$BENCHMARK_DIR" \
  --methods reference,random,random_search,retrieval,adam,cma_es,conditional_diffusion,dgdm \
  --candidate_budget 16 \
  --seeds 0,1,2,3,4 \
  --retrieval_data_dir "$DATASET_DIR/train" \
  --dynamics_checkpoint "$DYNAMICS_DIR/best.pt" \
  --diffusion_checkpoint "$DIFFUSION_DIR/best.pt" \
  --device cuda \
  --random_pool_size 256 \
  --adam_steps 300 \
  --adam_lr 0.03 \
  --cma_generations 100 \
  --cma_popsize 32 \
  --cma_sigma 0.5 \
  --diffusion_num_samples 256 \
  --diffusion_batch_size 256 \
  --diffusion_inference_steps 20 \
  --dgdm_guidance_scale 0.1 \
  "${TARGET_ARGS[@]}" \
  --run_benchmark \
  --benchmark_top_k 1 \
  --num_workers 1 \
  --timeout 1800
```

Use a different `BENCHMARK_DIR` for each protocol so results are not mixed or
mistaken for resumable copies of another experiment. For example, append
`/nominal`, `/orientation_family`, or `/generalist` to the exported path.

Useful options on this single command are:

- `--families orientation,branch_offset` limits the *simulation evaluation*
  suite; it does not change which conditions select the candidate.
- `--benchmark_top_k K` evaluates multiple candidates and is an oracle
  diagnostic. Keep `1` for the primary deployable comparison.
- `--render` enables simulator videos. Without it, benchmark runs disable video
  generation to save time and storage.
- `--surrogate_eval_budget N` overrides method-specific search lengths to
  approximately equalize surrogate evaluations. Record the chosen value when
  using it; its effect depends on whether the target contains 1, 9, or 28 cells.
- `--dgdm_guidance_scale` should be selected on validation scenarios and then
  frozen before the final test run.

Final tables and plots are written under `$BENCHMARK_DIR/summary/`, including
`method_summary.csv`, `method_aggregate.csv`, `surrogate_calibration.csv`, and
`method_comparison.png`.

> **Current versus legacy paths:** the workflow above is the authoritative
> 3-task/16-design benchmark pipeline. The standalone optimization notes below
> document older scripts and checkpoint layouts; do not mix their legacy
> 2-task/12–15-design artifacts with the benchmark commands above.

## 1. Generate a dataset

The current dataset generator samples unique finger designs, shuffles them,
then writes separate train and test splits:

```bash
cd TendonForces
python parallel_runner_ray.py \
  --num_train 2000 \
  --num_test 500 \
  --seed 123 \
  --num_cpus 24
cd ..
```

It creates the next available experiment directory:

```text
TendonForces/runs/expN/
├── metadata.json
├── train/
│   ├── split_info.json
│   └── master_log_*.npz
└── test/
    ├── split_info.json
    └── master_log_*.npz
```

The generator currently varies:

| Parameter | Sampled values or rule |
| --- | --- |
| Base radius | `0.01025, 0.011, 0.0115, 0.012, 0.0125, 0.013` m |
| Base length | `0.15, 0.20, 0.25, 0.30` m |
| Tendon tension | `1, 2, 2.5, 3, 4, 4.5, 5, 6` N |
| Ankle wrap radius | `0.015–0.025` m from a five-value list |
| Ankle stiffness | `300, 400, 500, 600, 700` |
| Approach angle | `45, 50, 60, 65, 70, 75` degrees |
| Joint softness | One of eight three-joint profiles |
| Joint positions | Three ordered nodes with a minimum 20-node separation |

The cylinder radius is fixed to `0.03 m` in this generator. Landing height,
landing speed, and initial horizontal gap are fixed to `0.04 m`, `0 m/s`, and
`0.06 m`.

New datasets use `joint_stiffness_mode=full_material`. Each joint multiplier
is treated as a joint/link modulus ratio and is applied to all three diagonal
terms in both the bending/twist and shear/axial constitutive matrices over the
eight-element joint region. The legacy `bending_only` mode reproduces the
older behavior that scaled only `bend_matrix[1,1]` and `bend_matrix[2,2]`.

The generator also enables `data_only`, which retains the `.npz`, contact CSV,
summary CSV, and scalar metrics but skips MP4, PNG/JPG, and interactive
visualization generation.

Each simulation archive contains the sampled physical parameters, complete
time-dependent rod state, contact information, final metrics, and the
per-frame `contact_counts` history.

### Run one simulation

For a single reference run:

```bash
cd TendonForces
bash run.sh
cd ..
```

Or call `finger.py` directly:

```bash
cd TendonForces
python finger.py \
  --approach_deg 45 \
  --cyl_rad 0.03 \
  --base_len 0.20 \
  --base_rad 0.01025 \
  --tension 3.0 \
  --v_mode manual \
  --v_list 38,58,80 \
  --joint_softness 0.003,0.002,0.001 \
  --joint_stiffness_mode full_material \
  --ankle_wrap_radius 0.02 \
  --ankle_stiffness 500 \
  --landing_motion \
  --landing_mode prescribed \
  --landing_height 0.04 \
  --landing_speed 0.0 \
  --initial_x_gap 0.06 \
  --final_time 2.0 \
  --curl_contact_ratio 0.8 \
  --curl_hold_time 0.2 \
  --curl_min_contacts 3 \
  --data_only \
  --output_dir squirrel_paw_results \
  --suffix manual_test
cd ..
```

Important simulation outputs include:

- `master_log_<run>_<suffix>.npz`: physical inputs, trajectories, and scalar
  metrics.
- `contact_log_<run>_<suffix>.csv`: per-frame contact geometry and force data.
- `contact_plot_<run>_<suffix>.png`: final contact visualization.
- `disturbance_force_*.png`: disturbance-response visualizations.
- `output_<run>_<suffix>.mp4`: simulation video when video generation is
  enabled.

## 2. Surrogate inputs and outputs

The surrogate receives 17 scalar values divided into three groups.

### Task input: 2 values

| Index | Physical value | Model normalization |
| --- | --- | --- |
| 0 | Approach angle in degrees | `angle / 90` |
| 1 | Cylinder radius in metres | `radius / 0.05` |

### Design input: 12 values

| Indices | Physical value | Model normalization |
| --- | --- | --- |
| 0–2 | Three joint-softness multipliers | `value / 0.001` |
| 3–6 | Four link lengths derived from joint nodes | `length / 0.3` |
| 7 | Finger/base radius | `radius / 0.02` |
| 8 | Total finger/base length | `length / 0.2` |
| 9 | Tendon tension | `tension / 10` |
| 10 | Ankle wrap radius | `radius / 0.025` |
| 11 | Ankle stiffness | `stiffness / 1000` |

### Initial configuration input: 3 values

| Index | Physical value | Model normalization |
| --- | --- | --- |
| 0 | Landing height | `height / 0.10` |
| 1 | Landing speed | `speed / 1.0` |
| 2 | Initial horizontal gap | `gap / 0.10` |

### Predicted targets: 3 values

The output ordering is fixed and must remain consistent between the dataset,
checkpoint, and optimizer:

| Index | Target | Definition |
| --- | --- | --- |
| 0 | `contact_norm` | `log(1 + num_contacts) / log(1 + n_elements)` |
| 1 | `disturbance_score` | Mean directional resistance score from the simulated disturbances |
| 2 | `angular_span_norm` | `clip(angular_span_degrees / 180, 0, 1)` |

## 3. Train the surrogate

Train the three-output dynamics model with:

```bash
python dynamics/main.py \
  --mode train \
  --data_dir TendonForces/runs/exp3/train \
  --test_data_dir TendonForces/runs/exp3/test \
  --save_dir checkpoints \
  --batch_size 32 \
  --num_workers 0 \
  --lr 1e-3 \
  --num_epochs 300 \
  --patience 10 \
  --val_step 5 \
  --save_ckpt_step 500 \
  --output_dim 3 \
  --use_design_noise
```

Four-output checkpoints are intentionally incompatible and must be retrained.

Training minimizes unweighted mean squared error over the three normalized
targets. Validation also reports per-output MAE, RMSE, R², Spearman
correlation, and top-k ranking quality to W&B.

Checkpoint behavior:

- `checkpoints/best.pt` is saved whenever total validation MSE reaches a new
  minimum. This is normally the checkpoint to use for optimization.
- `checkpoints/latest.pt` is a periodic training snapshot. It is newer in wall
  time but is not necessarily better on validation data.

The current three-output model is not checkpoint-compatible with earlier
four-output experiments.

To validate a checkpoint, note that the current validation entry point creates
a model but does not automatically load `--checkpoint_path`; checkpoint-only
validation therefore requires adding/loading the state dictionary in code
before relying on `--mode validate`.

## 4. Optimize finger profiles

Run Adam optimization with the best three-output checkpoint:

```bash
python optimization/profile_optimization.py \
  --checkpoint_path /home/real/Desktop/SquirrelGripper/ws/squirrel_gripper/checkpoints/best.pt \
  --num_epochs 100 \
  --batch_size 16 \
  --output_dim 3
```

Run CMA-ES:

```bash
python optimization/profile_optimization.py \
  --checkpoint_path /home/real/Desktop/SquirrelGripper/ws/squirrel_gripper/checkpoints/best.pt \
  --num_epochs 100 \
  --batch_size 1 \
  --output_dim 3 \
  --use_es
```

Use `--batch_size 1` for the current CMA-ES implementation. It flattens the
entire batch into one search vector, so batch size 16 would create a
`16 × 13 = 208` dimensional CMA-ES problem instead of one 13-dimensional
design problem. Adam may use a larger batch to optimize multiple initialized
candidates concurrently.

The script currently optimizes all of these objectives sequentially:

| Output directory | Surrogate objective maximized |
| --- | --- |
| `disturbance_surrogate_only` | `D` |
| `disturbance_contact_surrogate_only` | `D + 0.1 C` |
| `contact_surrogate_only` | `C` |
| `angular_span_surrogate_only` | `0.5 S` |
| `disturbance_span_surrogate_only` | `D + 0.5 S` |
| `disturbance_contact_span_surrogate_only` | `D + 0.1 C + 0.5 S` |

Here `C`, `D`, and `S` are predicted normalized contact, disturbance, and
angular-span outputs.

### Optimized variables and bounds

Each candidate has 13 optimized variables:

| Variables | Initialization | Physical bounds |
| --- | --- | --- |
| Joint softness ×3 | `0.003, 0.003, 0.003` | `0.0005–0.005` |
| Link-length proportions ×4 | `0.06, 0.056, 0.044, 0.04` m | Raw links mapped through `0.02–0.10`, then rescaled to sum to base length |
| Base radius | `0.01025` m | `0.01025–0.013` m |
| Base length | `0.20` m | `0.15–0.25` m |
| Tendon tension | `3.0` N | `1.0–6.0` N |
| Ankle wrap radius | `0.02` m | `0.015–0.025` m |
| Ankle stiffness | `500` | `300–700` |
| Approach angle | `45°` | `0–90°` |

The cylinder radius and initial landing configuration are fixed during a run
at `0.03 m`, `0.04 m`, `0 m/s`, and `0.06 m`, respectively.

The raw optimizer variables are mapped into physical bounds with sigmoid
functions. Values approaching large positive or negative magnitudes therefore
approach physical boundary values.

### Optimization outputs

Regardless of a supplied `--save_dir`, the current script creates the next
automatic directory:

```text
optimization/runs/expN/
```

Each objective directory contains:

```text
<objective>_surrogate_only/optimized_candidates.npz
```

The archive contains:

- `design_params`: physical 12-value finger designs, shape
  `[batch_size, 12]`.
- `task_params`: optimized approach angle and fixed cylinder radius, shape
  `[batch_size, 2]`.
- `pred_metrics`: predicted `[C, D, S, V]`, shape `[batch_size, 4]`.

The current `profile_optimization.py` sets `skip_sim=True`, so this stage uses
only the surrogate. Full simulation verification is the next explicit step.

## 5. Verify optimized candidates

Select the predicted top candidates and rerun them in `finger.py`:

```bash
python optimization/evaluate_optimized_candidates.py \
  --optimization_dir optimization/runs/exp12 \
  --output_dir optimization/runs/exp12/sim_verification \
  --top_k 3 \
  --num_cpus 3 \
  --objectives disturbance_contact_span
```

Replace `exp12` with the optimization directory printed as `[OPT RUN DIR]`.
If `--output_dir` is omitted, it defaults to
`<optimization_dir>/sim_verification`.

If `--objectives` is omitted, the evaluator processes all six objectives.
For each requested objective it:

1. Loads `<objective>_surrogate_only/optimized_candidates.npz`.
2. Recomputes the surrogate objective for every candidate.
3. Selects the highest-scoring `--top_k` candidates.
4. Saves `selected_candidates.npz`.
5. Runs each selected design through `TendonForces/finger.py` in parallel.
6. Saves `verification_results.npz` with predicted scores, simulation metrics,
   and result directories.

The verification layout is:

```text
sim_verification/
└── <objective>_verified_topK/
    ├── selected_candidates.npz
    ├── verification_results.npz
    └── finger_<index>/
        ├── design.json
        ├── finger_<index>.npz
        └── master_log_*.npz
```

`selected_candidates.npz` contains selected IDs, physical design/task values,
predicted metrics, and predicted objective scores.

`verification_results.npz` contains those predictions together with the
full-simulation metric dictionaries and paths. Important verified fields are
`num_contacts`, `disturbance_resistance_score`, and `angular_span`.

## Current implementation notes

- Always use `best.pt` unless intentionally inspecting a later training
  snapshot. New checkpoints contain exactly three outputs.
- The correct filename is `latest.pt`, not `lastest.pt`.
- `profile_optimization.py` currently forces CUDA with `.cuda()`.
- `profile_optimization.py` currently ignores the requested `--save_dir` and
  creates `optimization/runs/expN`.
- The shared parser exposes `--lr`, but profile optimization currently reads a
  separate internal `learning_rate` default of `1e-4`.
- Candidate verification is essential: the optimizer can exploit surrogate
  extrapolation near design bounds, especially for objectives trained with
  sparse or noisy target coverage.
- The W&B prediction-quality plotting code currently emits a Matplotlib
  open-figure warning during long training runs. This warning does not change
  checkpoint selection or optimization results.
