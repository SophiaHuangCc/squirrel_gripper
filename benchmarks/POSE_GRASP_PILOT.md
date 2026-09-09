# Pose guidance and grasp screening pilot

Run from the repository root with `.venv/bin/python`. These tools do not retrain
models or change V20 generation, objectives, simulation, or existing results.

## Measured smoke results (2026-09-08)

Two separate runs evaluated one randomly chosen anchor each, clean checkpoint,
scenario `approach_radius:00@nominal`, seed 17. All 14 simulations completed;
each batch of seven took about 2.5 minutes. Results are in
`outputs/pose_pilot_smoke` and `outputs/pose_pilot_dgdm_smoke`.

| Anchor source | Step | Pose-change cosine | Simulated surface-fit loss improved? |
|---|---:|---:|---|
| conditional_diffusion | 0.01 | 0.808 | yes |
| conditional_diffusion | 0.03 | 0.543 | yes |
| pose_dgdm_gs1 | 0.01 | 0.590 | yes |
| pose_dgdm_gs1 | 0.03 | 0.350 | no |

Reverse and random controls did not improve this objective on either anchor.
Absolute coordinate RMSE was 24.3 mm and 8.0 mm respectively. The large DGDM step
produced a 45.2 mm simulated pose-vector change versus 7.8 mm predicted, and
increased the loss despite a predicted decrease. This is evidence to inspect
nonlinear/contact transitions and expand small-step diagnostics, not declare
gradient reliability. Try `--steps 0.005,0.01` on more anchors next.

Cached V20 screening (`outputs/grasp_screen_pilot/report.json`, 320 trials per
method, one seed) gives 65.0% versus 57.5% for DGDM scale 1 versus diffusion-only
on the primary geometry/support profiles. Adding the provisional 5 N requirement
in every direction gives 1.875% versus 5.9375%. These contrasting exploratory
results must both be reported; threshold sensitivity cannot establish retention.

## 1. Local poses and task-directed steps: within a 20–30 minute pilot budget

The smallest useful experiment is one randomly chosen existing design, two step
sizes, and three directions (pose-loss descent, reverse, random), plus a baseline:
**7 simulations**. All seven can run concurrently on this 32-CPU machine. Each
simulation has a 1,200-second timeout. This bounds an individual attempt, not a
guarantee that it succeeds; contention and slow designs can cause timeouts.
The first measured seven-job run here completed in about 2.5 minutes with all jobs
successful. Use that as a local timing observation, not a guarantee for other designs.
Keep the full 2-second simulated closure. Shortening it changes the prediction
target and invalidates the comparison with the checkpoint.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.pose_pilot \
  --source outputs/from_links_v20_force_screened_fast24h/specialists \
  --checkpoint outputs/from_links_v16_pose_dgdm/pose_dynamics_clean/best.pt \
  --output outputs/pose_pilot_small \
  --methods conditional_diffusion --anchors 1 --workers 7 --timeout 1200
```

Use `--prepare-only` to inspect the exact designs and predictions in `manifest.json`
first. Repeat the identical command to resume: successful simulations are cached,
failed ones are retried. Use a new output directory when changing the checkpoint,
seed, methods, objective, or tolerances. One anchor is an end-to-end smoke test,
not evidence of population-level gradient quality.

The default objective is squared radial distance of the three joints and tip
from a cylinder-centerline target radius of `cyl_rad + 0.01 m`. The 10 mm clearance
is an explicit geometric choice matching the fixed nominal base radius; actual
rectangular-finger contact geometry can differ. Base keypoints are excluded from
the loss. This surface-fitting objective deliberately avoids C/D/A conversion,
but does not ensure wrapping, avoid all collisions, or ensure retention. Do not
interpret its improvement as improved grasp success.

For a task-defined reference configuration, supply `--target-npz path/to/master_log.npz`.
Choose this reference from development data in the same scenario before evaluating
held-out designs. It supplies a fixed object-relative target for joints and tip.
Multiple valid grasp shapes are possible; one reference need not be optimal.

Inspect `summary.json` and the detailed `pose_results.json`:

- `change_cosine`: 1 means predicted/simulated changes align; 0 is orthogonal;
  negative values indicate opposing directions. Null means below the noise floor.
- `change_rmse_mm` measures change-magnitude/vector error. Absolute pose RMSE is
  reported separately; small absolute error does not imply correct derivatives.
- Compare descent's `simulated_loss_change_m2` with reverse and random steps at
  both sizes. Negative means the same pose objective improved in simulation.
- Check `effective_step_norm`: projection onto feasible designs and clipping can
  change step lengths, so nominally equal steps need not be equally large.
- `valid_pairs` and `resolved_directions` show missing/too-small evidence.
  Initial floors (0.01 mm vector norm, 1e-8 m² loss change) are provisional;
  confirm them against repeat/timestep-refinement tests before a full study.

Then use `--methods conditional_diffusion,pose_dgdm_gs1 --anchors 2 --workers 7`
in a new output directory (28 simulations, four waves; about 10 minutes if the
first measured runtime holds, potentially much longer on slow designs).
Repeat on other `--scenario` values and seeds before interpreting improvement rates.

The clean checkpoint tests clean-design gradients. To test the noisy checkpoint
at clean t=0, pass its path and `--noise-conditioned`. This does **not** validate
guidance at noisy diffusion timesteps 3/6, or the effect of the actual DDIM update.
Those require a subsequent sampler-level ablation. No noisy vector is sent to the
physical simulator in this pilot.

## 2. Reuse V20 for approximate grasp screening: seconds

```bash
.venv/bin/python -m benchmarks.grasp_screen \
  --source outputs/from_links_v20_force_screened_fast24h/specialists \
  --thresholds benchmarks/grasp_thresholds_pilot.json \
  --output outputs/grasp_screen_small
```

The supplied thresholds are **provisional engineering examples**, not calibrated
retention criteria. All listed requirements in a profile must pass on each trial:

| Profile | Contacts | Span | Each direction's support | Each opposing-force projection |
|---|---:|---:|---:|---:|
| geometry_loose | 5 | 120° | — | — |
| geometry_primary | 10 | 180° | — | — |
| geometry_strict | 15 | 210° | — | — |
| support_primary | 10 | 180° | 0.9 | — |
| load_proxy_primary | 10 | 180° | 0.9 | 5 N |

Support and force use the **weakest of left/right/down**, not a mean that lets one
strong direction hide another weak direction. The force value is a positive
projection sum, not a net-wrench equilibrium test. The current disturbance rollout
is a lightweight nodal post-process without full Cosserat internal mechanics;
passing it is not a sustained-hanging result. No new full retention simulator is
implemented by this pilot.

`report.json` contains candidate-pool rates, threshold sensitivity, percentage-point
and relative differences versus diffusion-only, and budget counts. A zero baseline
makes relative improvement undefined (null). Completed failures and missing metrics
fail. Jobs not yet run are absent; use completed studies and check coverage before
comparing. Raw contacts depend on discretization: keep it consistent or later use
physical contact-length coverage. All seven V20 specialist methods currently use
the same recorded trial count; this does not alone establish equal generation cost.

The separate selected-design report picks the highest nominal simulator utility
per method/seed/scenario, then reports its recorded nonnominal-condition passes.
Missing selected conditions (among conditions observed for that scenario) and
absent nominal winners fail. This models nominal screening followed by perturbation
evaluation; it is exploratory because these conditions are already part of V20.
It does not select separate winners for each threshold or metric. For confirmatory
evaluation, freeze selection, thresholds, and guidance scale before fresh conditions.

Seed-bootstrap intervals are omitted for a single seed. Correlated perturbations
are not treated as independent seeds. Tiny seed counts give unreliable intervals;
do not infer significance from the pilot. Method differences are descriptive and
unpaired. For a full study use matched seed/scenario coverage and paired uncertainty.

## 3. Lightweight calibration preparation

The screening command also writes `retention_labels_template.csv`: up to four
random successful simulations per method, without using their scores for selection.
Run a separate physical or validated dynamic retention test on these designs under
the corresponding conditions. Fill `retention_success` with 0/1; visual wrapping
alone is not a retention label. Record the load, duration, release constraints, and
slip tolerance in your experiment notes. A fixed branch permits finger/body slip
measurement as long as the finger/body is not artificially fixed in that direction.

```bash
.venv/bin/python -m benchmarks.grasp_screen \
  --source outputs/from_links_v20_force_screened_fast24h/specialists \
  --thresholds benchmarks/grasp_thresholds_pilot.json \
  --labels outputs/grasp_screen_small/retention_labels_template.csv \
  --output outputs/grasp_screen_calibration
```

The report gives confusion counts, precision and recall for every profile against
independent labels. `split=calibration` is for choosing thresholds;
`split=validation` is for untouched labels after freezing thresholds. There is no
automatic threshold optimization to make DGDM win. Do not reuse a design's closely
related perturbations across calibration and validation. Without independent labels,
this stage provides sensitivity analysis only, not calibration to true retention.

## Verification

```bash
.venv/bin/python -m unittest benchmarks.tests.test_pose_pilot benchmarks.tests.test_pose_dynamics
```
