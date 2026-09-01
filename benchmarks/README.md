# Squirrel Gripper Simulation Benchmark V1

## Analyze a completed multi-objective specialist study

The specialist sweep is stored as
`STUDY_DIR/{combined,contact_only,disturbance_only}/specialists/approach_radius-XX`.
Each scenario directory contains `candidates/`, per-method/seed `runs/`, and a
`summary/`. One candidate file may contain several proposed designs, but the
primary benchmark normally evaluates only the preselected top design because
`--benchmark_top_k 1` is used.

Do not select a final design by the surrogate `selection_score`. That value is
used before simulation. Select it by the full-simulator `utility` for the
declared objective. The following command recursively discovers completed runs
and creates flat tables, including the best seed per method and the best method
and seed overall for every scenario:

```bash
python -m benchmarks.analyze_study \
  --study_dir "$BENCHMARK_DIR/nine_scenario_objectives"
```

The most useful outputs are:

- `method_by_scenario.csv`: mean/std across seeds for each method and scene;
- `method_overall.csv`: aggregate method comparison across the study;
- `best_per_method_scenario.csv`: simulator-best seed for each method/scene;
- `best_overall_per_scenario.csv`: one simulator-best design across all methods;
- `timing_summary.csv`: proposal time and simulator time, reported separately;
- `surrogate_gap_summary.csv`: surrogate-minus-simulator diagnostic;
- `all_rollouts.csv`: a flat index containing paths to every result and master log.

To regenerate an MP4 only for the single simulator-best design across all
methods in every objective/scenario, add `--render_best_overall`. Use
`--dry_run` first to print and validate all render commands without simulating:

```bash
python -m benchmarks.analyze_study \
  --study_dir "$BENCHMARK_DIR/nine_scenario_objectives" \
  --render_best_overall \
  --num_workers 1 \
  --timeout 1800 \
  --dry_run
```

Remove `--dry_run` to produce the videos. Rendering one best design for each of
three objectives and nine scenes launches 27 new rendered simulations. These
are visualization reruns and should not replace the cached numerical results.
Use `--objectives combined` or, for example,
`--scenario_ids approach_radius:04` to render a smaller declared subset.
Add `--measure_energy` to an actual render rerun to log tendon displacement,
positive actuator work, and net actuator work in `final_design_energy.csv`.
This opt-in calculation is not used during dataset generation, model training,
candidate selection, or the primary utility benchmark.
Use `--render_best_per_method --measure_energy` instead when the post-hoc study
should compare the best simulated seed from every method rather than only the
single best method overall; results are written to
`per_method_final_design_energy.csv`.

### Analyze a generalist study

A generalist is one fixed design evaluated on every scenario. Its seed must be
selected by mean full-simulator utility across the complete grid, not by taking
a different best design in each scene. Point the same analyzer at the directory
containing the completed generalist runs:

```bash
python -m benchmarks.analyze_study \
  --study_dir "$GENERALIST_STUDY_DIR" \
  --protocol generalist
```

In addition to the common rollout, timing, and calibration tables, this writes:

- `generalist_candidate_summary.csv`: one fixed design/seed across all cells,
  including mean, cross-scenario standard deviation, CVaR20, and worst cell;
- `generalist_method_summary.csv`: mean and standard deviation across seeds;
- `best_generalist_per_method.csv`: the simulator-best seed for each method;
- `best_generalist_overall.csv`: one best method/seed by all-scenario mean.

Render the single best generalist over the same complete grid with
`--render_best_generalist`; use `--render_best_generalist_per_method` to compare
one generalist from every method. Either can be combined with `--measure_energy`.
Use `--dry_run` first when checking paths and scenario coverage.

### Unattended specialist + generalist study run

`scripts/run_design_studies.sh` exports the lab-machine defaults,
runs or reuses a separate noise-and-timestep-conditioned dynamics checkpoint,
runs all three objective profiles for specialists and generalists, analyzes both
studies, optionally performs final-design video/energy reruns, and runs the
paired conditional-DGDM guidance-scale sweep. It uses a V3
result root so corrected surrogate-clamping experiments do not mix with V2.
Run it in a persistent terminal such as `tmux`; progress and final status are
stored under `outputs/from_links_v3/study_logs/`. The default
keeps the specialist and generalist studies sequential because each benchmark
already runs 30 simulator groups concurrently. Set `RUN_STUDIES_IN_PARALLEL=1`
only on a machine that can safely support roughly twice that load.
The default `TRAIN_DGDM_DYNAMICS=auto` trains
`outputs/from_links_v3/dynamics_noisy/best.pt` only when it is missing. Set it
to `always` for an intentional retrain or `never` to require an existing noisy
checkpoint. The clean checkpoint remains unchanged and is used by Adam,
CMA-ES, random search, and final diffusion candidate ranking; the noisy
checkpoint is passed only as `--dgdm_dynamics_checkpoint`.

> **Current default protocol (V2):** `scenarios_v2.json` replaces the original
> 28-cell multi-family study with a 25-cell `approach_deg × cyl_rad` grid. The
> approach values are `5, 25, 45, 65, 85°` (every other level across the full
> dataset-supported range), and the radii are all five dataset levels:
> `0.015, 0.020, 0.025, 0.030, 0.035 m`. The old `scenarios_v1.json` is retained
> only to reproduce earlier experiments. Under the default automatic evaluation
> scope, an exact specialist is simulated only on its target cell; a generalist
> is selected and simulated over all 25 cells. Raw contact count, disturbance,
> and angular span in degrees are reported alongside normalized utility.
> A predeclared compact alternative, `scenarios_v2_compact.json`, uses the
> minimum, center, and maximum training-supported levels for a 3×3 grid. V2
> rejects family-specialist selection because its single family is the entire
> grid and would therefore duplicate the generalist target.

## Start here: what point 1 is doing

Point 1 builds an **exam for finger designs**. It does not yet build the
student that learns how to design the best finger.

Your interpretation is correct: if we have designs A and B and scenarios 1
through 5, the benchmark runs every fixed design in every scenario:

| Fixed design | Scenario 1 | Scenario 2 | Scenario 3 | Scenario 4 | Scenario 5 |
|---|---:|---:|---:|---:|---:|
| Design A | score A1 | score A2 | score A3 | score A4 | score A5 |
| Design B | score B1 | score B2 | score B3 | score B4 | score B5 |

One cell in this table is one simulator **rollout**. During a rollout, the
design does not change. The scenario changes the deployment conditions around
the finger, such as its installed angle, landing direction, branch radius, or
initial offset.

This full matrix is important. Testing A only in scenario 1 and B only in
scenario 2 would not tell us whether a score difference came from the design or
from the scenario. Giving every design the same exam makes the comparison fair.

The implemented data flow is:

```text
design method -> candidate designs -> common 16D From Links format
                                      |
                                      v
                         each candidate x each scenario
                                      |
                                      v
                       simulator metrics for every rollout
                                      |
                                      v
                  candidate summary and method comparison
```

### Four terms that should not be mixed together

| Term | Meaning in this repository | Example |
|---|---|---|
| **Design** | One fixed physical/actuation configuration represented by 16 From Links values | link lengths, joint lengths, stiffness ratios, cross-section, tension |
| **Candidate** | A design proposed for evaluation | “candidate 7” is one particular finger |
| **Method** | An algorithm or rule that proposes candidates | reference, random, retrieval, diffusion, Adam |
| **Scenario** | A deployment condition supplied by the benchmark | a 75-degree landing onto a thin branch |

“Candidate” therefore does not mean a candidate scenario. It means a candidate
finger design. A method may propose one candidate or a budget of `K` candidates.

### What reference, random, and retrieval mean

- **Reference** proposes the existing manufactured/default finger. It is the
  control: can a new method outperform the design we already have?
- **Random** samples feasible 16D From Links designs from the allowed design
  ranges. In the current top-1 implementation, the first seeded random design
  is precommitted for evaluation; it is not secretly selected using test
  simulator results.
- **Retrieval** does not invent a new finger. It searches the existing `.npz`
  dataset for designs collected under environmental conditions closest to a
  target condition and reuses those designs. It asks whether a generator is
  actually better than looking up an old, similar example.

`TendonForces/runs/exp27` is useful for checking that retrieval and the complete
pipeline run, but it is too small and repetitive to support a scientific claim
about retrieval quality.

### What a method comparison does

First, `candidate_summary.csv` creates one row for each fixed candidate. It
aggregates that candidate's scores across all scenarios. This answers questions
such as:

- Does design A have a higher average score than design B?
- Does A fail badly in its worst conditions even though its average is high?
- Is B especially weak for the orientation family?

The main aggregate statistics are:

- **mean utility:** average performance over the scenario suite;
- **CVaR20:** average performance in the worst 20% of scenarios;
- **worst-family mean:** performance in the candidate's weakest scenario
  family;
- **failure rate:** fraction of simulator rollouts that did not complete.

Then `method_summary.csv` groups candidates by the method that proposed them.
There are two different experimental protocols, and their distinction matters:

1. **Deployable top-1 comparison (primary):** each method proposes candidates,
   selects one without seeing test-scenario simulator scores, and that one
   design takes the 28-scenario exam. Use `--benchmark_top_k 1`.
2. **Oracle best-of-K (diagnostic only):** simulate all `K` candidates, then
   report the candidate with the best benchmark mean. Use
   `--benchmark_top_k K`. This uses the test simulator to choose the winner,
   costs `K` times more, and must not be presented as a fair one-shot method.

At present, if several candidates from a method are evaluated, the method
summary reports the best simulated candidate as well as candidate-average
statistics. Thus a multi-candidate method summary is an oracle diagnostic unless
the winner was selected beforehand by a training-only model.

### Specialist versus general-purpose design

A **specialist** is proposed for one scenario or one scenario family. We still
test it on every scenario: its target score measures specialization, while the
remaining cells measure transfer.

A **generalist** is one fixed design deliberately selected for good performance
over a distribution of scenarios. It also takes exactly the same exam. A useful
story is then: “the specialist is best on its target, but the generalist gives
up only a little peak performance and improves average, worst-case, and
worst-family performance.”

The benchmark can evaluate either kind of design. Adam, CMA-ES, random search,
conditional diffusion, and dynamics-guided diffusion all support cell,
family, and generalist selection. For a multi-scenario diffusion target, the
conditional prior uses the target-set centroid and every proposal is ranked by
mean surrogate utility over the complete target set. DGDM additionally averages
its differentiable guidance objective over the complete target set at every
denoising step.

### What point 1 has and has not completed

Implemented for point 1:

- a versioned 28-scenario core suite;
- a common candidate format for comparing different design sources;
- the full candidate-by-scenario simulator runner;
- stable IDs, caching, failure records, and resumable execution;
- raw per-rollout metrics and candidate/method summaries;
- mean, lower-tail, worst-family, confidence-interval, and plot reporting.

Still needed before making final experimental claims:

- calibrate success thresholds and simulator metrics against real grasps;
- add repeated physics/random seeds rather than one deterministic rollout per
  core cell;
- freeze scenario-disjoint train/validation/test manifests;
- produce actual specialist and generalist candidates with point-2/3 methods;
- run a sufficiently large dataset and candidate/seed budget.

In short, point 1 provides the controlled test and reporting machinery. It does
not by itself prove which design-generation method is best.

## Research question

Can a data-driven method produce a squirrel-finger design that is better than
fixed, retrieval, random-search, and surrogate-optimization baselines for:

1. one specified landing/grasp condition (a **specialist**), and
2. a distribution of landing/grasp conditions (a **generalist**)?

A benchmark task is a distribution of deployment conditions, not an optimized
design parameter. The benchmark therefore keeps the following roles separate.

### Design and actuation variables

These may be selected by a design method and then remain fixed throughout an
evaluation episode: three joint stiffness ratios, four free-link lengths,
three joint lengths, finger cross-section/radius, total finger length, tendon
tension, ankle wrap radius, and ankle stiffness.

The common model/candidate representation has 16 coordinates, but V2 does not
optimize every coordinate. Joint lengths, base radius (`0.01 m`), ankle wrap
radius, and ankle stiffness have equal lower/upper bounds and therefore remain
fixed. Keeping them in the vector preserves one dynamics/diffusion checkpoint
contract; it does not make them active design degrees of freedom.

Tension is treated as part of morphology-actuation co-design in V1. A later
ablation should fix tension across all methods to isolate morphology quality.

### Deployment-condition variables

These are specified by the benchmark and must not be optimized independently
for each evaluation rollout:

- `approach_deg`: installed/rest orientation of the finger;
- `landing_approach_deg`: direction of the landing trajectory in the XZ plane;
- `cyl_rad`: branch radius;
- `initial_x_gap`: initial horizontal offset from the branch center;
- `landing_height`: available landing/drop distance;
- `landing_speed`: prescribed impact/approach speed;
- `body_mass`, `mu_contact`: payload and contact uncertainty.

The current optimizer treats `approach_deg` as optimizable. Benchmark runs must
instead hold it to the scenario value. Optimizing mounting angle can be reported
separately as a morphology-plus-installation co-design ablation.

## Scenario suite

The canonical values are stored in `scenarios_v1.json`. The four core families
are deliberately factorized, so a failure can be attributed to one source.

| Family | Purpose | Grid | Rollouts/design |
|---|---|---:|---:|
| Nominal | Regression and sim-to-real anchor | one manufactured condition | 1 |
| Orientation | Separates installed finger angle from landing direction | 3 approach x 3 landing angles | 9 |
| Branch/offset | Tests branch-scale and placement generalization | 3 radii x 3 gaps | 9 |
| Landing severity | Tests approach direction and available landing distance | 3 heights x 3 landing angles | 9 |

The core suite has 28 deterministic cells per design. The robustness tier repeats
selected cells under nominal, slippery-heavy, and light-payload physics. Keep
the core and robustness results separate: environmental coverage and physics
uncertainty answer different questions.

## Specialist and generalist tasks

- **Cell specialist:** designed for one exact scenario cell and evaluated both
  on that cell and on all other cells. This measures peak performance and
  transfer degradation.
- **Family specialist:** designed for the mean performance over every cell in
  one family (for example, all nine orientation cells).
- **Generalist:** one fixed design selected using all 28 core cells. Optimize
  both mean utility and lower-tail robustness (CVaR at 20%).
- **Leave-one-family-out generalist:** designed on three families and evaluated
  on the held-out family. This is the main out-of-distribution test.

Task-specific versus general-purpose results should be presented as a trade-off:
specialist score on its target, generalist mean score, worst-family score, and
specialist-to-generalist regret.

## Metrics

Save raw simulator measurements for every rollout. Do not rank methods using
the dynamics prediction.

Primary continuous metrics:

- disturbance resistance score (maximize);
- normalized contact coverage `log1p(num_contacts) / log1p(n_elements)`;
- normalized wrap `clip(angular_span_deg / 180, 0, 1)`;
- tendon tension and total energy (minimize);
- maximum penetration and simulation failure rate (minimize).

The V1 scalar utility, used only for ranking when a scalar is required, is:

The V2 primary utility is
`U = 0.55 disturbance + 0.35 contact + 0.10 wrap`.

Always report the three components beside `U`. Report mean, median, standard
deviation, 20%-CVaR (mean of the worst 20%), worst-family mean, and bootstrap
95% confidence intervals across scenario cells.

A binary success rate should be added only after thresholds are calibrated
against real successful and failed grasps. Until then, continuous metrics are
the primary claims; arbitrary success thresholds would weaken the sim-to-real
story.

## Baselines

Use the same feasible 16D From Links design domain and candidate budget.

1. **Reference:** the manufactured/run.sh design, unchanged for every task.
2. **Random:** feasible uniform random sampling followed by the common geometry
   projection. The current primary top-1 run precommits a seeded random sample.
   Dynamics-ranked random search remains point-2 work.
3. **Dataset retrieval:** retrieve designs whose recorded environment is nearest
   to the target environment. This tests whether generation is better than
   reusing an existing, similar design.
4. **Gradient descent:** direct Adam optimization through the dynamics model.
5. **CMA-ES:** gradient-free optimization through the same dynamics model.
6. **Unguided diffusion:** sample from the design prior with guidance scale 0.
7. **Dynamics-guided diffusion:** identical diffusion checkpoint, seeds, and
   candidate budget, with dynamics guidance enabled.

For fairness use `K=16` candidates and five method seeds initially. Report both
wall-clock proposal time and the number of surrogate and simulator evaluations.
The primary deployable protocol is: propose `K`, rank without test simulation,
then simulate the top one. Oracle-best-of-16 is secondary because it spends 16
test simulations and cannot represent one-shot deployment.

## Required ablations

- guidance scale (including zero);
- number of denoising steps;
- candidate budget;
- fixed versus co-designed tension;
- target-cell versus family-aggregate guidance;
- in-distribution versus leave-one-family-out scenarios;
- dynamics predicted score versus simulator score (surrogate exploitation gap).

### Paired guidance-scale sweep

The current task/target-conditioned DGDM adaptation can be swept at five scales
with identical initial noise, checkpoints, sample budget, DDIM steps, and seeds:

```bash
TendonForces/.venv/bin/python -m benchmarks.run_guidance_sweep \
  --output_dir outputs/guidance_sweep_v1 \
  --diffusion_checkpoint PATH/TO/DIFFUSION/best.pt \
  --dynamics_checkpoint PATH/TO/DYNAMICS/best.pt \
  --dgdm_dynamics_checkpoint PATH/TO/NOISE_CONDITIONED_DYNAMICS/best.pt \
  --scales 0,0.1,1,2,10 \
  --seeds 0,1,2,3,4 \
  --candidate_budget 16 \
  --num_samples 256 \
  --inference_steps 20 \
  --generalist \
  --run_benchmark \
  --benchmark_top_k 1 \
  --num_workers 4
```

The scale-zero arm is the paired unguided control. Results are labeled
`conditional_dgdm_gs0`, `...gs0p1`, `...gs1`, `...gs2`, and `...gs10` so the
summary cannot collapse different scales into one method. This runner studies
the existing conditional adaptation; the separate unconditional DGDM path must
be swept with its own unconditional prior and noisy-timestep dynamics checkpoint.

## Experimental story

1. Validate simulator metrics against real successful/failed grasps.
2. Show specialist designs outperform generalists on their target cells.
3. Show generalists lose modest peak performance but improve mean, CVaR, and
   worst-family performance across changing landing and branch conditions.
4. Compare search methods under equal proposal budgets.
5. Show dynamics guidance improves diffusion over the identical unguided prior.
6. Finally replace scalar utility with Pareto/preference guidance and report
   hypervolume, IGD, non-dominated fraction, and objective-space diversity.

## Current implementation gaps

- DGDM generalist guidance must aggregate dynamics gradients across a scenario batch.
- The standalone legacy optimizer can optimize `approach_deg`; benchmark-native
  Adam and CMA-ES correctly fix it to the selected task cells.
- The existing conditional generator is not a clean unconditional DGDM prior.
- Existing Pareto code studies input parameters, not the performance-efficiency
  Pareto front, and lacks hypervolume/diversity evaluation.
- Scenario-disjoint train/test manifests and repeated random seeds are not yet
  enforced by the data generator.

## Running the implemented benchmark

All commands below run from the `sg_ws` repository root. Use the Python
interpreter from the TendonForces environment so simulator dependencies remain
available.

Validate configuration and write a manifest without simulating:

```bash
TendonForces/.venv/bin/python -m benchmarks.run_sim_benchmark \
  --candidates outputs/benchmark_v1/candidates/reference_s0.npz \
  --output_dir outputs/benchmark_v1/reference_dry \
  --dry_run
```

Generate reference, random, and nearest-neighbor retrieval candidates:

```bash
TendonForces/.venv/bin/python -m benchmarks.run_baselines \
  --output_dir outputs/benchmark_v1 \
  --methods reference,random,retrieval \
  --retrieval_data_dir TendonForces/runs/exp27/train
```

Run one preselected candidate from each method on the nominal scenario first:

```bash
TendonForces/.venv/bin/python -m benchmarks.run_baselines \
  --output_dir outputs/benchmark_v1 \
  --methods reference,random,retrieval \
  --retrieval_data_dir TendonForces/runs/exp27/train \
  --run_benchmark \
  --benchmark_top_k 1 \
  --families nominal \
  --num_workers 1
```

Remove `--families nominal` for all 28 scenarios. Completed successful rollouts
are cached. Repeating the same command resumes missing/failed work and does not
rerun successful design-scenario pairs.

Adapt existing generated or optimized candidates:

```bash
TendonForces/.venv/bin/python -m benchmarks.run_baselines \
  --output_dir outputs/benchmark_v1 \
  --methods reference \
  --adapt diffusion=generator/runs/sample/generated_candidates.npz \
  --adapt adam=optimization/runs/exp1/disturbance_contact_span_surrogate_only/optimized_candidates.npz
```

Summarize any collection of completed runs directly:

```bash
TendonForces/.venv/bin/python -m benchmarks.summarize \
  outputs/benchmark_v1/runs/*/records.jsonl \
  --output_dir outputs/benchmark_v1/summary
```

The primary deployable comparison uses `--benchmark_top_k 1`. To compute the
diagnostic oracle-best-of-16 curves, run all candidates with
`--benchmark_top_k 16`; this spends sixteen times as many simulator rollouts and
must be labeled separately.

## Point 2: baseline proposal and comparison

Point 2 is implemented for the reference, seeded random, surrogate-ranked
random search, dataset retrieval, Adam, and CMA-ES methods. Random search,
Adam, and CMA-ES use the frozen three-output dynamics model. They never use
core test-simulator results to select their top candidate.

The shared surrogate utility is aligned with the benchmark utility:

```text
0.55 disturbance + 0.35 normalized contact coverage + 0.10 normalized wrap
```

Each optimizer produces `candidate_budget` feasible From Links designs, stores
their surrogate scores and model-evaluation count, and orders them before the
simulation benchmark starts. `--benchmark_top_k 1` therefore evaluates the
preselected top design from each optimizer.

Use `--surrogate_eval_budget N` to give random search, Adam, and CMA-ES the
same approximate number of candidate-scenario dynamics predictions. Without
it, the explicit pool/step/generation settings control compute independently.

First train a new three-output dynamics checkpoint. Old four-output checkpoints
are intentionally incompatible:

```bash
TendonForces/.venv/bin/python dynamics/main.py \
  --mode train \
  --device cpu \
  --data_dir TendonForces/runs/exp27/train \
  --test_data_dir TendonForces/runs/exp27/train \
  --save_dir outputs/point2_smoke/dynamics \
  --batch_size 4 \
  --num_workers 0 \
  --lr 1e-3 \
  --num_epochs 3 \
  --output_dim 3
```

The repeated train directory above is acceptable only for a smoke test. Use
disjoint train and test directories for an experiment.

DGDM additionally requires a separate dynamics checkpoint trained on the
diffusion noise distribution. Use the same training command with a different
`--save_dir` and add:

```bash
  --use_design_noise \
  --num_train_timesteps 100 \
  --num_timesteps_per_batch 4
```

Pass the clean checkpoint as `--dynamics_checkpoint` and the noisy checkpoint
as `--dgdm_dynamics_checkpoint`. The latter is used at every DDIM step; the
former remains the common clean-design ranking model for all methods.

Generate nominal-specialist candidates and validate the benchmark plan without
running the expensive simulator:

```bash
TendonForces/.venv/bin/python -m benchmarks.run_baselines \
  --output_dir outputs/point2_smoke/nominal \
  --methods reference,random,random_search,retrieval,adam,cma_es \
  --candidate_budget 4 \
  --seeds 0 \
  --retrieval_data_dir TendonForces/runs/exp27/train \
  --dynamics_checkpoint outputs/point2_smoke/dynamics/best.pt \
  --device cpu \
  --adam_steps 20 \
  --random_pool_size 32 \
  --cma_generations 10 \
  --cma_popsize 8 \
  --run_benchmark \
  --benchmark_top_k 1 \
  --families nominal \
  --dry_run
```

Remove `--dry_run` to execute the nominal smoke simulations. For the full
28-scenario exam, also remove `--families nominal`.

For final nominal-specialist experiments, a suitable initial budget is:

```bash
TendonForces/.venv/bin/python -m benchmarks.run_baselines \
  --output_dir outputs/benchmark_v1/nominal_specialist \
  --methods reference,random,random_search,retrieval,adam,cma_es \
  --candidate_budget 16 \
  --seeds 0,1,2,3,4 \
  --retrieval_data_dir TendonForces/runs/exp27/train \
  --dynamics_checkpoint outputs/dynamics_three_metric/best.pt \
  --device cpu \
  --adam_steps 300 \
  --surrogate_eval_budget 4096 \
  --cma_generations 100 \
  --cma_popsize 32 \
  --run_benchmark \
  --benchmark_top_k 1 \
  --num_workers 1
```

Use one of these mutually exclusive target selectors:

- no selector: nominal-cell specialist;
- `--target_scenario_id orientation:04`: exact-cell specialist;
- `--target_family orientation`: family specialist;
- `--generalist`: optimize mean surrogate utility over all 28 core cells.

The summary directory contains:

- `candidate_summary.csv`: one report card per simulated design;
- `method_summary.csv`: one selected/oracle result per method seed;
- `method_aggregate.csv`: mean and standard deviation across method seeds;
- `method_comparison.png`: mean, lower-tail, and worst-family comparison.

### Current surrogate-coverage limitation

The frozen dynamics model currently receives mounting angle, branch radius,
landing height, landing speed, and horizontal gap. It does not receive landing
trajectory direction, payload, or friction. The simulator benchmark can still
test those variables, but Adam and CMA-ES cannot yet adapt their proposal score
to them. Consequently, a claimed all-scenario generalist should wait until
those environment variables are added to the dynamics dataset and model input.

DGDM is deliberately not part of point 2. It will plug into the same candidate
schema and benchmark as the point-3 method, after these baselines are frozen.
