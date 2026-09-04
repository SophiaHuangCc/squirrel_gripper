# Pareto preference-guided diffusion

This implements the preference-guided multi-objective diffusion pipeline for
SquirrelGripper as a separate benchmark method. It uses only three maximization
objectives, always in this order:

1. disturbance resistance in `[0, 1]`;
2. contact coverage `log1p(num_contacts) / log1p(n_elements)`;
3. angular span `clip(angular_span / 360, 0, 1)`.

Curl time and curl speed are not loaded, ranked, trained, or evaluated here.

## Pipeline

Run commands from `sg_ws` using the TendonForces environment.

```bash
# 1. Aggregate simulator rollouts by exact design and make design-disjoint splits.
python -m pareto_diffusion.prepare_data \
  --data_dir TendonForces/runs/exp3 \
  --output pareto_diffusion/runs/objectives.npz \
  --min_scenarios 1 --split_seed 0

# 2. Train noisy-timestep pair preferences from Pareto rank + crowding distance.
python -m pareto_diffusion.train_preference \
  --table pareto_diffusion/runs/objectives.npz \
  --prior dgdm/runs/prior/last.pt \
  --save_dir pareto_diffusion/runs/preference

# 3. Sample from the same unconditional prior using preference gradients.
python -m pareto_diffusion.generate \
  --prior dgdm/runs/prior/last.pt \
  --preference pareto_diffusion/runs/preference/best.pt \
  --table pareto_diffusion/runs/objectives.npz \
  --output outputs/pareto/candidates/pareto_s0.npz \
  --num_samples 256 --guidance_scale 0.1 --seed 0

# 4. Give the candidates the same simulator benchmark as every other method.
python -m benchmarks.run_sim_benchmark \
  --candidates outputs/pareto/candidates/pareto_s0.npz \
  --output_dir outputs/pareto/runs/pareto_s0

# 5. Measure the simulator-evaluated front.
python -m pareto_diffusion.evaluate_front \
  --records outputs/pareto/runs/pareto_s0/records.jsonl \
  --reference_table pareto_diffusion/runs/objectives.npz \
  --output outputs/pareto/front_metrics.json
```

The generated NPZ uses `benchmarks.candidates.save_candidates`, so caching,
scenario evaluation, and accounting work without an adapter. Its selection score
is the learned preference score against training-only Pareto references; simulator
test outcomes are never used to select the deployable top candidate.

## Experimental controls

- Build separate objective tables for a cell specialist, family specialist, and
  generalist unless every design has the same intended scenario aggregation.
- Set `--min_scenarios` to the expected number of rollouts per design for final
  generalist experiments. A value of one is only convenient for smoke tests.
- Never combine train and benchmark-test archives when preparing preferences.
- Compare guidance scales including zero, using the same prior, DDIM steps, seeds,
  sample counts, simulator scenarios, and simulator budget.
- Hypervolume assumes normalized maximization objectives and defaults to reference
  point `(0, 0, 0)`. IGD uses the feasible held-out table as an empirical reference,
  so label it as empirical IGD rather than distance to the unknown true front.
