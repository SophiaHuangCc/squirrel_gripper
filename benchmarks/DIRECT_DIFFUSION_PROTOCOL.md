# Direct diffusion candidate evaluation

Conditional diffusion and DGDM now generate exactly `candidate_budget` designs,
in the same batch partition and using the same seed/inference schedule when run
together. Every design is saved in generation order and sent to simulation.
There is no clean-surrogate evaluation, candidate ranking, or oversampled pool.
DGDM still uses the noisy dynamics model internally for guidance; its objective
remains the current pose-derived C/D/A utility. Pose-loss guidance has not replaced it.

`--diffusion_num_samples` in run_baselines and `--num_samples` in run_guidance_sweep
are deprecated and ignored in favor of the candidate budget, with a message when
values differ. Omit `--benchmark_top_k` or set it to the full candidate budget.
Old ranked candidates cannot be resumed as direct candidates. Use a new directory.
No unconditional training is launched by this change.

Example: scene 0, 16 candidates per method, five initial conditions each (160
simulation trials total). Run without `--run_benchmark` to generate candidates only.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.run_baselines \
  --config benchmarks/scenarios_v5_robust_four.json \
  --benchmark_config benchmarks/scenarios_v6_force_four.json \
  --output_dir outputs/from_links_v22_direct/scene00 \
  --methods conditional_diffusion,dgdm --seeds 0 --candidate_budget 16 \
  --target_scenario_id approach_radius:00 \
  --diffusion_checkpoint outputs/from_links_v14_fixed10_span360/diffusion_conditional_15/best.pt \
  --dgdm_dynamics_checkpoint outputs/from_links_v16_pose_dgdm/pose_dynamics_noisy_t036/best.pt \
  --diffusion_batch_size 16 --diffusion_inference_steps 5 \
  --dgdm_guidance_scale 1 --dgdm_guidance_timesteps 0,3,6 \
  --device cpu --run_benchmark --num_workers 8 --timeout 1200
```

Candidate metadata records sample count, batch size/count, denoising evaluations,
guidance evaluations, zero ranking evaluations, and elapsed proposal time. Counts
of model evaluations are candidate-level (guidance also counts each condition),
not comparable FLOP estimates. DGDM necessarily adds gradient computation. Equal
candidate counts and simulator trials do not imply equal total compute. Adam and
CMA-ES retain their optimizer iterations/populations and surrogate objectives;
report these budgets and runtimes alongside their final candidate counts.

The generalist continues to use the averaged condition vector for proposals;
guidance sees actual conditions. This change isolates removal of ranking. The
primary comparison should report all generated candidates, including simulator
failures, rather than separate best designs per metric. Any winner selection for
deployment needs subsequent evaluation on fresh conditions.
