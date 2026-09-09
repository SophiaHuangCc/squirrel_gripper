# Requested-performance conditioning probe

Keep the six environment inputs; vary only the three requested-performance values.
Environmental context is distinct from the optimization goal. The existing pose
dynamics model does not receive requested-performance targets as inputs.

## Quick frozen-checkpoint probe

```bash
bash scripts/run_performance_target_probe.sh --run_benchmark
```

This resumes the prepared candidates and runs 32 simulations: 16 designs with
targets (0.8,0.8,0.8), and 16 with targets (0,0,0). Both arms use the same checkpoint,
Gaussian-noise seed, batch size, scene-0 nominal environment and V20 evaluation.
All generated designs are evaluated; there is no guidance or surrogate ranking.
Output: `outputs/performance_target_probe/probe_summary.json` and per-arm results.
Expect roughly 5–10 minutes on this machine; simulation times vary. Completed
simulations are cached by the normal resume protocol. Do not start duplicate runs.

The default command without `--run_benchmark` only prepares designs. Preparation
and dry-run validation found that all 16 generated design vectors changed when
targets were zeroed. This establishes dependence, not better grasp performance.

Zero is not a learned missing-input token. For the current trained checkpoint it
is a different performance request, potentially outside its familiar conditioning
distribution. Poor performance of this arm cannot establish that requested targets
should be removed, or predict the quality of a retrained six-input prior. The probe
does not explain whether DGDM's advantage is small because of conditioning.

## Proper six-input training experiment

`context_only` masks the last three condition channels in both training and sampling,
while preserving all six environment channels. Network width remains nine for
compatibility, with only six active inputs. Start matched training locally:

```bash
bash scripts/train_context_only_v20.sh
```

This is a full training run, not the short probe. It uses the conditional prior's
training settings, dataset, split seed, 15 training timesteps and five inference steps.
No training has been started automatically. For later benchmark runs, provide:

```text
--diffusion_conditioning context_only
--diffusion_checkpoint outputs/context_only_v20/diffusion_context_only_15/best.pt
```

The loader checks training metadata; this flag cannot reinterpret an old conditional
checkpoint as a trained context-only model. Compare properly trained conditional and
context-only priors, each with guidance off/on, using equal generated sample counts
and unchanged simulator evaluation. One-epoch smoke training is not a fair replacement
for a fully trained prior.
