#!/usr/bin/env bash
# Frozen-checkpoint sensitivity test: requested metrics .8 versus zero.
# This is NOT a retrained six-input model; zero is a different performance request.
# Default generates only. Add --run_benchmark to simulate 16 nominal designs per arm.
set -Eeuo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PROBE_ROOT="$PROJECT_DIR/outputs/performance_target_probe"
mkdir -p "$PROBE_ROOT"
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('outputs/performance_target_probe')
config=json.loads(Path('benchmarks/scenarios_v6_force_four.json').read_text())
config['fixed_simulation']['disturbance_score_mode']='directional_support'
config['families'][0]['grid']={key:[values[0]] for key,values in config['families'][0]['grid'].items()}
config['evaluation']['physical_condition_ensemble']['variants']=[{'id':'nominal','offsets':{}}]
(root/'nominal_config.json').write_text(json.dumps(config,indent=2)+'\n')
PY
for arm in requested_08 zeroed_targets; do
  target=0.8
  [[ "$arm" != zeroed_targets ]] || target=0
  .venv/bin/python -u -m benchmarks.run_baselines \
    --config benchmarks/scenarios_v5_robust_four.json \
    --benchmark_config "$PROBE_ROOT/nominal_config.json" --output_dir "$PROBE_ROOT/$arm" \
    --methods conditional_diffusion --seeds 0 --candidate_budget 16 \
    --target_scenario_id approach_radius:00 --evaluation_scope all \
    --diffusion_checkpoint outputs/from_links_v14_fixed10_span360/diffusion_conditional_15/best.pt \
    --diffusion_batch_size 16 --diffusion_inference_steps 5 \
    --target_contacts "$target" --target_disturbance "$target" --target_angular_span "$target" \
    --device cpu --num_workers 16 --timeout 1200 --resume "$@"
done
.venv/bin/python - <<'PY'
from pathlib import Path
import json, statistics
from benchmarks.candidates import load_candidates
root=Path('outputs/performance_target_probe')
arms=('requested_08','zeroed_targets')
a,b=[load_candidates(root/arm/'candidates/conditional_diffusion_s0.npz')['design_params'] for arm in arms]
print('Designs with any parameter changed:',int((abs(a-b)>1e-7).any(axis=1).sum()),'of',len(a))
results={}
for arm in arms:
    paths=list((root/arm/'runs').rglob('benchmark_result.json'))
    rows=[json.loads(p.read_text()) for p in paths]
    good=[r for r in rows if r['status']=='ok']
    results[arm]={'finished_trials':len(rows),'successful_simulations':len(good),
        'mean_v20_utility':statistics.mean(r['utility'] for r in good) if good else None}
    print(arm,results[arm])
(root/'probe_summary.json').write_text(json.dumps({'interpretation':'Frozen-checkpoint target sensitivity; zero is not a learned missing-input token. Not a context-only model comparison.', 'arms':results},indent=2)+'\n')
PY
