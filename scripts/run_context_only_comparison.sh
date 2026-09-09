#!/usr/bin/env bash
# Match the completed scene-0 comparison, changing only the trained diffusion prior.
set -Eeuo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CONTEXT_OUTPUT=outputs/context_only_scene00_scale1
mkdir -p "$CONTEXT_OUTPUT"
exec 9>"$CONTEXT_OUTPUT/run.lock"
flock -n 9 || { echo "This comparison is already running."; exit 1; }
prepare=()
if [[ "${1:-}" == --prepare-only ]]; then
  prepare=(--dry_run)
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/run_context_only_comparison.sh [--prepare-only]"; exit 2
fi
.venv/bin/python - <<'PY'
import json
from pathlib import Path
config=json.loads(Path('outputs/direct_scene00_scale1/benchmark_effective_config.json').read_text())
Path('outputs/context_only_scene00_scale1/evaluation_config.json').write_text(json.dumps(config,indent=2)+'\n')
PY
.venv/bin/python -u -m benchmarks.run_baselines \
  --config outputs/direct_scene00_scale1/effective_config.json \
  --benchmark_config "$CONTEXT_OUTPUT/evaluation_config.json" \
  --output_dir "$CONTEXT_OUTPUT" \
  --methods conditional_diffusion,dgdm --seeds 0 --candidate_budget 16 \
  --target_scenario_id approach_radius:00 \
  --diffusion_conditioning context_only \
  --diffusion_checkpoint outputs/context_only_v20/diffusion_context_only_15/best.pt \
  --dgdm_dynamics_checkpoint outputs/from_links_v16_pose_dgdm/pose_dynamics_noisy_t036/best.pt \
  --diffusion_batch_size 16 --diffusion_inference_steps 5 \
  --dgdm_guidance_scale 1 --dgdm_guidance_timesteps 0,3,6 \
  --device cpu --run_benchmark --num_workers "${NUM_WORKERS:-24}" --timeout 1200 --resume "${prepare[@]}"
if [[ ${#prepare[@]} -eq 0 ]]; then
  .venv/bin/python -m benchmarks.compare_conditioning_runs
fi
