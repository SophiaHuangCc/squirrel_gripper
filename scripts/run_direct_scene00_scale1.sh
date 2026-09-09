#!/usr/bin/env bash
# Resume the 16-design, five-condition comparison and produce its final report.
set -Eeuo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Avoid starting a duplicate of the already running comparison.
if ! .venv/bin/python - <<'PY'
from pathlib import Path
import json, os, sys
root = Path('outputs/direct_scene00_scale1').resolve()
for method in ('conditional_diffusion', 'dgdm'):
    records = [json.loads(p.read_text()) for p in (root/'runs'/f'{method}_s0').rglob('benchmark_result.json')]
    print(f"{method}: {sum(r['status']=='ok' for r in records)}/80 successful; {len(records)} finished")
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        argv = (proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
        if 'benchmarks.run_baselines' not in argv or '--output_dir' not in argv:
            continue
        output = Path(argv[argv.index('--output_dir')+1])
        if not output.is_absolute():
            output = (proc/'cwd').resolve()/output
        if output.resolve() == root:
            print(f"Already running (PID {proc.name}). No duplicate started. Run this script again after it finishes to produce the report.")
            sys.exit(1)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        continue
PY
then
  exit 0
fi
.venv/bin/python -u -m benchmarks.run_baselines \
  --config benchmarks/scenarios_v5_robust_four.json \
  --benchmark_config outputs/direct_scene00_scale1/evaluation_config.json \
  --output_dir outputs/direct_scene00_scale1 \
  --methods conditional_diffusion,dgdm --seeds 0 --candidate_budget 16 \
  --target_scenario_id approach_radius:00 \
  --diffusion_checkpoint outputs/from_links_v14_fixed10_span360/diffusion_conditional_15/best.pt \
  --dgdm_dynamics_checkpoint outputs/from_links_v16_pose_dgdm/pose_dynamics_noisy_t036/best.pt \
  --diffusion_batch_size 16 --diffusion_inference_steps 5 \
  --dgdm_guidance_scale 1 --dgdm_guidance_timesteps 0,3,6 \
  --device cpu --run_benchmark --num_workers "${NUM_WORKERS:-24}" --timeout 1200 --resume
.venv/bin/python outputs/direct_scene00_scale1/summarize_comparison.py
echo "Report: $PROJECT_DIR/outputs/direct_scene00_scale1/comparison.md"
