#!/usr/bin/env bash
# Ten-seed pose-DGDM method comparison using the direction-only disturbance score.
# Reuses trained checkpoints; does not retrain or render videos.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_DIR/outputs/from_links_v16_pose_dgdm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v18_pose_dgdm_10seed_directional_support}"
CONFIG="${CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v5_robust_four.json}"
CLEAN_CHECKPOINT="${CLEAN_CHECKPOINT:-$SOURCE_ROOT/pose_dynamics_clean/best.pt}"
NOISY_CHECKPOINT="${NOISY_CHECKPOINT:-$SOURCE_ROOT/pose_dynamics_noisy_t036/best.pt}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v14_fixed10_span360/diffusion_conditional_15/best.pt}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-0.1,0.5,1,2}"
GUIDANCE_TIMESTEPS="${GUIDANCE_TIMESTEPS:-0,3,6}"
NUM_WORKERS="${NUM_WORKERS:-30}"
TIMEOUT="${TIMEOUT:-1800}"
STUDY_DIR="$OUTPUT_ROOT/method_comparison"

mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "$OUTPUT_ROOT/logs/massive_10seed_$STAMP.log") 2>&1
cd "$PROJECT_DIR"

for path in "$PYTHON_BIN" "$CONFIG" "$CLEAN_CHECKPOINT" "$NOISY_CHECKPOINT" "$DIFFUSION_CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path"; exit 2; }
done

common=(
  --config "$CONFIG" --candidate_budget 8 --seeds "$SEEDS"
  --dynamics_checkpoint "$CLEAN_CHECKPOINT"
  --dgdm_dynamics_checkpoint "$NOISY_CHECKPOINT"
  --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" --device cuda
  --adam_steps 300 --adam_lr 0.03
  --cma_generations 100 --cma_popsize 32 --cma_sigma 0.5
  --diffusion_num_samples 256 --diffusion_batch_size 256 --diffusion_inference_steps 5
  --utility_weights 0.45,0.20,0.35
  --dgdm_guidance_timesteps "$GUIDANCE_TIMESTEPS"
  --target_scenario_id all --evaluation_scope auto
  --run_benchmark --resume --benchmark_top_k 8
  --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"
)

echo "[1/3] ADAM, CMA-ES, AND CONDITIONAL DIFFUSION — 10 SEEDS"
"$PYTHON_BIN" -m benchmarks.run_baselines \
  --output_dir "$STUDY_DIR/base" \
  --methods adam,cma_es,conditional_diffusion "${common[@]}"

echo "[2/3] POSE-DGDM GUIDANCE SWEEP — 10 SEEDS"
IFS=',' read -r -a scales <<< "$GUIDANCE_SCALES"
for scale in "${scales[@]}"; do
  label="pose_dgdm_gs${scale//./p}"
  echo "[DGDM] guidance_scale=$scale label=$label"
  "$PYTHON_BIN" -m benchmarks.run_baselines \
    --output_dir "$STUDY_DIR/$label" \
    --methods dgdm --dgdm_guidance_scale "$scale" --dgdm_method_label "$label" \
    "${common[@]}"
done

echo "[3/3] ORACLE SIMULATION ANALYSIS"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$STUDY_DIR" --output_dir "$OUTPUT_ROOT/study_analysis"

touch "$OUTPUT_ROOT/POSTPROCESS_DONE.status"
echo "[MASSIVE 10-SEED COMPARISON DONE] $OUTPUT_ROOT"
