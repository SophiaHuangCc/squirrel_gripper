#!/usr/bin/env bash
# Full specialists + generalist study with pose-based proposal generation and
# four-factor simulator/Oracle screening (direction + force + contact + span).
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_DIR/outputs/from_links_v16_pose_dgdm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v20_force_screened_full}"
GENERATION_CONFIG="${GENERATION_CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v5_robust_four.json}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v6_force_four.json}"
CLEAN_CHECKPOINT="${CLEAN_CHECKPOINT:-$SOURCE_ROOT/pose_dynamics_clean/best.pt}"
NOISY_CHECKPOINT="${NOISY_CHECKPOINT:-$SOURCE_ROOT/pose_dynamics_noisy_t036/best.pt}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v14_fixed10_span360/diffusion_conditional_15/best.pt}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
CANDIDATE_BUDGET="${CANDIDATE_BUDGET:-8}"
NUM_WORKERS="${NUM_WORKERS:-30}"
TIMEOUT="${TIMEOUT:-1800}"
METHODS="${METHODS:-adam,cma_es,conditional_diffusion}"
DGDM_SCALES="${DGDM_SCALES:-1}"
GUIDANCE_TIMESTEPS="${GUIDANCE_TIMESTEPS:-0,3,6}"

SPECIALIST_DIR="$OUTPUT_ROOT/specialists"
GENERALIST_DIR="$OUTPUT_ROOT/generalist"
mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "$OUTPUT_ROOT/logs/full_force_screen_$STAMP.log") 2>&1
cd "$PROJECT_DIR"

for path in "$PYTHON_BIN" "$GENERATION_CONFIG" "$BENCHMARK_CONFIG" \
            "$CLEAN_CHECKPOINT" "$NOISY_CHECKPOINT" "$DIFFUSION_CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path"; exit 2; }
done

common=(
  --config "$GENERATION_CONFIG" --benchmark_config "$BENCHMARK_CONFIG"
  --candidate_budget "$CANDIDATE_BUDGET" --benchmark_top_k "$CANDIDATE_BUDGET"
  --seeds "$SEEDS" --dynamics_checkpoint "$CLEAN_CHECKPOINT"
  --dgdm_dynamics_checkpoint "$NOISY_CHECKPOINT"
  --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" --device cuda
  --adam_steps 300 --adam_lr 0.03
  --cma_generations 100 --cma_popsize 32 --cma_sigma 0.5
  --diffusion_num_samples 256 --diffusion_batch_size 256 --diffusion_inference_steps 5
  --utility_weights 0.45,0.20,0.35
  --dgdm_guidance_timesteps "$GUIDANCE_TIMESTEPS"
  --evaluation_scope auto --run_benchmark --resume
  --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"
)

run_protocol() {
  local protocol="$1" root="$2"
  local target=()
  if [[ "$protocol" == specialist ]]; then
    target=(--target_scenario_id all)
  else
    target=(--generalist)
  fi
  echo "[$protocol BASE] $METHODS"
  "$PYTHON_BIN" -m benchmarks.run_baselines \
    --output_dir "$root/base" --methods "$METHODS" "${common[@]}" "${target[@]}"
  IFS=',' read -r -a scales <<< "$DGDM_SCALES"
  for scale in "${scales[@]}"; do
    local safe_scale="${scale//./p}"
    local label="pose_dgdm_gs${safe_scale}"
    echo "[$protocol DGDM] guidance scale $scale ($label)"
    "$PYTHON_BIN" -m benchmarks.run_baselines \
      --output_dir "$root/$label" --methods dgdm \
      --dgdm_guidance_scale "$scale" --dgdm_method_label "$label" \
      "${common[@]}" "${target[@]}"
  done
}

echo "[1/4] SPECIALISTS"
run_protocol specialist "$SPECIALIST_DIR"
echo "[2/4] GENERALIST"
run_protocol generalist "$GENERALIST_DIR"
echo "[3/4] SPECIALIST ANALYSIS + ONE WINNER PER SCENARIO"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$SPECIALIST_DIR" --output_dir "$OUTPUT_ROOT/specialist_analysis" \
  --protocol specialist --render_best_overall --num_workers 4 --timeout "$TIMEOUT"
echo "[4/4] GENERALIST ANALYSIS + ONE GENERALIST WINNER"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$GENERALIST_DIR" --output_dir "$OUTPUT_ROOT/generalist_analysis" \
  --protocol generalist --render_best_generalist --num_workers 20 --timeout "$TIMEOUT"

touch "$OUTPUT_ROOT/POSTPROCESS_DONE.status"
echo "[FULL FORCE-SCREENED STUDY DONE] $OUTPUT_ROOT"
