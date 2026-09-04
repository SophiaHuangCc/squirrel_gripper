#!/usr/bin/env bash
# Four held-out scenarios; five seeds; random, Adam, CMA-ES, conditional
# diffusion, and a separately labeled DGDM guidance-scale sweep.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
CONFIG="${CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v4_unseen_four.json}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/outputs/from_links_v6_unseen_four}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DYNAMICS_CHECKPOINT="${DYNAMICS_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v2/dynamics/best.pt}"
DGDM_DYNAMICS_CHECKPOINT="${DGDM_DYNAMICS_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v3/dynamics_noisy/best.pt}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v2/diffusion/best.pt}"

SEEDS="${SEEDS:-0,1,2,3,4}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-0.1,0.5,1,2}"
NUM_WORKERS="${NUM_WORKERS:-20}"
RENDER_WORKERS="${RENDER_WORKERS:-4}"
TIMEOUT="${TIMEOUT:-1800}"
SPECIALIST_DIR="$RESULT_ROOT/four_scenario_specialists"
GENERALIST_DIR="$RESULT_ROOT/four_scenario_generalists"
LOG_DIR="$RESULT_ROOT/study_logs"

mkdir -p "$SPECIALIST_DIR" "$GENERALIST_DIR" "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/dgdm_scale_study_$STAMP.log}"
STATUS_FILE="${STATUS_FILE:-$LOG_DIR/dgdm_scale_study_$STAMP.status}"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local code=$?
  echo "failed exit_code=$code time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
  echo "[FAILED] exit_code=$code; inspect $LOG_FILE"
  exit "$code"
}
trap on_error ERR

cd "$PROJECT_DIR"
for required in "$PYTHON_BIN" "$CONFIG" "$DYNAMICS_CHECKPOINT" \
  "$DGDM_DYNAMICS_CHECKPOINT" "$DIFFUSION_CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "ERROR: missing required path: $required"; exit 2; }
done

exec 9>"$RESULT_ROOT/.study.lock"
flock -n 9 || { echo "ERROR: another V6 study holds $RESULT_ROOT/.study.lock"; exit 1; }

echo "running pid=$$ start=$(date --iso-8601=seconds)" > "$STATUS_FILE"
echo "[START] held-out four-scenario DGDM scale study"
echo "[SCENARIOS] angles=13,77 radii=0.018,0.032 (4 combinations)"
echo "[BASE METHODS] random,adam,cma_es,conditional_diffusion"
echo "[DGDM SCALES] $GUIDANCE_SCALES"
echo "[SEEDS] $SEEDS"
echo "[UTILITY] D=0.45 C=0.20 A=0.35"
echo "[OUTPUT] $RESULT_ROOT"

common_args=(
  --config "$CONFIG"
  --candidate_budget 16
  --seeds "$SEEDS"
  --dynamics_checkpoint "$DYNAMICS_CHECKPOINT"
  --dgdm_dynamics_checkpoint "$DGDM_DYNAMICS_CHECKPOINT"
  --diffusion_checkpoint "$DIFFUSION_CHECKPOINT"
  --device cuda
  --random_pool_size 256
  --adam_steps 300
  --adam_lr 0.03
  --cma_generations 100
  --cma_popsize 32
  --cma_sigma 0.5
  --diffusion_num_samples 256
  --diffusion_batch_size 256
  --diffusion_inference_steps 20
  --utility_weights 0.45,0.20,0.35
  --evaluation_scope auto
  --run_benchmark
  --benchmark_top_k 16
  --num_workers "$NUM_WORKERS"
  --timeout "$TIMEOUT"
)

run_base() {
  local protocol=$1
  local root=$2
  local target_args=()
  [[ "$protocol" == specialist ]] && target_args=(--target_scenario_id all) || target_args=(--generalist)
  echo "[BASE START] protocol=$protocol $(date --iso-8601=seconds)"
  "$PYTHON_BIN" -m benchmarks.run_baselines \
    --output_dir "$root/combined/base" \
    --methods random,adam,cma_es,conditional_diffusion \
    "${common_args[@]}" "${target_args[@]}"
  echo "[BASE DONE] protocol=$protocol $(date --iso-8601=seconds)"
}

scale_label() {
  local scale=$1
  local safe=${scale//./p}
  safe=${safe//-/m}
  echo "dgdm_gs$safe"
}

run_dgdm_scale() {
  local protocol=$1
  local root=$2
  local scale=$3
  local label
  label=$(scale_label "$scale")
  local target_args=()
  [[ "$protocol" == specialist ]] && target_args=(--target_scenario_id all) || target_args=(--generalist)
  echo "[DGDM START] protocol=$protocol scale=$scale label=$label $(date --iso-8601=seconds)"
  "$PYTHON_BIN" -m benchmarks.run_baselines \
    --output_dir "$root/combined/$label" \
    --methods dgdm \
    --dgdm_guidance_scale "$scale" \
    --dgdm_method_label "$label" \
    "${common_args[@]}" "${target_args[@]}"
  echo "[DGDM DONE] protocol=$protocol scale=$scale $(date --iso-8601=seconds)"
}

run_base specialist "$SPECIALIST_DIR"
run_base generalist "$GENERALIST_DIR"

IFS=',' read -r -a scale_values <<< "$GUIDANCE_SCALES"
for scale in "${scale_values[@]}"; do
  run_dgdm_scale specialist "$SPECIALIST_DIR" "$scale"
  run_dgdm_scale generalist "$GENERALIST_DIR" "$scale"
done

# Each DGDM scale remains a distinct method in every analysis table. Rendering
# per method preserves comparisons instead of silently cherry-picking a scale.
echo "[SPECIALIST ANALYSIS/RENDER START] $(date --iso-8601=seconds)"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$SPECIALIST_DIR" --protocol specialist \
  --render_best_per_method --num_workers "$RENDER_WORKERS" --timeout "$TIMEOUT"
echo "[SPECIALIST ANALYSIS/RENDER DONE] $(date --iso-8601=seconds)"

echo "[GENERALIST ANALYSIS/RENDER START] $(date --iso-8601=seconds)"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$GENERALIST_DIR" --protocol generalist \
  --render_best_generalist_per_method --num_workers "$RENDER_WORKERS" --timeout "$TIMEOUT"
echo "[GENERALIST ANALYSIS/RENDER DONE] $(date --iso-8601=seconds)"

echo "complete time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
echo "[DONE] four-scenario DGDM guidance-scale study $(date --iso-8601=seconds)"
