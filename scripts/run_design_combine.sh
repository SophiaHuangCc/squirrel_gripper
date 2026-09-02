#!/usr/bin/env bash
# Combined-utility-only overnight study: nine specialists, one generalist
# protocol, benchmark analysis, and final overall-winner MP4 rendering.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/TendonForces/runs/exp1}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v2_compact.json}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/outputs/from_links_v4_old_utility}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

# The benchmark search code requires the clean three-output surrogate trained
# for the current 16-dimensional design representation.  The older
# outputs/dynamics/exp1 checkpoint is architecture-incompatible.
if [[ -z "${DYNAMICS_CHECKPOINT:-}" ]]; then
  if [[ -f "$PROJECT_DIR/outputs/from_links_v2/dynamics/best.pt" ]]; then
    DYNAMICS_CHECKPOINT="$PROJECT_DIR/outputs/from_links_v2/dynamics/best.pt"
  else
    echo "ERROR: compatible clean dynamics checkpoint is missing: $PROJECT_DIR/outputs/from_links_v2/dynamics/best.pt"
    exit 2
  fi
fi
DGDM_DYNAMICS_CHECKPOINT="${DGDM_DYNAMICS_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v3/dynamics_noisy/best.pt}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v2/diffusion/best.pt}"
UNCONDITIONAL_DIFFUSION_CHECKPOINT="${UNCONDITIONAL_DIFFUSION_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v2/unconditional_diffusion/best.pt}"

SPECIALIST_DIR="$RESULT_ROOT/nine_scenario_objectives"
GENERALIST_DIR="$RESULT_ROOT/nine_scenario_generalists"
LOG_DIR="$RESULT_ROOT/study_logs"
NUM_WORKERS="${NUM_WORKERS:-30}"
RENDER_WORKERS="${RENDER_WORKERS:-4}"
TIMEOUT="${TIMEOUT:-1800}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
METHODS="${METHODS:-reference,random,random_search,retrieval,adam,cma_es,unconditional_diffusion,unconditional_dgdm,conditional_diffusion,dgdm}"
DGDM_GUIDANCE_SCALE="${DGDM_GUIDANCE_SCALE:-0.1}"

mkdir -p "$LOG_DIR" "$SPECIALIST_DIR" "$GENERALIST_DIR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/combined_designs_$RUN_STAMP.log}"
STATUS_FILE="${STATUS_FILE:-$LOG_DIR/combined_designs_$RUN_STAMP.status}"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local code=$?
  echo "failed exit_code=$code time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
  echo "[FAILED] exit_code=$code; inspect $LOG_FILE"
  exit "$code"
}
trap on_error ERR

cd "$PROJECT_DIR"

for required in \
  "$PYTHON_BIN" \
  "$DATASET_DIR/train" \
  "$BENCHMARK_CONFIG" \
  "$DYNAMICS_CHECKPOINT" \
  "$DGDM_DYNAMICS_CHECKPOINT" \
  "$DIFFUSION_CHECKPOINT" \
  "$UNCONDITIONAL_DIFFUSION_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required path does not exist: $required"
    exit 2
  fi
done

exec 9>"$RESULT_ROOT/.combined_designs.lock"
if ! flock -n 9; then
  echo "ERROR: another combined-design run holds $RESULT_ROOT/.combined_designs.lock"
  exit 1
fi

echo "running pid=$$ start=$(date --iso-8601=seconds) log=$LOG_FILE" > "$STATUS_FILE"
echo "[START] combined-only old utility time=$(date --iso-8601=seconds)"
echo "[UTILITY] D=0.45 C=0.20 A=0.35"
echo "[CONFIG] seeds=$SEEDS benchmark_workers=$NUM_WORKERS render_workers=$RENDER_WORKERS"
echo "[CONFIG] dgdm_guidance_scale=$DGDM_GUIDANCE_SCALE"
echo "[OUTPUT] $RESULT_ROOT"

echo "[CHECKPOINT TEST] clean dynamics=$DYNAMICS_CHECKPOINT"
"$PYTHON_BIN" -c \
  'from benchmarks.baselines.surrogate_search import load_surrogate; import sys; load_surrogate(sys.argv[1], device="cpu"); print("[CHECKPOINT OK] clean dynamics")' \
  "$DYNAMICS_CHECKPOINT"
echo "[CHECKPOINT TEST] noisy dynamics=$DGDM_DYNAMICS_CHECKPOINT"
"$PYTHON_BIN" -c \
  'from benchmarks.baselines.surrogate_search import load_surrogate; import sys; load_surrogate(sys.argv[1], device="cpu", expected_noise_conditioned=True); print("[CHECKPOINT OK] noisy dynamics")' \
  "$DGDM_DYNAMICS_CHECKPOINT"

run_candidates_and_benchmarks() {
  local protocol=$1
  local output_dir=$2
  local target_args=()
  if [[ "$protocol" == "specialist" ]]; then
    target_args=(--target_scenario_id all)
  else
    target_args=(--generalist)
  fi

  echo "[BENCHMARK START] protocol=$protocol time=$(date --iso-8601=seconds)"
  "$PYTHON_BIN" -m benchmarks.run_baselines \
    --output_dir "$output_dir/combined" \
    --config "$BENCHMARK_CONFIG" \
    --methods "$METHODS" \
    --candidate_budget 16 \
    --seeds "$SEEDS" \
    --retrieval_data_dir "$DATASET_DIR/train" \
    --dynamics_checkpoint "$DYNAMICS_CHECKPOINT" \
    --dgdm_dynamics_checkpoint "$DGDM_DYNAMICS_CHECKPOINT" \
    --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
    --unconditional_diffusion_checkpoint "$UNCONDITIONAL_DIFFUSION_CHECKPOINT" \
    --device cuda \
    --random_pool_size 256 \
    --adam_steps 300 \
    --adam_lr 0.03 \
    --cma_generations 100 \
    --cma_popsize 32 \
    --cma_sigma 0.5 \
    --diffusion_num_samples 256 \
    --diffusion_batch_size 256 \
    --diffusion_inference_steps 20 \
    --dgdm_guidance_scale "$DGDM_GUIDANCE_SCALE" \
    --utility_weights 0.45,0.20,0.35 \
    "${target_args[@]}" \
    --evaluation_scope auto \
    --run_benchmark \
    --benchmark_top_k 1 \
    --num_workers "$NUM_WORKERS" \
    --timeout "$TIMEOUT"
  echo "[BENCHMARK DONE] protocol=$protocol time=$(date --iso-8601=seconds)"
}

run_candidates_and_benchmarks specialist "$SPECIALIST_DIR"
run_candidates_and_benchmarks generalist "$GENERALIST_DIR"

echo "[ANALYSIS START] protocol=specialist time=$(date --iso-8601=seconds)"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$SPECIALIST_DIR" --protocol specialist \
  --render_best_overall --num_workers "$RENDER_WORKERS" --timeout "$TIMEOUT"
echo "[ANALYSIS DONE] protocol=specialist time=$(date --iso-8601=seconds)"

echo "[ANALYSIS START] protocol=generalist time=$(date --iso-8601=seconds)"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$GENERALIST_DIR" --protocol generalist \
  --render_best_generalist --num_workers "$RENDER_WORKERS" --timeout "$TIMEOUT"
echo "[ANALYSIS DONE] protocol=generalist time=$(date --iso-8601=seconds)"

echo "complete time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
echo "[DONE] combined specialists, generalist, analysis, and final videos time=$(date --iso-8601=seconds)"
