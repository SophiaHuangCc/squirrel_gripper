#!/usr/bin/env bash
# Stage 1: test physical-condition aggregation with existing V7 candidates/models.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
V7_ROOT="${V7_ROOT:-$PROJECT_DIR/outputs/from_links_v7_dgdm_retrain}"
SOURCE="${SOURCE:-$V7_ROOT/one_seed_comparison}"
CONFIG="${CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v5_robust_four.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v9_robust_gradient_test}"
ROBUST_RUNS="$OUTPUT_ROOT/robust_rebenchmark"
DIAGNOSTICS="$OUTPUT_ROOT/robust_diagnostics"
CLEAN_CHECKPOINT="${CLEAN_CHECKPOINT:-$V7_ROOT/dynamics_clean_weighted_rank/best.pt}"
NOISY_CHECKPOINT="${NOISY_CHECKPOINT:-$V7_ROOT/dynamics_noisy_weighted_rank_15/best.pt}"
METHODS="${METHODS:-adam,conditional_diffusion,dgdm_gs0p1}"
TOP_K="${TOP_K:-8}"
NUM_WORKERS="${NUM_WORKERS:-20}"
TIMEOUT="${TIMEOUT:-1800}"

mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_ROOT/logs/robust_gradient_test_$STAMP.log"
STATUS_FILE="$OUTPUT_ROOT/logs/robust_gradient_test_$STAMP.status"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'code=$?; echo "failed exit_code=$code" > "$STATUS_FILE"; exit "$code"' ERR
echo "running time=$(date --iso-8601=seconds)" > "$STATUS_FILE"

cd "$PROJECT_DIR"
echo "[1/2] SIMULATE EACH EXISTING DESIGN UNDER FIVE PHYSICAL CONDITIONS"
"$PYTHON_BIN" -m benchmarks.rebenchmark_candidate_pools \
  --study_dir "$SOURCE" --output_dir "$ROBUST_RUNS" --config "$CONFIG" \
  --methods "$METHODS" --top_k "$TOP_K" \
  --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"

echo "[2/2] COMPARE AGGREGATED MODEL GRADIENTS WITH ROBUST SIMULATOR UTILITY"
"$PYTHON_BIN" -m benchmarks.diagnose_robust_gradients \
  --benchmark_dir "$ROBUST_RUNS" --config "$CONFIG" \
  --clean_checkpoint "$CLEAN_CHECKPOINT" --noisy_checkpoint "$NOISY_CHECKPOINT" \
  --output_dir "$DIAGNOSTICS" --timesteps 0,3,6,9,12 --device cuda --seed 0

echo "complete time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
echo "[DONE] $OUTPUT_ROOT"
echo "[READ] $DIAGNOSTICS/robust_gradient_diagnostics.csv"
echo "[READ] $DIAGNOSTICS/robust_direction_pairs.csv"
echo "[READ] $DIAGNOSTICS/best_robust_design_per_method.csv"
