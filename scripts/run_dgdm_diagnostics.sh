#!/usr/bin/env bash
# Cheap diagnosis: held-out model audit plus a paired, one-seed simulator sweep.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/TendonForces/runs/exp1}"
CONFIG="${CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v4_unseen_four.json}"
CLEAN_CHECKPOINT="${CLEAN_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v2/dynamics/best.pt}"
NOISY_CHECKPOINT="${NOISY_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v3/dynamics_noisy/best.pt}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v2/diffusion/best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v6_dgdm_diagnostics}"
SEED="${SEED:-0}"
SCALES="${SCALES:-0,0.1,0.5,1,2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-2048}"
TIMEOUT="${TIMEOUT:-1800}"

mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_ROOT/logs/diagnostics_$STAMP.log"
STATUS_FILE="$OUTPUT_ROOT/logs/diagnostics_$STAMP.status"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'code=$?; echo "failed exit_code=$code" > "$STATUS_FILE"; exit "$code"' ERR

cd "$PROJECT_DIR"
for path in "$PYTHON_BIN" "$DATASET_DIR/test" "$CONFIG" "$CLEAN_CHECKPOINT" \
  "$NOISY_CHECKPOINT" "$DIFFUSION_CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path"; exit 2; }
done

echo "running seed=$SEED scales=$SCALES" > "$STATUS_FILE"

echo "[1/3] TIMESTEP AND GRADIENT DIAGNOSTICS"
"$PYTHON_BIN" -m benchmarks.diagnose_dgdm \
  --data_dir "$DATASET_DIR/test" \
  --clean_checkpoint "$CLEAN_CHECKPOINT" \
  --noisy_checkpoint "$NOISY_CHECKPOINT" \
  --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_ROOT/model_diagnostics" \
  --timesteps 0,10,25,50,75,90,99 \
  --max_samples "$MAX_SAMPLES" \
  --seed "$SEED" --device cuda

echo "[2/3] PAIRED ONE-SEED SIMULATOR SWEEP"
for scenario in approach_radius:00 approach_radius:01 approach_radius:02 approach_radius:03; do
  safe="${scenario//:/-}"
  "$PYTHON_BIN" -m benchmarks.run_guidance_sweep \
    --output_dir "$OUTPUT_ROOT/paired_simulator/$safe" \
    --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
    --dynamics_checkpoint "$CLEAN_CHECKPOINT" \
    --dgdm_dynamics_checkpoint "$NOISY_CHECKPOINT" \
    --config "$CONFIG" --scales "$SCALES" --seeds "$SEED" \
    --candidate_budget 16 --num_samples 256 --batch_size 256 \
    --inference_steps 20 --target_scenario_id "$scenario" \
    --device cuda --run_benchmark --benchmark_top_k 1 \
    --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"
done

echo "[3/3] COMBINED SIMULATOR ANALYSIS"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$OUTPUT_ROOT/paired_simulator" \
  --output_dir "$OUTPUT_ROOT/paired_simulator_analysis" \
  --protocol specialist

echo "complete time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
echo "[DONE] $OUTPUT_ROOT"
