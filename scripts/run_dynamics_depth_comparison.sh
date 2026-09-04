#!/usr/bin/env bash
# Controlled DGDM-style 3-vs-8 trunk-layer comparison. All other settings match.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v11_depth_control}"
SEED="${SEED:-0}"

run_arm() {
  local depth="$1"
  OUTPUT_ROOT="$OUTPUT_ROOT/dgdm${depth}" \
  MODEL_ARCHITECTURE=dgdm NUM_HIDDEN_LAYERS="$depth" SEED="$SEED" \
  RUN_TRAINING=1 RUN_EVALUATION=0 TRAIN_DIFFUSION=0 \
  "$PROJECT_DIR/scripts/run_dgdm_retrain_debug.sh"
}

echo "[CONTROL] Only prediction-trunk depth changes: 3 versus 8 layers"
run_arm 3
run_arm 8

"$PYTHON_BIN" -m benchmarks.compare_dynamics_depths \
  --shallow "$OUTPUT_ROOT/dgdm3/model_diagnostics/timestep_diagnostics.csv" \
  --deep "$OUTPUT_ROOT/dgdm8/model_diagnostics/timestep_diagnostics.csv" \
  --output "$OUTPUT_ROOT/depth_comparison.csv"

echo "[DONE] $OUTPUT_ROOT"
