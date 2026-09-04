#!/usr/bin/env bash
# Controlled DGDM-style 3-vs-8 trunk-layer comparison. All other settings match.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v12_depth_control_fixed10}"
SEED="${SEED:-0}"

run_arm() {
  local depth="$1"
  local train_diffusion="$2"
  local diffusion_checkpoint="$3"
  OUTPUT_ROOT="$OUTPUT_ROOT/dgdm${depth}" \
  MODEL_ARCHITECTURE=dgdm NUM_HIDDEN_LAYERS="$depth" SEED="$SEED" \
  RUN_TRAINING=1 RUN_EVALUATION=0 TRAIN_DIFFUSION="$train_diffusion" \
  DIFFUSION_CHECKPOINT="$diffusion_checkpoint" \
  "$PROJECT_DIR/scripts/run_dgdm_retrain_debug.sh"
}

echo "[CONTROL] Only prediction-trunk depth changes: 3 versus 8 layers"
echo "[MASK] Diffusion/noisy dynamics optimize 10 variable coordinates; 6 remain fixed"
# The prior must be retrained once because its old checkpoint denoised all 16
# coordinates. Both depth arms then use this exact same new prior.
run_arm 3 1 "$OUTPUT_ROOT/dgdm3/diffusion_conditional_15/best.pt"
run_arm 8 0 "$OUTPUT_ROOT/dgdm3/diffusion_conditional_15/best.pt"

"$PYTHON_BIN" -m benchmarks.compare_dynamics_depths \
  --shallow "$OUTPUT_ROOT/dgdm3/model_diagnostics/timestep_diagnostics.csv" \
  --deep "$OUTPUT_ROOT/dgdm8/model_diagnostics/timestep_diagnostics.csv" \
  --output "$OUTPUT_ROOT/depth_comparison.csv"

echo "[DONE] $OUTPUT_ROOT"
