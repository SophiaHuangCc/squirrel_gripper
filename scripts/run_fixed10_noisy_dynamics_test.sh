#!/usr/bin/env bash
# Focused test: original shallow dynamics setup plus a 10D variable-design mask.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v12_fixed10_legacy3_bs32}"

echo "[FIXED10 TEST] legacy shallow=3 batch=32; six constant design coordinates masked"
echo "[OUTPUT] $OUTPUT_ROOT"

OUTPUT_ROOT="$OUTPUT_ROOT" \
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-legacy}" \
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-3}" \
DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-32}" \
DYNAMICS_LR="${DYNAMICS_LR:-1e-4}" \
DYNAMICS_EPOCHS="${DYNAMICS_EPOCHS:-300}" \
DYNAMICS_PATIENCE="${DYNAMICS_PATIENCE:-100}" \
DYNAMICS_VAL_STEP="${DYNAMICS_VAL_STEP:-5}" \
TRAIN_DIFFUSION="${TRAIN_DIFFUSION:-1}" \
RUN_TRAINING="${RUN_TRAINING:-1}" \
RUN_EVALUATION="${RUN_EVALUATION:-0}" \
"$PROJECT_DIR/scripts/run_dgdm_retrain_debug.sh"
