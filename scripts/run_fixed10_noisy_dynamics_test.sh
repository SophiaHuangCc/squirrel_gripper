#!/usr/bin/env bash
# Matched 360-degree / fixed-10 training stack. Changing angular normalization
# invalidates old dynamics and conditional-diffusion checkpoints, so all three
# models are retrained before diagnostics.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v14_fixed10_span360}"

OUTPUT_ROOT="$OUTPUT_ROOT" \
MODEL_ARCHITECTURE=legacy \
NUM_HIDDEN_LAYERS=3 \
DYNAMICS_BATCH_SIZE=32 \
DYNAMICS_LR=1e-3 \
DYNAMICS_EPOCHS=300 \
DYNAMICS_PATIENCE=100 \
DYNAMICS_VAL_STEP=5 \
METRIC_LOSS_WEIGHTS=1,1,1 \
RANKING_LOSS_WEIGHT=0 \
ANGULAR_TARGET_NORMALIZATION=none \
LATE_GUIDANCE_TIMESTEPS=0,3,6 \
TRAIN_DIFFUSION=1 \
RUN_TRAINING=1 \
RUN_EVALUATION="${RUN_EVALUATION:-0}" \
exec "$PROJECT_DIR/scripts/run_dgdm_retrain_debug.sh"
