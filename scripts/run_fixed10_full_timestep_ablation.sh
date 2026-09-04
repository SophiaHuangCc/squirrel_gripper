#!/usr/bin/env bash
# Controlled ablation: V14 noisy dynamics trained at 0,3,6 versus a model
# trained across every timestep used by the 15/5 DDIM sampler: 0,3,6,9,12.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/TendonForces/runs/exp1}"
CONFIG="${CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v5_robust_four.json}"
V14_ROOT="${V14_ROOT:-$PROJECT_DIR/outputs/from_links_v14_fixed10_span360}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v15_fixed10_full_timesteps}"
WANDB_PROJECT="${WANDB_PROJECT:-squirrel-gripper-dgdm-debugging}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED="${SEED:-0}"

CLEAN_CHECKPOINT="${CLEAN_CHECKPOINT:-$V14_ROOT/dynamics_clean_legacy3_long_equal_zscore/best.pt}"
CURRENT_NOISY_CHECKPOINT="${CURRENT_NOISY_CHECKPOINT:-$V14_ROOT/dynamics_noisy_legacy3_long_equal_zscore_15/best.pt}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$V14_ROOT/diffusion_conditional_15/best.pt}"
FULL_NOISY_DIR="$OUTPUT_ROOT/dynamics_noisy_legacy3_fixed10_t0_3_6_9_12"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/full_timestep_ablation_$STAMP.log"
STATUS_FILE="$LOG_DIR/full_timestep_ablation_$STAMP.status"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'code=$?; echo "failed exit_code=$code" > "$STATUS_FILE"; exit "$code"' ERR

cd "$PROJECT_DIR"
for path in "$PYTHON_BIN" "$DATASET_DIR/train" "$DATASET_DIR/test" "$CONFIG" \
  "$CLEAN_CHECKPOINT" "$CURRENT_NOISY_CHECKPOINT" "$DIFFUSION_CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path"; exit 2; }
done

echo "running time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
echo "[1/4] TRAIN FIXED-10 NOISY DYNAMICS AT t=0,3,6,9,12"
"$PYTHON_BIN" dynamics/main.py --mode train --device cuda \
  --data_dir "$DATASET_DIR/train" --test_data_dir "$DATASET_DIR/test" \
  --save_dir "$FULL_NOISY_DIR" --batch_size 32 --num_workers 8 --lr 1e-3 --seed "$SEED" \
  --num_epochs 300 --patience 100 --val_step 5 --save_ckpt_step 500 \
  --output_dim 3 --use_design_noise --model_architecture legacy --num_hidden_layers 3 \
  --num_train_timesteps 15 --num_inference_steps 5 \
  --num_timesteps_per_batch 5 --noise_timesteps 0,3,6,9,12 \
  --metric_loss_weights 1,1,1 --angular_target_normalization none \
  --utility_weights 0.20,0.45,0.35 --ranking_loss_weight 0 \
  --ranking_margin 0.05 --ranking_min_target_delta 0.05 \
  --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
  --wandb_run_name "noisy-legacy3-fixed10-t0-3-6-9-12-seed${SEED}-$STAMP"

echo "[2/4] DIAGNOSE CURRENT t=0,3,6-TRAINED MODEL"
"$PYTHON_BIN" -m benchmarks.diagnose_dgdm \
  --data_dir "$DATASET_DIR/test" --clean_checkpoint "$CLEAN_CHECKPOINT" \
  --noisy_checkpoint "$CURRENT_NOISY_CHECKPOINT" --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
  --config "$CONFIG" --output_dir "$OUTPUT_ROOT/diagnostics_t036" \
  --timesteps 0,3,6,9,12 --max_samples 2048 --seed "$SEED" --device cuda

echo "[3/4] DIAGNOSE FULL-TIMESTEP MODEL"
"$PYTHON_BIN" -m benchmarks.diagnose_dgdm \
  --data_dir "$DATASET_DIR/test" --clean_checkpoint "$CLEAN_CHECKPOINT" \
  --noisy_checkpoint "$FULL_NOISY_DIR/best.pt" --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
  --config "$CONFIG" --output_dir "$OUTPUT_ROOT/diagnostics_t036912" \
  --timesteps 0,3,6,9,12 --max_samples 2048 --seed "$SEED" --device cuda

echo "[4/4] WRITE CONTROLLED COMPARISON"
"$PYTHON_BIN" -m benchmarks.compare_dgdm_diagnostics \
  --baseline "$OUTPUT_ROOT/diagnostics_t036/timestep_diagnostics.csv" \
  --candidate "$OUTPUT_ROOT/diagnostics_t036912/timestep_diagnostics.csv" \
  --candidate_label full_timestep \
  --output "$OUTPUT_ROOT/full_timestep_comparison.csv"

echo "complete time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
echo "[DONE] $OUTPUT_ROOT"
