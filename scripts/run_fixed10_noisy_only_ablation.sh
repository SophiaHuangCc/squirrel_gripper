#!/usr/bin/env bash
# Hold clean dynamics, architecture, batch size, data, and prior fixed. Retrain
# only noisy dynamics with the six zero-range design coordinates masked.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/TendonForces/runs/exp1}"
CONFIG="${CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v4_unseen_four.json}"
BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_DIR/outputs/from_links_v7_dgdm_retrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v13_fixed10_defaultloss_bs32_lr1e3}"
CLEAN_CHECKPOINT="${CLEAN_CHECKPOINT:-$BASELINE_ROOT/dynamics_clean_weighted_rank/best.pt}"
BASELINE_NOISY_CHECKPOINT="${BASELINE_NOISY_CHECKPOINT:-$BASELINE_ROOT/dynamics_noisy_weighted_rank_15/best.pt}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$BASELINE_ROOT/diffusion_conditional_15/best.pt}"
NOISY_DIR="$OUTPUT_ROOT/dynamics_noisy_fixed10_equal_norank_15"
WANDB_PROJECT="${WANDB_PROJECT:-squirrel-gripper-dgdm-debugging}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED="${SEED:-0}"
mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "$OUTPUT_ROOT/logs/noisy_only_$STAMP.log") 2>&1

for path in "$PYTHON_BIN" "$DATASET_DIR/train" "$DATASET_DIR/test" "$CONFIG" \
            "$CLEAN_CHECKPOINT" "$BASELINE_NOISY_CHECKPOINT" "$DIFFUSION_CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path"; exit 2; }
done
cd "$PROJECT_DIR"

echo "[SETUP] legacy3, batch=32, lr=1e-3, patience=100, equal loss, no ranking"
echo "[NOISE] timesteps=0,3,6; fixed indices 7,8,9,10,14,15 are never corrupted"
echo "[REFERENCE] V7 checkpoints are retained only as the diagnostic baseline"
"$PYTHON_BIN" dynamics/main.py --mode train --device cuda --seed "$SEED" \
  --data_dir "$DATASET_DIR/train" --test_data_dir "$DATASET_DIR/test" \
  --save_dir "$NOISY_DIR" --batch_size 32 --num_workers 8 --lr 1e-3 \
  --num_epochs 300 --patience 100 --val_step 5 --save_ckpt_step 500 \
  --output_dim 3 --use_design_noise --model_architecture legacy --num_hidden_layers 3 \
  --num_train_timesteps 15 --num_inference_steps 5 \
  --num_timesteps_per_batch 3 --noise_timesteps 0,3,6 \
  --metric_loss_weights 1,1,1 --angular_target_normalization none \
  --utility_weights 0.20,0.45,0.35 --ranking_loss_weight 0 \
  --ranking_margin 0.05 --ranking_min_target_delta 0.05 \
  --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
  --wandb_run_name "noisy-fixed10-equal-norank-bs32-lr1e3-$STAMP"

diagnose() {
  local noisy_checkpoint="$1"
  local output_dir="$2"
  local label="$3"
  "$PYTHON_BIN" -m benchmarks.diagnose_dgdm \
    --data_dir "$DATASET_DIR/test" --clean_checkpoint "$CLEAN_CHECKPOINT" \
    --noisy_checkpoint "$noisy_checkpoint" --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
    --config "$CONFIG" --output_dir "$output_dir" --timesteps 0,3,6 \
    --max_samples 2048 --seed "$SEED" --device cuda \
    --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "$label-$STAMP"
}

# Re-evaluate both checkpoints with the same corrected, masked diagnostic inputs.
diagnose "$BASELINE_NOISY_CHECKPOINT" "$OUTPUT_ROOT/baseline_recomputed" "baseline-unmasked-training"
diagnose "$NOISY_DIR/best.pt" "$OUTPUT_ROOT/fixed10" "fixed10-noisy-training"

"$PYTHON_BIN" -m benchmarks.compare_dgdm_diagnostics \
  --baseline "$OUTPUT_ROOT/baseline_recomputed/timestep_diagnostics.csv" \
  --candidate "$OUTPUT_ROOT/fixed10/timestep_diagnostics.csv" \
  --output "$OUTPUT_ROOT/noisy_mask_comparison.csv"

echo "[DONE] $OUTPUT_ROOT"
