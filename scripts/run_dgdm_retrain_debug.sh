#!/usr/bin/env bash
# Retrain a controlled 15/5 conditional-DGDM stack and run a one-seed comparison.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/TendonForces/runs/exp1}"
CONFIG="${CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v5_robust_four.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v10_fast_robust}"
WANDB_PROJECT="${WANDB_PROJECT:-squirrel-gripper-dgdm-debugging}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
TRAIN_DIFFUSION="${TRAIN_DIFFUSION:-0}"
NUM_WORKERS="${NUM_WORKERS:-20}"
TIMEOUT="${TIMEOUT:-1800}"
DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-128}"
DYNAMICS_LR="${DYNAMICS_LR:-1e-4}"
DYNAMICS_EPOCHS="${DYNAMICS_EPOCHS:-300}"
DYNAMICS_PATIENCE="${DYNAMICS_PATIENCE:-100}"
DYNAMICS_VAL_STEP="${DYNAMICS_VAL_STEP:-5}"

TRAIN_TIMESTEPS="${TRAIN_TIMESTEPS:-15}"
INFERENCE_STEPS="${INFERENCE_STEPS:-5}"
METRIC_LOSS_WEIGHTS="${METRIC_LOSS_WEIGHTS:-1,1,1}"
RANKING_LOSS_WEIGHT="${RANKING_LOSS_WEIGHT:-0}"
ANGULAR_TARGET_NORMALIZATION="${ANGULAR_TARGET_NORMALIZATION:-zscore}"
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-dgdm}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-8}"
UTILITY_WEIGHTS_CDA="${UTILITY_WEIGHTS_CDA:-0.20,0.45,0.35}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-0.1,0.5,1,2}"
LATE_GUIDANCE_TIMESTEPS="${LATE_GUIDANCE_TIMESTEPS:-0,3,6}"
SEED="${SEED:-0}"

CLEAN_DIR="$OUTPUT_ROOT/dynamics_clean_dgdm8_long_equal_zscore"
NOISY_DIR="$OUTPUT_ROOT/dynamics_noisy_dgdm8_long_equal_zscore_15"
DIFFUSION_DIR="$OUTPUT_ROOT/diffusion_conditional_15"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v7_dgdm_retrain/diffusion_conditional_15/best.pt}"
STUDY_DIR="$OUTPUT_ROOT/one_seed_comparison"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/retrain_debug_$STAMP.log"
STATUS_FILE="$LOG_DIR/retrain_debug_$STAMP.status"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'code=$?; echo "failed exit_code=$code" > "$STATUS_FILE"; exit "$code"' ERR

cd "$PROJECT_DIR"
for path in "$PYTHON_BIN" "$DATASET_DIR/train" "$DATASET_DIR/test" "$CONFIG"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path"; exit 2; }
done

echo "running time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
echo "[CONFIG] T=$TRAIN_TIMESTEPS inference=$INFERENCE_STEPS metric_loss=$METRIC_LOSS_WEIGHTS ranking=$RANKING_LOSS_WEIGHT"
echo "[ARCHITECTURE] model=$MODEL_ARCHITECTURE hidden_layers=$NUM_HIDDEN_LAYERS width=256"
echo "[DYNAMICS TRAINING] batch=$DYNAMICS_BATCH_SIZE lr=$DYNAMICS_LR epochs=$DYNAMICS_EPOCHS patience_epochs=$DYNAMICS_PATIENCE val_step=$DYNAMICS_VAL_STEP"

if [[ "$RUN_TRAINING" == 1 ]]; then
  echo "[1/5] TRAIN CLEAN DYNAMICS"
  "$PYTHON_BIN" dynamics/main.py --mode train --device cuda \
    --data_dir "$DATASET_DIR/train" --test_data_dir "$DATASET_DIR/test" \
    --save_dir "$CLEAN_DIR" --batch_size "$DYNAMICS_BATCH_SIZE" \
    --num_workers 8 --lr "$DYNAMICS_LR" \
    --num_epochs "$DYNAMICS_EPOCHS" --patience "$DYNAMICS_PATIENCE" \
    --val_step "$DYNAMICS_VAL_STEP" --save_ckpt_step 500 \
    --output_dim 3 --model_architecture "$MODEL_ARCHITECTURE" \
    --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
    --metric_loss_weights "$METRIC_LOSS_WEIGHTS" \
    --angular_target_normalization "$ANGULAR_TARGET_NORMALIZATION" \
    --utility_weights "$UTILITY_WEIGHTS_CDA" \
    --ranking_loss_weight "$RANKING_LOSS_WEIGHT" \
    --ranking_margin 0.05 --ranking_min_target_delta 0.05 \
    --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "clean-dgdm8-long-equal-zscore-$STAMP"

  if [[ "$TRAIN_DIFFUSION" == 1 ]]; then
    echo "[2/5] TRAIN CONDITIONAL DIFFUSION AT 15/5"
    "$PYTHON_BIN" generator/train.py --device cuda \
      --data_dir "$DATASET_DIR/train" --save_dir "$DIFFUSION_DIR" \
      --conditioning conditional --batch_size 512 --num_workers 8 \
      --num_epochs 500 --learning_rate 1e-4 --patience 30 --min_delta 1e-5 \
      --num_train_timesteps "$TRAIN_TIMESTEPS" --num_inference_steps "$INFERENCE_STEPS" \
      --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
      --wandb_run_name "conditional-diffusion-15x5-$STAMP"
    DIFFUSION_CHECKPOINT="$DIFFUSION_DIR/best.pt"
  else
    echo "[2/5] REUSE UNCHANGED CONDITIONAL DIFFUSION: $DIFFUSION_CHECKPOINT"
  fi

  echo "[3/5] TRAIN MATCHED NOISY DYNAMICS"
  "$PYTHON_BIN" dynamics/main.py --mode train --device cuda \
    --data_dir "$DATASET_DIR/train" --test_data_dir "$DATASET_DIR/test" \
    --save_dir "$NOISY_DIR" --batch_size "$DYNAMICS_BATCH_SIZE" \
    --num_workers 8 --lr "$DYNAMICS_LR" \
    --num_epochs "$DYNAMICS_EPOCHS" --patience "$DYNAMICS_PATIENCE" \
    --val_step "$DYNAMICS_VAL_STEP" --save_ckpt_step 500 \
    --output_dim 3 --use_design_noise \
    --model_architecture "$MODEL_ARCHITECTURE" \
    --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
    --num_train_timesteps "$TRAIN_TIMESTEPS" --num_inference_steps "$INFERENCE_STEPS" \
    --num_timesteps_per_batch 3 --noise_timesteps "$LATE_GUIDANCE_TIMESTEPS" \
    --metric_loss_weights "$METRIC_LOSS_WEIGHTS" \
    --angular_target_normalization "$ANGULAR_TARGET_NORMALIZATION" \
    --utility_weights "$UTILITY_WEIGHTS_CDA" \
    --ranking_loss_weight "$RANKING_LOSS_WEIGHT" \
    --ranking_margin 0.05 --ranking_min_target_delta 0.05 \
    --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "noisy-dgdm8-long-equal-zscore-15x5-$STAMP"
fi

for checkpoint in "$CLEAN_DIR/best.pt" "$NOISY_DIR/best.pt" "$DIFFUSION_CHECKPOINT"; do
  [[ -f "$checkpoint" ]] || { echo "ERROR: expected checkpoint missing: $checkpoint"; exit 2; }
done

echo "[4/5] TIMESTEP/GRADIENT DIAGNOSTICS"
"$PYTHON_BIN" -m benchmarks.diagnose_dgdm \
  --data_dir "$DATASET_DIR/test" --clean_checkpoint "$CLEAN_DIR/best.pt" \
  --noisy_checkpoint "$NOISY_DIR/best.pt" --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
  --config "$CONFIG" --output_dir "$OUTPUT_ROOT/model_diagnostics" \
  --timesteps "$LATE_GUIDANCE_TIMESTEPS" --max_samples 2048 --seed "$SEED" --device cuda \
  --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
  --wandb_run_name "diagnostic-dgdm8-long-late-guidance-15x5-$STAMP"

if [[ "$RUN_EVALUATION" == 1 ]]; then
  echo "[5/5] ONE-SEED ADAM/CMA/DIFFUSION/DGDM COMPARISON"
  common=(--config "$CONFIG" --candidate_budget 8 --seeds "$SEED"
    --dynamics_checkpoint "$CLEAN_DIR/best.pt"
    --dgdm_dynamics_checkpoint "$NOISY_DIR/best.pt"
    --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" --device cuda
    --adam_steps 300 --adam_lr 0.03 --cma_generations 100 --cma_popsize 32 --cma_sigma 0.5
    --diffusion_num_samples 256 --diffusion_batch_size 256
    --diffusion_inference_steps "$INFERENCE_STEPS" --utility_weights 0.45,0.20,0.35
    --dgdm_guidance_timesteps "$LATE_GUIDANCE_TIMESTEPS"
    --target_scenario_id all --evaluation_scope auto --run_benchmark
    --benchmark_top_k 8 --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT")
  "$PYTHON_BIN" -m benchmarks.run_baselines --output_dir "$STUDY_DIR/base" \
    --methods adam,cma_es,conditional_diffusion "${common[@]}"
  IFS=',' read -r -a scales <<< "$GUIDANCE_SCALES"
  for scale in "${scales[@]}"; do
    label="dgdm_gs${scale//./p}"
    "$PYTHON_BIN" -m benchmarks.run_baselines --output_dir "$STUDY_DIR/$label" \
      --methods dgdm --dgdm_guidance_scale "$scale" --dgdm_method_label "$label" \
      "${common[@]}"
  done
  "$PYTHON_BIN" -m benchmarks.analyze_study --study_dir "$STUDY_DIR" \
    --output_dir "$OUTPUT_ROOT/one_seed_analysis" --protocol specialist
  "$PYTHON_BIN" -m benchmarks.diagnose_robust_gradients \
    --benchmark_dir "$STUDY_DIR" --config "$CONFIG" \
    --clean_checkpoint "$CLEAN_DIR/best.pt" --noisy_checkpoint "$NOISY_DIR/best.pt" \
    --output_dir "$OUTPUT_ROOT/robust_diagnostics" \
    --timesteps "$LATE_GUIDANCE_TIMESTEPS" --device cuda --seed "$SEED"
fi

echo "complete time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
echo "[DONE] $OUTPUT_ROOT"
