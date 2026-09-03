#!/usr/bin/env bash
# Retrain only the noisy guidance model with matched ranking pairs, then compare methods.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/TendonForces/runs/exp1}"
CONFIG="${CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v4_unseen_four.json}"
V7_ROOT="${V7_ROOT:-$PROJECT_DIR/outputs/from_links_v7_dgdm_retrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v8_directional}"
CLEAN_CHECKPOINT="${CLEAN_CHECKPOINT:-$V7_ROOT/dynamics_clean_weighted_rank/best.pt}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$V7_ROOT/diffusion_conditional_15/best.pt}"
NOISY_DIR="$OUTPUT_ROOT/dynamics_noisy_context_rank_15x10"
STUDY_DIR="$OUTPUT_ROOT/comparison"
ANALYSIS_DIR="$OUTPUT_ROOT/analysis"
LOG_DIR="$OUTPUT_ROOT/logs"

TRAIN_TIMESTEPS="${TRAIN_TIMESTEPS:-15}"
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"
SEEDS="${SEEDS:-0}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-0.5,1,2,4}"
NUM_WORKERS="${NUM_WORKERS:-20}"
TIMEOUT="${TIMEOUT:-1800}"
WANDB_PROJECT="${WANDB_PROJECT:-squirrel-gripper-dgdm-debugging}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_TRAINING="${RUN_TRAINING:-1}"

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/directional_retrain_$STAMP.log"
STATUS_FILE="$LOG_DIR/directional_retrain_$STAMP.status"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'code=$?; echo "failed exit_code=$code" > "$STATUS_FILE"; exit "$code"' ERR

cd "$PROJECT_DIR"
for path in "$PYTHON_BIN" "$DATASET_DIR/train" "$DATASET_DIR/test" \
  "$CONFIG" "$CLEAN_CHECKPOINT" "$DIFFUSION_CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path"; exit 2; }
done
echo "running time=$(date --iso-8601=seconds)" > "$STATUS_FILE"

if [[ "$RUN_TRAINING" == 1 ]]; then
  echo "[1/3] TRAIN NOISY DYNAMICS WITH CONTEXT/TIMESTEP-MATCHED RANKING"
  "$PYTHON_BIN" dynamics/main.py --mode train --device cuda \
    --data_dir "$DATASET_DIR/train" --test_data_dir "$DATASET_DIR/test" \
    --save_dir "$NOISY_DIR" --batch_size 32 --num_workers 8 --lr 1e-3 \
    --num_epochs 300 --patience 20 --val_step 5 --save_ckpt_step 500 \
    --output_dim 3 --use_design_noise \
    --num_train_timesteps "$TRAIN_TIMESTEPS" \
    --num_inference_steps "$INFERENCE_STEPS" \
    --num_timesteps_per_batch "$INFERENCE_STEPS" \
    --noise_timestep_sampling inference \
    --metric_loss_weights 1,1,2 --utility_weights 0.20,0.45,0.35 \
    --ranking_loss_weight 0.2 --ranking_margin 0.05 \
    --ranking_min_target_delta 0.05 \
    --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "noisy-context-rank-15x10-$STAMP"
fi

[[ -f "$NOISY_DIR/best.pt" ]] || { echo "ERROR: missing $NOISY_DIR/best.pt"; exit 2; }

echo "[2/3] DIAGNOSTICS"
"$PYTHON_BIN" -m benchmarks.diagnose_dgdm \
  --data_dir "$DATASET_DIR/test" --clean_checkpoint "$CLEAN_CHECKPOINT" \
  --noisy_checkpoint "$NOISY_DIR/best.pt" \
  --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
  --config "$CONFIG" --output_dir "$OUTPUT_ROOT/model_diagnostics" \
  --timesteps 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
  --max_samples 2048 --seed 0 --device cuda \
  --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
  --wandb_run_name "diagnostic-context-rank-15x10-$STAMP"

echo "[3/3] GENERATE AND SIMULATE ALL 16 CANDIDATES"
common=(--config "$CONFIG" --candidate_budget 16 --seeds "$SEEDS"
  --dynamics_checkpoint "$CLEAN_CHECKPOINT"
  --dgdm_dynamics_checkpoint "$NOISY_DIR/best.pt"
  --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" --device cuda
  --adam_steps 300 --adam_lr 0.03
  --cma_generations 100 --cma_popsize 32 --cma_sigma 0.5
  --diffusion_num_samples 256 --diffusion_batch_size 256
  --diffusion_inference_steps "$INFERENCE_STEPS"
  --utility_weights 0.45,0.20,0.35 --target_scenario_id all
  --evaluation_scope auto --run_benchmark --benchmark_top_k 16
  --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT")

"$PYTHON_BIN" -m benchmarks.run_baselines \
  --output_dir "$STUDY_DIR/base" \
  --methods adam,cma_es,conditional_diffusion "${common[@]}"

IFS=',' read -r -a scales <<< "$GUIDANCE_SCALES"
for scale in "${scales[@]}"; do
  label="dgdm_gs${scale//./p}"
  "$PYTHON_BIN" -m benchmarks.run_baselines \
    --output_dir "$STUDY_DIR/$label" --methods dgdm \
    --dgdm_guidance_scale "$scale" --dgdm_method_label "$label" \
    "${common[@]}"
done

"$PYTHON_BIN" -m benchmarks.analyze_candidate_pools \
  --study_dir "$STUDY_DIR" --output_dir "$ANALYSIS_DIR/pools"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$STUDY_DIR" --output_dir "$ANALYSIS_DIR/study_analysis" \
  --protocol specialist

echo "complete time=$(date --iso-8601=seconds)" > "$STATUS_FILE"
echo "[DONE] $OUTPUT_ROOT"
