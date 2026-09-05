#!/usr/bin/env bash
# Train and evaluate the additive pose-guided DGDM formulation.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/TendonForces/runs/exp1}"
CONFIG="${CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v5_robust_four.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v16_pose_dgdm}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$PROJECT_DIR/outputs/from_links_v14_fixed10_span360/diffusion_conditional_15/best.pt}"
WANDB_PROJECT="${WANDB_PROJECT:-squirrel-gripper-pose-dgdm}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
NUM_WORKERS="${NUM_WORKERS:-30}"
SEED="${SEED:-0}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-0.5,1,2}"
GUIDANCE_TIMESTEPS="${GUIDANCE_TIMESTEPS:-0,3,6}"

CLEAN_DIR="$OUTPUT_ROOT/pose_dynamics_clean"
NOISY_DIR="$OUTPUT_ROOT/pose_dynamics_noisy_t036"
CLEAN_CHECKPOINT="${CLEAN_CHECKPOINT:-$CLEAN_DIR/best.pt}"
NOISY_CHECKPOINT="${NOISY_CHECKPOINT:-$NOISY_DIR/best.pt}"
STUDY_DIR="$OUTPUT_ROOT/one_seed_comparison"
mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "$OUTPUT_ROOT/logs/pose_ablation_$STAMP.log") 2>&1

cd "$PROJECT_DIR"
for path in "$PYTHON_BIN" "$DATASET_DIR/train" "$DATASET_DIR/test" "$CONFIG" "$DIFFUSION_CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path"; exit 2; }
done

echo "[1/5] AUDIT POSE LABELS"
"$PYTHON_BIN" -m benchmarks.audit_pose_dataset "$DATASET_DIR/train" "$DATASET_DIR/test"

train_pose_model() {
  local output_dir="$1"
  shift
  "$PYTHON_BIN" dynamics/main.py --mode train --device cuda \
    --data_dir "$DATASET_DIR/train" --test_data_dir "$DATASET_DIR/test" \
    --save_dir "$output_dir" --target_representation pose_keypoints \
    --pose_scale_m 0.10 --pose_contact_sigma_m 0.005 \
    --batch_size 32 --lr 1e-3 --num_epochs 300 --patience 100 --val_step 5 \
    --num_workers 8 --seed "$SEED" --model_architecture legacy --num_hidden_layers 3 \
    --ranking_loss_weight 0 --wandb_project "$WANDB_PROJECT" \
    --wandb_mode "$WANDB_MODE" "$@"
}

if [[ "$RUN_TRAINING" == 1 ]]; then
  echo "[2/5] TRAIN CLEAN 10-D KEYPOINT-POSE DYNAMICS"
  train_pose_model "$CLEAN_DIR" --wandb_run_name "pose-clean-$STAMP"
  echo "[3/5] TRAIN NOISE-CONDITIONED POSE DYNAMICS (ACTIVE 10 DESIGN VARIABLES ONLY)"
  train_pose_model "$NOISY_DIR" --use_design_noise --num_train_timesteps 15 \
    --num_inference_steps 5 --num_timesteps_per_batch 3 \
    --noise_timesteps "$GUIDANCE_TIMESTEPS" --wandb_run_name "pose-noisy-t036-$STAMP"
fi

for checkpoint in "$CLEAN_CHECKPOINT" "$NOISY_CHECKPOINT"; do
  [[ -f "$checkpoint" ]] || { echo "ERROR: expected checkpoint missing: $checkpoint"; exit 2; }
done

echo "[4/5] HELD-OUT POSE, PROXY-METRIC, AND GRADIENT DIAGNOSTICS"
"$PYTHON_BIN" -m benchmarks.diagnose_dgdm \
  --data_dir "$DATASET_DIR/test" --clean_checkpoint "$CLEAN_CHECKPOINT" \
  --noisy_checkpoint "$NOISY_CHECKPOINT" --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" \
  --config "$CONFIG" --output_dir "$OUTPUT_ROOT/model_diagnostics" \
  --timesteps "$GUIDANCE_TIMESTEPS" --max_samples 2048 --seed "$SEED" --device cuda \
  --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
  --wandb_run_name "pose-diagnostics-$STAMP"

if [[ "$RUN_EVALUATION" == 1 ]]; then
  echo "[5/5] ONE-SEED POSE-BASED ADAM/CMA/DIFFUSION/DGDM ABLATION"
  common=(--config "$CONFIG" --candidate_budget 8 --seeds "$SEED"
    --dynamics_checkpoint "$CLEAN_CHECKPOINT"
    --dgdm_dynamics_checkpoint "$NOISY_CHECKPOINT"
    --diffusion_checkpoint "$DIFFUSION_CHECKPOINT" --device cuda
    --adam_steps 300 --adam_lr 0.03 --cma_generations 100 --cma_popsize 32 --cma_sigma 0.5
    --diffusion_num_samples 256 --diffusion_batch_size 256 --diffusion_inference_steps 5
    --utility_weights 0.45,0.20,0.35 --dgdm_guidance_timesteps "$GUIDANCE_TIMESTEPS"
    --target_scenario_id all --evaluation_scope auto --run_benchmark --resume
    --benchmark_top_k 8 --num_workers "$NUM_WORKERS" --timeout 1800)
  "$PYTHON_BIN" -m benchmarks.run_baselines --output_dir "$STUDY_DIR/base" \
    --methods adam,cma_es,conditional_diffusion "${common[@]}"
  IFS=',' read -r -a scales <<< "$GUIDANCE_SCALES"
  for scale in "${scales[@]}"; do
    label="pose_dgdm_gs${scale//./p}"
    "$PYTHON_BIN" -m benchmarks.run_baselines --output_dir "$STUDY_DIR/$label" \
      --methods dgdm --dgdm_guidance_scale "$scale" --dgdm_method_label "$label" \
      "${common[@]}"
  done
  "$PYTHON_BIN" -m benchmarks.analyze_study --study_dir "$STUDY_DIR" \
    --output_dir "$OUTPUT_ROOT/study_analysis" --config "$CONFIG"
fi

echo "[POSE DGDM ABLATION DONE] $OUTPUT_ROOT"
