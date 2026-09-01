#!/usr/bin/env bash
# Run corrected 9-scenario specialist/generalist studies, guidance sweep,
# analyses, and optional
# final-design energy/video evaluations without interactive supervision.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/TendonForces/runs/exp1}"
DYNAMICS_DIR="${DYNAMICS_DIR:-$PROJECT_DIR/outputs/from_links_v2/dynamics}"
DIFFUSION_DIR="${DIFFUSION_DIR:-$PROJECT_DIR/outputs/from_links_v2/diffusion}"
UNCONDITIONAL_DIFFUSION_DIR="${UNCONDITIONAL_DIFFUSION_DIR:-$PROJECT_DIR/outputs/from_links_v2/unconditional_diffusion}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v2_compact.json}"

# V3 intentionally separates corrected clamped-surrogate runs from older V2 results.
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/outputs/from_links_v3}"
DGDM_DYNAMICS_DIR="${DGDM_DYNAMICS_DIR:-$RESULT_ROOT/dynamics_noisy}"
SPECIALIST_STUDY_DIR="${SPECIALIST_STUDY_DIR:-$RESULT_ROOT/nine_scenario_objectives}"
GENERALIST_STUDY_DIR="${GENERALIST_STUDY_DIR:-$RESULT_ROOT/nine_scenario_generalists}"
LOG_DIR="${LOG_DIR:-$RESULT_ROOT/study_logs}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-30}"
TIMEOUT="${TIMEOUT:-1800}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
PROFILES="${PROFILES:-combined contact_only disturbance_only}"
METHODS="${METHODS:-reference,random,random_search,retrieval,adam,cma_es,unconditional_diffusion,unconditional_dgdm,conditional_diffusion,dgdm}"

# per_method performs the scientifically useful energy comparison but adds many
# rendered simulations. Set to overall for one winner per objective, or none.
ENERGY_SELECTION="${ENERGY_SELECTION:-per_method}"
# Keep 0 on a 30-core/single-GPU machine. Set to 1 only if roughly 60 simulator
# workers and concurrent GPU proposal generation are known to fit.
RUN_STUDIES_IN_PARALLEL="${RUN_STUDIES_IN_PARALLEL:-0}"
RUN_GUIDANCE_SWEEP="${RUN_GUIDANCE_SWEEP:-1}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-0,0.1,1,2,10}"
GUIDANCE_SEEDS="${GUIDANCE_SEEDS:-$SEEDS}"
GUIDANCE_SWEEP_DIR="${GUIDANCE_SWEEP_DIR:-$RESULT_ROOT/guidance_scale_sweep/generalist}"
# auto reuses best.pt when present and trains it otherwise; use always to
# retrain intentionally or never to require an existing checkpoint.
TRAIN_DGDM_DYNAMICS="${TRAIN_DGDM_DYNAMICS:-auto}"
DGDM_DYNAMICS_EPOCHS="${DGDM_DYNAMICS_EPOCHS:-300}"
DGDM_DYNAMICS_BATCH_SIZE="${DGDM_DYNAMICS_BATCH_SIZE:-32}"
DGDM_DYNAMICS_WORKERS="${DGDM_DYNAMICS_WORKERS:-8}"
DGDM_DYNAMICS_LR="${DGDM_DYNAMICS_LR:-1e-3}"
DGDM_DYNAMICS_PATIENCE="${DGDM_DYNAMICS_PATIENCE:-10}"
DIFFUSION_TRAIN_TIMESTEPS="${DIFFUSION_TRAIN_TIMESTEPS:-100}"
NOISY_TIMESTEPS_PER_BATCH="${NOISY_TIMESTEPS_PER_BATCH:-4}"

mkdir -p "$LOG_DIR" "$SPECIALIST_STUDY_DIR" "$GENERALIST_STUDY_DIR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/design_studies_$RUN_STAMP.log}"
STATUS_FILE="${STATUS_FILE:-$LOG_DIR/design_studies_$RUN_STAMP.status}"
exec > >(tee -a "$LOG_FILE") 2>&1

exec 9>"$RESULT_ROOT/.design_studies.lock"
if ! flock -n 9; then
  echo "ERROR: another study script already holds $RESULT_ROOT/.design_studies.lock"
  exit 1
fi

on_error() {
  local code=$?
  echo "failed exit_code=$code time=$(date --iso-8601=seconds)" >"$STATUS_FILE"
  echo "[FAILED] exit_code=$code; inspect $LOG_FILE"
  exit "$code"
}
trap on_error ERR

cd "$PROJECT_DIR"

for required in \
  "$PYTHON_BIN" \
  "$DYNAMICS_DIR/best.pt" \
  "$DIFFUSION_DIR/best.pt" \
  "$UNCONDITIONAL_DIFFUSION_DIR/best.pt" \
  "$BENCHMARK_CONFIG" \
  "$DATASET_DIR/train" \
  "$DATASET_DIR/test"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required path does not exist: $required"
    exit 2
  fi
done

echo "running pid=$$ start=$(date --iso-8601=seconds) log=$LOG_FILE" >"$STATUS_FILE"
echo "[START] $(date --iso-8601=seconds)"
echo "[PATH] specialist=$SPECIALIST_STUDY_DIR"
echo "[PATH] generalist=$GENERALIST_STUDY_DIR"
echo "[CONFIG] workers=$NUM_WORKERS seeds=$SEEDS profiles=$PROFILES energy=$ENERGY_SELECTION"
echo "[CONFIG] guidance_sweep=$RUN_GUIDANCE_SWEEP scales=$GUIDANCE_SCALES"
echo "[CONFIG] noisy_dynamics=$DGDM_DYNAMICS_DIR train_mode=$TRAIN_DGDM_DYNAMICS"

train_dgdm_dynamics() {
  local checkpoint="$DGDM_DYNAMICS_DIR/best.pt"
  if [[ "$TRAIN_DGDM_DYNAMICS" == "auto" && -f "$checkpoint" ]]; then
    echo "[NOISY DYNAMICS REUSE] $checkpoint"
    return
  fi
  if [[ "$TRAIN_DGDM_DYNAMICS" == "never" ]]; then
    if [[ ! -f "$checkpoint" ]]; then
      echo "ERROR: TRAIN_DGDM_DYNAMICS=never but checkpoint is missing: $checkpoint"
      return 2
    fi
    echo "[NOISY DYNAMICS REUSE] $checkpoint"
    return
  fi
  if [[ "$TRAIN_DGDM_DYNAMICS" != "auto" && "$TRAIN_DGDM_DYNAMICS" != "always" ]]; then
    echo "ERROR: TRAIN_DGDM_DYNAMICS must be auto, always, or never"
    return 2
  fi

  mkdir -p "$DGDM_DYNAMICS_DIR"
  echo "[NOISY DYNAMICS TRAIN START] time=$(date --iso-8601=seconds)"
  "$PYTHON_BIN" dynamics/main.py \
    --mode train \
    --device "$DEVICE" \
    --data_dir "$DATASET_DIR/train" \
    --test_data_dir "$DATASET_DIR/test" \
    --save_dir "$DGDM_DYNAMICS_DIR" \
    --batch_size "$DGDM_DYNAMICS_BATCH_SIZE" \
    --num_workers "$DGDM_DYNAMICS_WORKERS" \
    --lr "$DGDM_DYNAMICS_LR" \
    --num_epochs "$DGDM_DYNAMICS_EPOCHS" \
    --patience "$DGDM_DYNAMICS_PATIENCE" \
    --val_step 5 \
    --save_ckpt_step 500 \
    --output_dim 3 \
    --use_design_noise \
    --num_train_timesteps "$DIFFUSION_TRAIN_TIMESTEPS" \
    --num_inference_steps 20 \
    --num_timesteps_per_batch "$NOISY_TIMESTEPS_PER_BATCH"
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: noisy dynamics training finished without creating $checkpoint"
    return 2
  fi
  echo "[NOISY DYNAMICS TRAIN DONE] checkpoint=$checkpoint time=$(date --iso-8601=seconds)"
}

train_dgdm_dynamics

run_benchmark_profile() {
  local protocol=$1
  local profile=$2
  local study_root target_args
  if [[ "$protocol" == "specialist" ]]; then
    study_root="$SPECIALIST_STUDY_DIR"
    target_args=(--target_scenario_id all)
  else
    study_root="$GENERALIST_STUDY_DIR"
    target_args=(--generalist)
  fi

  echo "[BENCHMARK START] protocol=$protocol profile=$profile time=$(date --iso-8601=seconds)"
  "$PYTHON_BIN" -m benchmarks.run_baselines \
    --output_dir "$study_root/$profile" \
    --config "$BENCHMARK_CONFIG" \
    --methods "$METHODS" \
    --candidate_budget 16 \
    --seeds "$SEEDS" \
    --retrieval_data_dir "$DATASET_DIR/train" \
    --dynamics_checkpoint "$DYNAMICS_DIR/best.pt" \
    --dgdm_dynamics_checkpoint "$DGDM_DYNAMICS_DIR/best.pt" \
    --diffusion_checkpoint "$DIFFUSION_DIR/best.pt" \
    --unconditional_diffusion_checkpoint "$UNCONDITIONAL_DIFFUSION_DIR/best.pt" \
    --device "$DEVICE" \
    --random_pool_size 256 \
    --adam_steps 300 \
    --adam_lr 0.03 \
    --cma_generations 100 \
    --cma_popsize 32 \
    --cma_sigma 0.5 \
    --diffusion_num_samples 256 \
    --diffusion_batch_size 256 \
    --diffusion_inference_steps 20 \
    --dgdm_guidance_scale 0.1 \
    --utility_profile "$profile" \
    "${target_args[@]}" \
    --evaluation_scope auto \
    --run_benchmark \
    --benchmark_top_k 1 \
    --num_workers "$NUM_WORKERS" \
    --timeout "$TIMEOUT"
  echo "[BENCHMARK DONE] protocol=$protocol profile=$profile time=$(date --iso-8601=seconds)"
}

run_regular_analysis() {
  local protocol=$1
  local study_root=$2
  echo "[ANALYSIS START] protocol=$protocol time=$(date --iso-8601=seconds)"
  "$PYTHON_BIN" -m benchmarks.analyze_study \
    --study_dir "$study_root" \
    --protocol "$protocol"
  echo "[ANALYSIS DONE] protocol=$protocol time=$(date --iso-8601=seconds)"
}

run_energy_analysis() {
  local protocol=$1
  local study_root=$2
  echo "[ENERGY START] protocol=$protocol time=$(date --iso-8601=seconds)"
  if [[ "$ENERGY_SELECTION" == "none" ]]; then
    echo "[ENERGY SKIPPED] protocol=$protocol"
  elif [[ "$protocol" == "specialist" && "$ENERGY_SELECTION" == "per_method" ]]; then
    "$PYTHON_BIN" -m benchmarks.analyze_study \
      --study_dir "$study_root" --protocol specialist \
      --render_best_per_method --measure_energy \
      --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"
  elif [[ "$protocol" == "specialist" && "$ENERGY_SELECTION" == "overall" ]]; then
    "$PYTHON_BIN" -m benchmarks.analyze_study \
      --study_dir "$study_root" --protocol specialist \
      --render_best_overall --measure_energy \
      --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"
  elif [[ "$protocol" == "generalist" && "$ENERGY_SELECTION" == "per_method" ]]; then
    "$PYTHON_BIN" -m benchmarks.analyze_study \
      --study_dir "$study_root" --protocol generalist \
      --render_best_generalist_per_method --measure_energy \
      --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"
  elif [[ "$protocol" == "generalist" && "$ENERGY_SELECTION" == "overall" ]]; then
    "$PYTHON_BIN" -m benchmarks.analyze_study \
      --study_dir "$study_root" --protocol generalist \
      --render_best_generalist --measure_energy \
      --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"
  else
    echo "ERROR: ENERGY_SELECTION must be per_method, overall, or none"
    return 2
  fi
  echo "[ENERGY DONE] protocol=$protocol time=$(date --iso-8601=seconds)"
}

run_pipeline() {
  local protocol=$1
  local study_root=$2
  local profile
  for profile in $PROFILES; do
    run_benchmark_profile "$protocol" "$profile"
  done
  run_regular_analysis "$protocol" "$study_root"
}

run_guidance_sweep() {
  if [[ "$RUN_GUIDANCE_SWEEP" != "1" ]]; then
    echo "[GUIDANCE SWEEP SKIPPED]"
    return
  fi
  echo "[GUIDANCE SWEEP START] time=$(date --iso-8601=seconds)"
  "$PYTHON_BIN" -m benchmarks.run_guidance_sweep \
    --output_dir "$GUIDANCE_SWEEP_DIR" \
    --config "$BENCHMARK_CONFIG" \
    --diffusion_checkpoint "$DIFFUSION_DIR/best.pt" \
    --dynamics_checkpoint "$DYNAMICS_DIR/best.pt" \
    --dgdm_dynamics_checkpoint "$DGDM_DYNAMICS_DIR/best.pt" \
    --scales "$GUIDANCE_SCALES" \
    --seeds "$GUIDANCE_SEEDS" \
    --candidate_budget 16 \
    --num_samples 256 \
    --batch_size 256 \
    --inference_steps 20 \
    --generalist \
    --device "$DEVICE" \
    --run_benchmark \
    --benchmark_top_k 1 \
    --num_workers 9 \
    --timeout "$TIMEOUT"

  "$PYTHON_BIN" -m benchmarks.analyze_study \
    --study_dir "$GUIDANCE_SWEEP_DIR" \
    --output_dir "$GUIDANCE_SWEEP_DIR/study_analysis" \
    --protocol generalist
  echo "[GUIDANCE SWEEP DONE] time=$(date --iso-8601=seconds)"
}

if [[ "$RUN_STUDIES_IN_PARALLEL" == "1" ]]; then
  echo "[WARNING] specialist and generalist pipelines will share GPU and CPU resources"
  run_pipeline specialist "$SPECIALIST_STUDY_DIR" &
  specialist_pid=$!
  run_pipeline generalist "$GENERALIST_STUDY_DIR" &
  generalist_pid=$!
  wait "$specialist_pid"
  wait "$generalist_pid"
else
  run_pipeline specialist "$SPECIALIST_STUDY_DIR"
  run_pipeline generalist "$GENERALIST_STUDY_DIR"
fi

# Run the numerical guidance study before expensive video/energy reruns.
run_guidance_sweep
run_energy_analysis specialist "$SPECIALIST_STUDY_DIR"
run_energy_analysis generalist "$GENERALIST_STUDY_DIR"

echo "complete time=$(date --iso-8601=seconds)" >"$STATUS_FILE"
echo "[COMPLETE] $(date --iso-8601=seconds)"
echo "[STATUS] $STATUS_FILE"
echo "[LOG] $LOG_FILE"
