#!/usr/bin/env bash
# Resume only final-design PyElastica video and energy post-processing.
# This script intentionally never calls run_baselines or run_guidance_sweep.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/outputs/from_links_v3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
# Video encoding and PyElastica simulation are both CPU/memory intensive.
# Keep post-processing concurrency conservative; callers can override it.
NUM_WORKERS="${NUM_WORKERS:-4}"
TIMEOUT="${TIMEOUT:-1800}"
ENERGY_SELECTION="${ENERGY_SELECTION:-per_method}"

SPECIALIST_STUDY_DIR="${SPECIALIST_STUDY_DIR:-$RESULT_ROOT/nine_scenario_objectives}"
GENERALIST_STUDY_DIR="${GENERALIST_STUDY_DIR:-$RESULT_ROOT/nine_scenario_generalists}"
LOG_DIR="${LOG_DIR:-$RESULT_ROOT/study_logs}"

mkdir -p "$LOG_DIR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/postprocessing_$RUN_STAMP.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN"
  exit 2
fi

case "$ENERGY_SELECTION" in
  per_method)
    specialist_render_flag=--render_best_per_method
    generalist_render_flag=--render_best_generalist_per_method
    ;;
  overall)
    specialist_render_flag=--render_best_overall
    generalist_render_flag=--render_best_generalist
    ;;
  *)
    echo "ERROR: ENERGY_SELECTION must be per_method or overall"
    exit 2
    ;;
esac

echo "[POSTPROCESS START] selection=$ENERGY_SELECTION time=$(date --iso-8601=seconds)"
echo "[POSTPROCESS LOG] $LOG_FILE"

echo "[ENERGY START] protocol=specialist time=$(date --iso-8601=seconds)"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$SPECIALIST_STUDY_DIR" \
  --protocol specialist \
  "$specialist_render_flag" --measure_energy \
  --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"
echo "[ENERGY DONE] protocol=specialist time=$(date --iso-8601=seconds)"

echo "[ENERGY START] protocol=generalist time=$(date --iso-8601=seconds)"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$GENERALIST_STUDY_DIR" \
  --protocol generalist \
  "$generalist_render_flag" --measure_energy \
  --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"
echo "[ENERGY DONE] protocol=generalist time=$(date --iso-8601=seconds)"

echo "[POSTPROCESS DONE] time=$(date --iso-8601=seconds)"
