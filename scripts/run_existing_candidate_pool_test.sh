#!/usr/bin/env bash
# Re-evaluate all 16 candidates from the completed V7 run. No training or proposal generation.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
SOURCE="${SOURCE:-$PROJECT_DIR/outputs/from_links_v7_dgdm_retrain/one_seed_comparison}"
OUTPUT="${OUTPUT:-$PROJECT_DIR/outputs/from_links_v7_dgdm_retrain/all16_rebenchmark}"
ANALYSIS="${ANALYSIS:-$PROJECT_DIR/outputs/from_links_v7_dgdm_retrain/all16_analysis}"
TOP_K="${TOP_K:-16}"
NUM_WORKERS="${NUM_WORKERS:-20}"
TIMEOUT="${TIMEOUT:-1800}"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m benchmarks.rebenchmark_candidate_pools \
  --study_dir "$SOURCE" --output_dir "$OUTPUT" \
  --top_k "$TOP_K" --num_workers "$NUM_WORKERS" --timeout "$TIMEOUT"

"$PYTHON_BIN" -m benchmarks.analyze_candidate_pools \
  --study_dir "$OUTPUT" --output_dir "$ANALYSIS"

"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$OUTPUT" --output_dir "$ANALYSIS/study_analysis" \
  --protocol specialist

echo "[DONE] Existing candidate pool analysis: $ANALYSIS"
