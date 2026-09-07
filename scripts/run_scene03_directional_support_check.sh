#!/usr/bin/env bash
# Sanitation ablation: rescore the eight pose-DGDM gs=1 scene-03 candidates
# with direction-only contact-normal support, then render the robust winner.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_DIR/outputs/from_links_v16_pose_dgdm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v17_scene03_directional_support}"
CANDIDATES="$SOURCE_ROOT/one_seed_comparison/pose_dgdm_gs1/specialists/approach_radius-03/candidates/pose_dgdm_gs1_s0.npz"
CONFIG="$SOURCE_ROOT/one_seed_comparison/pose_dgdm_gs1/specialists/approach_radius-03/effective_config.json"

mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "$OUTPUT_ROOT/logs/scene03_support_$STAMP.log") 2>&1
cd "$PROJECT_DIR"

for path in "$PYTHON_BIN" "$CANDIDATES" "$CONFIG"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path"; exit 2; }
done

echo "[1/2] EVALUATE ALL 8 CANDIDATES X 5 INITIAL CONDITIONS"
"$PYTHON_BIN" -m benchmarks.run_sim_benchmark \
  --candidates "$CANDIDATES" \
  --output_dir "$OUTPUT_ROOT/scene03_all_candidates" \
  --config "$CONFIG" --scenario_ids 'approach_radius:03' \
  --top_k 8 --num_workers "${NUM_WORKERS:-20}" --timeout "${TIMEOUT:-1800}"

echo "[2/2] SELECT BY MEAN ORACLE UTILITY AND RENDER WINNER IN ALL 5 CONDITIONS"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$OUTPUT_ROOT/scene03_all_candidates" \
  --output_dir "$OUTPUT_ROOT/study_analysis" --protocol generalist \
  --render_best_generalist --num_workers 5 --timeout "${TIMEOUT:-1800}"

echo "[DONE] tables: $OUTPUT_ROOT/study_analysis"
echo "[DONE] videos: $OUTPUT_ROOT/study_analysis/best_generalist_visualizations"
