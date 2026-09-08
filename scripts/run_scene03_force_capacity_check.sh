#!/usr/bin/env bash
# Re-evaluate existing pose-DGDM gs=1 scene-03 candidates with four-factor utility.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/real/Desktop/Squirrel_Gripper/ws/squirrel_gripper}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_DIR/outputs/from_links_v16_pose_dgdm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/from_links_v19_scene03_force_capacity}"
CONFIG="${CONFIG:-$PROJECT_DIR/benchmarks/scenarios_v6_force_four.json}"
CANDIDATES="$SOURCE_ROOT/one_seed_comparison/pose_dgdm_gs1/specialists/approach_radius-03/candidates/pose_dgdm_gs1_s0.npz"

mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "$OUTPUT_ROOT/logs/scene03_force_$STAMP.log") 2>&1
cd "$PROJECT_DIR"
for path in "$PYTHON_BIN" "$CONFIG" "$CANDIDATES"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path"; exit 2; }
done
cp "$CONFIG" "$OUTPUT_ROOT/effective_config.json"

echo "[1/2] EVALUATE 8 EXISTING CANDIDATES X 5 INITIAL CONDITIONS"
"$PYTHON_BIN" -m benchmarks.run_sim_benchmark \
  --candidates "$CANDIDATES" --output_dir "$OUTPUT_ROOT/scene03_all_candidates" \
  --config "$CONFIG" --scenario_ids 'approach_radius:03' \
  --top_k 8 --num_workers "${NUM_WORKERS:-30}" --timeout "${TIMEOUT:-1800}"

echo "[2/2] SELECT BY FOUR-FACTOR MEAN ORACLE UTILITY AND RENDER WINNER"
"$PYTHON_BIN" -m benchmarks.analyze_study \
  --study_dir "$OUTPUT_ROOT/scene03_all_candidates" \
  --output_dir "$OUTPUT_ROOT/study_analysis" --protocol generalist \
  --render_best_generalist --num_workers 5 --timeout "${TIMEOUT:-1800}"

touch "$OUTPUT_ROOT/POSTPROCESS_DONE.status"
echo "[DONE] $OUTPUT_ROOT"
