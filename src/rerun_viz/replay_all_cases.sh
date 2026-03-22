#!/usr/bin/env bash
#
# Run replay_recorded.py for all cases and save RRD files into a single output folder.
# Cases are discovered from data/different_types/. Skips cases missing required artifacts.
#
# To run in background and survive terminal close:
#   nohup ./src/rerun_viz/replay_all_cases.sh &
# Log file: replay_all.log (or LOG_FILE=my.log ./replay_all_cases.sh)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives under src/rerun_viz/; repository root is two levels up.
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_PATH="${BASE_PATH:-./data/different_types}"
OUTPUT_DIR="${OUTPUT_DIR:-./replay_rrds}"
REPLAY_OPTS="${REPLAY_OPTS:---medium}"  # --medium: original + spring stretch + masses + ground
LOG_FILE="${LOG_FILE:-replay_all.log}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

# Redirect all output to log file
exec >> "$LOG_FILE" 2>&1

echo "========================================"
echo "$(date -Iseconds) Replay all cases started"
echo "========================================"
echo "Base path: $BASE_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Replay options: $REPLAY_OPTS"
echo ""

for case_dir in "$BASE_PATH"/*/; do
  [[ -d "$case_dir" ]] || continue
  case_name="$(basename "$case_dir")"

  # Check required artifacts
  if [[ ! -f "$BASE_PATH/$case_name/final_data.pkl" ]]; then
    echo "[SKIP] $case_name: no final_data.pkl"
    continue
  fi
  if [[ ! -f "$BASE_PATH/$case_name/calibrate.pkl" ]]; then
    echo "[SKIP] $case_name: no calibrate.pkl"
    continue
  fi
  if [[ ! -f "$BASE_PATH/$case_name/metadata.json" ]]; then
    echo "[SKIP] $case_name: no metadata.json"
    continue
  fi
  if [[ ! -f "experiments_optimization/$case_name/optimal_params.pkl" ]]; then
    echo "[SKIP] $case_name: no optimal_params.pkl"
    continue
  fi
  if ! compgen -G "experiments/$case_name/train/best_*.pth" > /dev/null 2>&1; then
    echo "[SKIP] $case_name: no best_*.pth checkpoint"
    continue
  fi

  out_rrd="$OUTPUT_DIR/${case_name}.rrd"
  echo "[RUN] $(date -Iseconds) $case_name -> $out_rrd"
  if python -m rerun_viz.replay_recorded \
    --base_path "$BASE_PATH" \
    --case_name "$case_name" \
    --rerun_mode file \
    --output-rrd "$out_rrd" \
    $REPLAY_OPTS
  then
    echo "[OK] $case_name"
  else
    echo "[FAIL] $case_name (continuing to next)"
  fi
done

echo ""
echo "========================================"
echo "$(date -Iseconds) Done. RRD files in: $OUTPUT_DIR"
echo "========================================"
ls -la "$OUTPUT_DIR" 2>/dev/null || true
