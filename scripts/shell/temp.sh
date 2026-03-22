# python script_optimize.py
# python script_train.py
# python script_inference.py

# python export_gaussian_data.py
# python export_video_human_mask.py

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/shell/gs_run.sh"