#!/bin/bash
# SmoothNav run script
# Usage: ./run.sh [smoothnav|baseline] [ins-image|text] [num_episodes]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CLAUDDY_ENV="${ROOT_DIR}/.local/clauddy.env.sh"

if [[ -f "${LOCAL_CLAUDDY_ENV}" ]]; then
    # Prefer repo-local secrets so experiments can override stale shell/conda env.
    # shellcheck source=/dev/null
    source "${LOCAL_CLAUDDY_ENV}"
fi

MODE=${1:-baseline}
GOAL_TYPE=${2:-ins-image}
NUM_EVAL=${3:-1}
shift 3 2>/dev/null || true

cd "${ROOT_DIR}/base_UniGoal"

echo "Running SmoothNav: mode=$MODE, goal=$GOAL_TYPE, episodes=$NUM_EVAL"
echo "Results will be written under ${ROOT_DIR}/results"

"${PYTHON_BIN}" -m smoothnav.main \
    --config-file configs/config_habitat.yaml \
    --mode "${MODE}" \
    --goal_type "${GOAL_TYPE}" \
    --num_eval "${NUM_EVAL}" \
    "$@"
