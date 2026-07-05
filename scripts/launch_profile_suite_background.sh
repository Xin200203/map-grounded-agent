#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PROFILE="${1:?profile}"
RESULTS_ROOT="${2:?results-root}"
EPISODES="${3:?episodes}"
GPU="${4:?gpu}"
TAG="${5:-${PROFILE}}"
mkdir -p "${ROOT_DIR}/.omx/locks"
LOCK_NAME="$(printf '%s__%s' "$PROFILE" "$RESULTS_ROOT" | tr '/ :' '___')"
LOCK_PATH="${ROOT_DIR}/.omx/locks/${LOCK_NAME}.pid"
PATTERN="controller-profile ${PROFILE} .*results-root ${RESULTS_ROOT}"
if pgrep -af "$PATTERN" >/tmp/smoothnav_launcher_match.$$ 2>/dev/null; then
  if [ -s /tmp/smoothnav_launcher_match.$$ ]; then
    echo "existing_process"
    cat /tmp/smoothnav_launcher_match.$$
    rm -f /tmp/smoothnav_launcher_match.$$
    exit 0
  fi
fi
rm -f /tmp/smoothnav_launcher_match.$$
source /mnt/sdd/xxy/miniconda3/etc/profile.d/conda.sh
conda activate unigoal
if [[ -f ${ROOT_DIR}/.local/clauddy.env.sh ]]; then
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/.local/clauddy.env.sh"
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export EPISODES="${EPISODES}"
export RESULTS_ROOT="${RESULTS_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${ROOT_DIR}/${RESULTS_ROOT}/${TAG}_${STAMP}.launcher.log"
mkdir -p "$(dirname "$LOG_PATH")"
nohup bash -lc "cd '${ROOT_DIR}' && ./scripts/run_b3_intact_dev5.sh '${PROFILE}'" >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$LOCK_PATH"
echo "launched ${PROFILE} pid=${PID} log=${LOG_PATH}"
