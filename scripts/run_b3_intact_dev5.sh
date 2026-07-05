#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export SMOOTHNAV_LLM_MAX_RETRIES="${SMOOTHNAV_LLM_MAX_RETRIES:-2}"
export SMOOTHNAV_LLM_RETRY_DELAYS="${SMOOTHNAV_LLM_RETRY_DELAYS:-1,3}"

LOCAL_CLAUDDY_ENV="${ROOT_DIR}/.local/clauddy.env.sh"
if [[ -f "${LOCAL_CLAUDDY_ENV}" ]]; then
    # shellcheck source=/dev/null
    source "${LOCAL_CLAUDDY_ENV}"
fi

CONFIG_FILE="${CONFIG_FILE:-base_UniGoal/configs/config_habitat.yaml}"
GOAL_TYPE="${GOAL_TYPE:-text}"
API_PROVIDER="${API_PROVIDER:-anthropic}"
API_PROTOCOL="${API_PROTOCOL:-anthropic-messages}"
RESULTS_ROOT="${RESULTS_ROOT:-results/phase2_revalidation/s2_dev5_intact_scene}"
EPISODES="${EPISODES:-286 287 291 294 296}"
read -r -a EPISODE_ARRAY <<< "${EPISODES}"

if [[ $# -gt 0 ]]; then
    PROFILES=("$@")
else
    PROFILES=(
        baseline-periodic
        smoothnav-fixed-interval
        smoothnav-no-monitor
        smoothnav-full
        smoothnav-no-prefetch
    )
fi

echo "Running B3 intact dev5 suite"
echo "Config: ${CONFIG_FILE}"
echo "Goal type: ${GOAL_TYPE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Episodes: ${EPISODES}"
echo "Profiles: ${PROFILES[*]}"
echo "Results root: ${RESULTS_ROOT}"

for profile in "${PROFILES[@]}"; do
    echo
    echo "===== PROFILE ${profile} ====="
    profile_root="${RESULTS_ROOT}/${profile}"
    mkdir -p "${profile_root}"
    for episode_id in "${EPISODE_ARRAY[@]}"; do
        echo "--- episode ${episode_id} ---"
        python smoothnav/main.py \
            --config-file "${CONFIG_FILE}" \
            --goal_type "${GOAL_TYPE}" \
            --controller-profile "${profile}" \
            --episode_id "${episode_id}" \
            --num_eval 1 \
            --api-provider "${API_PROVIDER}" \
            --api-protocol "${API_PROTOCOL}" \
            --results-root "${profile_root}"
    done
done

python scripts/summarize_explicit_suite.py \
    --results-root "${RESULTS_ROOT}" \
    --episodes "${EPISODE_ARRAY[@]}" \
    --profiles "${PROFILES[@]}"
