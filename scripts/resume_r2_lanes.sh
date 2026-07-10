#!/usr/bin/env bash
# Resume interrupted r2 cross-48 lanes: for each lane, compute the episodes
# that already have an episode_results.json (plus any episode currently being
# run by a live worker) and relaunch the lane with only the remaining ones.
# Idempotent: lanes with nothing remaining are skipped.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${DEEPSEEK_API_KEY:?source .local/deepseek.env.sh first}"

CONFIG=base_UniGoal/configs/config_habitat_deepseek_recall_r2.yaml
STAMP="${STAMP:-20260710}"
PROBE12="228 527 661 64 159 358 486 574 717 778 859 955"
EXP36="65 79 93 160 170 180 229 248 267 359 372 385 487 500 513 528 543 558 575 585 595 662 674 686 718 734 750 779 788 797 860 886 912 956 970 984"

inflight_ids="$(ps -ef | grep '[m]ain.py' | grep -oE -- '--episode_id [0-9]+' | awk '{print $2}' | tr '\n' ' ')"
echo "in-flight episode ids: ${inflight_ids:-none}"

remaining_for() {
    local root="$1" profile="$2" episodes="$3"
    local done_ids
    done_ids="$(find "${root}/${profile}" -name episode_results.json 2>/dev/null \
        | xargs -r grep -ho '"habitat_episode_no": [0-9]*' \
        | grep -o '[0-9]*' | sort -u | tr '\n' ' ')"
    local out=""
    for ep in ${episodes}; do
        case " ${done_ids} ${inflight_ids} " in
            *" ${ep} "*) ;;
            *) out="${out} ${ep}" ;;
        esac
    done
    echo "${out# }"
}

launch_lane() {
    local gpu="$1" episodes="$2" root="$3" profile="$4" log="$5"
    local remaining
    remaining="$(remaining_for "${root}" "${profile}" "${episodes}")"
    if [[ -z "${remaining}" ]]; then
        echo "lane ${root}/${profile}: nothing remaining, skip"
        return 0
    fi
    mkdir -p "${root}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    CONFIG_FILE="${CONFIG}" \
    EPISODES="${remaining}" \
    API_PROVIDER=openai \
    API_PROTOCOL=openai-chat-completions \
    RESULTS_ROOT="${root}" \
    nohup bash scripts/run_b3_intact_dev5.sh "${profile}" \
        >> "${log}" 2>&1 &
    echo "lane gpu=${gpu} profile=${profile} remaining=[${remaining}] pid=$!"
}

launch_lane 1 "${PROBE12}" "results/r2_probe12_${STAMP}" smoothnav-full        "results/r2_probe12_${STAMP}/driver_full.log"
launch_lane 2 "${PROBE12}" "results/r2_probe12_${STAMP}" smoothnav-no-prefetch "results/r2_probe12_${STAMP}/driver_np.log"
launch_lane 0 "${EXP36}"   "results/r2_exp36_${STAMP}"   smoothnav-full        "results/r2_exp36_${STAMP}/driver_full.log"
launch_lane 3 "${EXP36}"   "results/r2_exp36_${STAMP}"   smoothnav-no-prefetch "results/r2_exp36_${STAMP}/driver_np.log"

echo "resume complete"
