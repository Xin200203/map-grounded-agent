#!/usr/bin/env bash
# Launch the R1 target-grounding cross-48 suite: 4 lanes
# (probe12 x {full,np} on GPUs 1/2, exp36 x {full,np} on GPUs 4/5).
# Run on the experiment server from the repo root, inside the unigoal env,
# with .local/deepseek.env.sh and .local/vapeur.env.sh sourced.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${DEEPSEEK_API_KEY:?source .local/deepseek.env.sh first}"
# vapeur channel not needed: R2 is graph-side only

CONFIG=base_UniGoal/configs/config_habitat_deepseek_recall_r1.yaml
STAMP="${STAMP:-20260710}"
# GPU ids per lane (probe-full, probe-np, exp-full, exp-np). Override via env
# when the default set is occupied, e.g. GPU_PF=4 GPU_PN=5 GPU_EF=6 GPU_EN=7.
GPU_PF="${GPU_PF:-1}"; GPU_PN="${GPU_PN:-2}"; GPU_EF="${GPU_EF:-4}"; GPU_EN="${GPU_EN:-5}"
PROBE12="228 527 661 64 159 358 486 574 717 778 859 955"
EXP36="65 79 93 160 170 180 229 248 267 359 372 385 487 500 513 528 543 558 575 585 595 662 674 686 718 734 750 779 788 797 860 886 912 956 970 984"

launch_lane() {
    local gpu="$1" episodes="$2" root="$3" profile="$4" log="$5"
    mkdir -p "${root}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    CONFIG_FILE="${CONFIG}" \
    EPISODES="${episodes}" \
    API_PROVIDER=openai \
    API_PROTOCOL=openai-chat-completions \
    RESULTS_ROOT="${root}" \
    nohup bash scripts/run_b3_intact_dev5.sh "${profile}" \
        > "${log}" 2>&1 &
    echo "lane gpu=${gpu} profile=${profile} root=${root} pid=$!"
}

launch_lane "${GPU_PF}" "${PROBE12}" "results/r1_probe12_${STAMP}" smoothnav-full        "results/r1_probe12_${STAMP}/driver_full.log"
launch_lane "${GPU_PN}" "${PROBE12}" "results/r1_probe12_${STAMP}" smoothnav-no-prefetch "results/r1_probe12_${STAMP}/driver_np.log"
launch_lane "${GPU_EF}" "${EXP36}"   "results/r1_exp36_${STAMP}"   smoothnav-full        "results/r1_exp36_${STAMP}/driver_full.log"
launch_lane "${GPU_EN}" "${EXP36}"   "results/r1_exp36_${STAMP}"   smoothnav-no-prefetch "results/r1_exp36_${STAMP}/driver_np.log"

echo "all lanes launched"
