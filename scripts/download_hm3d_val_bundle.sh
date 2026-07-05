#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/base_UniGoal"

: "${HM3D_USERNAME:?Set HM3D_USERNAME to your Matterport API token id}"
: "${HM3D_PASSWORD:?Set HM3D_PASSWORD to your Matterport API token secret}"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/base_UniGoal/data}"
UIDS="${UIDS:-hm3d_val_habitat_v0.2 hm3d_val_configs_v0.2 hm3d_val_semantic_annots_v0.2 hm3d_val_semantic_configs_v0.2}"
REPLACE_FLAG="${REPLACE_FLAG:---replace}"

echo "Downloading HM3D val assets to: ${DATA_PATH}"
echo "UIDS: ${UIDS}"

python -m habitat_sim.utils.datasets_download \
  --username "${HM3D_USERNAME}" \
  --password "${HM3D_PASSWORD}" \
  --data-path "${DATA_PATH}" \
  ${REPLACE_FLAG} \
  --uids ${UIDS}

echo
echo "Download complete. Next suggested step:"
echo "python ${ROOT_DIR}/scripts/audit_hm3d_scene_loadability.py --known-ok-scene MHPLjHsuG27"
