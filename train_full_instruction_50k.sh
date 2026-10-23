#!/usr/bin/env bash
# Full instruction training, 50k per task; battery_try is already complete.
set -euo pipefail

bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
tasks=(
  press_button
  cover_blocks
  place_block_mat
  classify_blocks
  rearrange_blocks
  swap_blocks
  storage_blocks
  blocks_ranking_try
  put_back_block
  observe_and_pickup
  swap_T
)

exec bash "${bundle_root}/train_tasks_sequential.sh" "$@" \
  --instruction-mode full --max-steps 50000 --tasks "${tasks[@]}"
