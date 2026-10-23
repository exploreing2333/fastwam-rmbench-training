#!/usr/bin/env bash
# Full instruction training, 50k per task, for the eight selected tasks.
set -euo pipefail

bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
tasks=(
  blocks_ranking_try
  cover_blocks
  observe_and_pickup
  press_button
  put_back_block
  rearrange_blocks
  swap_blocks
  swap_T
)

exec bash "${bundle_root}/train_tasks_sequential.sh" "$@" \
  --instruction-mode full --max-steps 50000 --tasks "${tasks[@]}"
