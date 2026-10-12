#!/usr/bin/env bash
set -euo pipefail
bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${bundle_root}/environment/activate.sh"
exec "${FASTWAM_ENV_PREFIX}/bin/python" "${bundle_root}/scripts/train_task.py" "$@"
