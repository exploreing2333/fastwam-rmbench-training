#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${project_root}/environment/activate.sh"
exec "${FASTWAM_ENV_PREFIX}/bin/python" "${project_root}/scripts/check_runtime.py" "$@"
