#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
bash "${project_root}/download_assets.sh" "$@"
bash "${project_root}/bootstrap.sh"
