#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if ! python3 - "${bundle_root}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
try:
    manifest_hash = hashlib.sha256((root / 'assets/manifest.json').read_bytes()).hexdigest()
    ready = json.loads((root / '.assets/READY').read_text())
    valid = ready.get('manifest_sha256') == manifest_hash
    valid = valid and (root / 'environment/fastwam_env.tar.gz').is_file()
    valid = valid and (root / 'environment/cuda12.1.tar.gz').is_file()
except (OSError, ValueError):
    valid = False
sys.exit(0 if valid else 1)
PY
then
  bash "${bundle_root}/download_assets.sh"
fi
runtime_root="${FASTWAM_RUNTIME_ROOT:-${bundle_root}/.runtime}"
export FASTWAM_RUNTIME_ROOT="${runtime_root}"
environment_prefix="${runtime_root}/fastwam"
mkdir -p "${runtime_root}"
exec 9>"${runtime_root}/bootstrap.lock"
flock -x 9

if [[ ! -f "${environment_prefix}/.bundle_prefix" ]] || [[ "$(cat "${environment_prefix}/.bundle_prefix" 2>/dev/null || true)" != "${environment_prefix}" ]]; then
  if [[ ! -s "${bundle_root}/environment/fastwam_env.tar.gz" ]]; then
    echo "The environment archive is missing or still being built." >&2
    exit 1
  fi
  if [[ -d "${environment_prefix}" ]]; then
    mv -- "${environment_prefix}" "${environment_prefix}.previous.$(date +%s)"
  fi
  mkdir -p "${environment_prefix}"
  echo "[setup] Restoring the bundled Python/CUDA training environment..." >&2
  tar -xzf "${bundle_root}/environment/fastwam_env.tar.gz" -C "${environment_prefix}"
  "${environment_prefix}/bin/python" "${environment_prefix}/bin/conda-unpack"
  FASTWAM_BUNDLE_ROOT="${bundle_root}" "${environment_prefix}/bin/python" - <<'PY'
import os
import sysconfig
from pathlib import Path
root = Path(os.environ['FASTWAM_BUNDLE_ROOT'])
site = Path(sysconfig.get_paths()['purelib'])
for path in site.glob('__editable__*.pth'):
    path.unlink()
(site / 'fastwam_bundle.pth').write_text('\n'.join(str(root / name) for name in ('src', 'vendor/curobo_src', 'vendor/LIBERO')) + '\n')
PY
  printf '%s\n' "${environment_prefix}" >"${environment_prefix}/.bundle_prefix"
fi

if [[ ! -f "${runtime_root}/cuda/.bundle_complete" ]]; then
  mkdir -p "${runtime_root}/cuda"
  echo "[setup] Restoring the bundled CUDA compiler toolkit..." >&2
  tar -xzf "${bundle_root}/environment/cuda12.1.tar.gz" -C "${runtime_root}/cuda"
  touch "${runtime_root}/cuda/.bundle_complete"
fi

FASTWAM_BUNDLE_ROOT="${bundle_root}" "${environment_prefix}/bin/python" "${bundle_root}/scripts/prepare_bundle_paths.py"
echo "[setup] Ready: ${bundle_root}" >&2
