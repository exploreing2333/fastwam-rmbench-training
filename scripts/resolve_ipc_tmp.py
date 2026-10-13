"""Configure socket-compatible TMPDIR before importing PyTorch/Accelerate."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastwam.runtime_paths import configure_ipc_tmpdir

if __name__ == '__main__':
    print(configure_ipc_tmpdir())
