"""Verify relocated runtime imports without allocating any GPU."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

root = Path(__file__).resolve().parents[1]
prefix = Path(sys.prefix).resolve()
modules = {}
for name in ('torch', 'torchvision', 'accelerate', 'deepspeed', 'transformers', 'av',
             'h5py', 'pyarrow', 'cv2', 'hydra', 'omegaconf', 'safetensors',
             'fastwam', 'fastwam_qwen'):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(prefix) or path.is_relative_to(root), (name, path)
    modules[name] = str(path)
assert all('/home/cenzl/conda_env/fastwam' not in value for value in sys.path)
assert '/home/cenzl/VLA/LatentWAM/FastWAM/src' not in sys.path
import torch
assert torch.version.cuda == '12.8'
assert not torch.cuda.is_available(), 'This verification is intended to run with CUDA_VISIBLE_DEVICES empty.'
torch.set_num_threads(1)
assert torch.isfinite(torch.ones(2, 2) @ torch.ones(2, 2)).all()
report = dict(passed=True, executable=sys.executable, relocated_prefix=str(prefix),
              platform=platform.platform(), torch_version=torch.__version__, torch_cuda=torch.version.cuda,
              transformers_version=importlib.metadata.version('transformers'),
              deepspeed_version=importlib.metadata.version('deepspeed'), gpu_training_started=False,
              imports=modules, original_environment_required=False, original_source_required=False)
(root / 'reports/environment_relocation_audit.json').write_text(json.dumps(report, indent=2)+'\n')
print('ENVIRONMENT_RELOCATION_OK '+json.dumps(report), flush=True)
