"""Rebuild exact GT-subtask Wan T5 contexts from bundled annotations/models."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from fastwam_qwen.datasets.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
manifest = json.loads((root / 'data/gt_subtask_text_embeds_manifest.json').read_text())
target = root / 'data/gt_subtask_text_embeds'
target.mkdir(exist_ok=True)
todo = []
for subtask in manifest['subtasks']:
    prompt = DEFAULT_PROMPT.format(task=subtask)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    path = target / f'{digest}.t5_len128.wan22ti2v5b.pt'
    if args.overwrite or not path.is_file():
        todo.append((prompt, path))
if not todo:
    print(f'All {manifest["count"]} GT-subtask contexts are already present.')
    raise SystemExit(0)
os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = str(root / 'models')
os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'true'
_, text, _, tokenizer = _resolve_configs('Wan-AI/Wan2.2-TI2V-5B', 'Wan-AI/Wan2.1-T2V-1.3B', True)
text.download_if_necessary()
tokenizer.download_if_necessary()
encoder = _load_registered_model(text.path, 'wan_video_text_encoder', torch.bfloat16, args.device).eval()
tokenizer = HuggingfaceTokenizer(name=tokenizer.path, seq_len=128, clean='whitespace')
with torch.inference_mode():
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        ids, mask = tokenizer([item[0] for item in batch], return_mask=True, add_special_tokens=True)
        mask = mask.to(device=args.device, dtype=torch.bool)
        contexts = encoder(ids.to(args.device), mask)
        for row, (_, path) in enumerate(batch):
            tmp = path.with_suffix(path.suffix + '.tmp')
            torch.save({'context': contexts[row].detach().cpu().to(torch.bfloat16).contiguous(),
                        'mask': mask[row].detach().cpu().contiguous()}, tmp)
            tmp.replace(path)
print(f'Built {len(todo)} exact GT-subtask contexts.')
