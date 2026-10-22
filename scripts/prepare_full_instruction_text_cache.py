"""Validate task instruction contexts; encode missing ones from bundled Wan weights."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import torch

from fastwam_qwen.datasets.robot_video_dataset import DEFAULT_PROMPT

ROOT = Path(__file__).resolve().parents[1]


def task_instructions(root, task):
    source = json.loads((Path(root) / 'data/source_map_template.json').read_text())
    texts = sorted({row['task'] for row in source['entries'] if row['source_task_name'] == task})
    if not texts:
        raise ValueError(f'No full task instructions for {task}.')
    return texts


def validate(path, context_len):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    context, mask = payload['context'], payload['mask']
    if (context.shape != (context_len, 4096) or mask.shape != (context_len,)
            or mask.dtype != torch.bool or not mask.any() or not torch.isfinite(context).all()):
        raise ValueError(f'Invalid full instruction T5 context: {path}')


def prepare(task, output_root, context_len=128, device='cpu', root=ROOT):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prompts = [DEFAULT_PROMPT.format(task=text) for text in task_instructions(root, task)]
    with (output_root / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing = []
        for prompt in prompts:
            name = hashlib.sha256(prompt.encode()).hexdigest() + f'.t5_len{context_len}.wan22ti2v5b.pt'
            path = output_root / name
            if path.is_file():
                validate(path, context_len)
            else:
                missing.append((prompt, path))
        if missing:
            from fastwam.models.wan22.helpers.loader import _resolve_configs, _load_registered_model
            from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
            os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH', str(Path(root) / 'models'))
            os.environ.setdefault('DIFFSYNTH_SKIP_DOWNLOAD', 'true')
            _, text_config, _, tokenizer_config = _resolve_configs(
                'Wan-AI/Wan2.2-TI2V-5B', 'Wan-AI/Wan2.1-T2V-1.3B', True)
            text_config.download_if_necessary()
            tokenizer_config.download_if_necessary()
            encoder = _load_registered_model(text_config.path, 'wan_video_text_encoder',
                                             torch_dtype=torch.bfloat16, device=device).eval()
            tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=context_len, clean='whitespace')
            with torch.inference_mode():
                for prompt, path in missing:
                    ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
                    mask = mask.to(device=device, dtype=torch.bool)
                    context = encoder(ids.to(device), mask)[0].to(device='cpu', dtype=torch.bfloat16)
                    pending = path.with_suffix(f'.{os.getpid()}.pending')
                    torch.save(dict(context=context.contiguous(), mask=mask[0].cpu().contiguous()), pending)
                    validate(pending, context_len)
                    pending.replace(path)
        print(f'[instruction cache] verified {len(prompts)} full instruction(s) for {task}: {output_root}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--context-len', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare(args.task, args.output_root, args.context_len, args.device)
