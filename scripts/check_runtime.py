"""Check the active source, IPC, tensor transfer, GPUs and optional real data."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import sys
import socket
import tempfile

ROOT = Path(__file__).resolve().parents[1]
from fastwam.runtime_paths import configure_ipc_tmpdir
from training_processes import conflicting_training_processes


class TensorProbeDataset:
    def __len__(self):
        return 8

    def __getitem__(self, index):
        import torch
        return {'context': torch.full((128, 4096), index, dtype=torch.bfloat16), 'index': index}


def check_loader(torch, dataset, collate_fn=None, workers=2):
    kwargs = dict(batch_size=1, num_workers=workers, timeout=30 if workers else 0,
                  collate_fn=collate_fn)
    if workers:
        kwargs['prefetch_factor'] = 1
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    iterator = iter(loader)
    checked = 0
    try:
        for _ in range(min(4, len(dataset))):
            batch = next(iterator)
            assert torch.isfinite(batch['context']).all()
            assert tuple(batch['context'].shape) == (1, 128, 4096)
            if 'video' in batch:
                assert tuple(batch['video'].shape) == (1, 3, 9, 384, 320)
                assert tuple(batch['action'].shape) == (1, 32, 14)
                assert tuple(batch['proprio'].shape) == (1, 1, 14)
                assert batch['context_mask'].all()
                assert torch.isfinite(batch['video']).all() and torch.isfinite(batch['action']).all()
            checked += 1
            del batch
    finally:
        if workers:
            iterator._shutdown_workers()
    return checked


def check_cache_files(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = None
    renamed = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.fastwam-check-', dir=str(directory), delete=False) as stream:
            path = Path(stream.name)
            stream.write(b'fastwam-cache-check')
        renamed = path.with_name(path.name + '.renamed')
        path.replace(renamed)
        with renamed.open('r+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert stream.read() == b'fastwam-cache-check'
    finally:
        for item in (path, renamed):
            if item is not None:
                item.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--master-port', type=int)
    parser.add_argument('--cpu', action='store_true', help='Check IPC and data without accessing GPUs')
    parser.add_argument('--task', default='observe_and_pickup')
    parser.add_argument('--instruction-mode', choices=('subtask', 'full'), default='subtask')
    parser.add_argument('--real-data', action='store_true')
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'data/RMBench/data/data')
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'data/lerobot/rmbench_all_12tasks_demo_clean_50')
    parser.add_argument('--text-cache', type=Path, default=ROOT / 'data/gt_subtask_text_embeds')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.microbatch < 1 or args.num_workers < 0:
        parser.error('microbatch must be positive and num-workers must be nonnegative')
    if args.master_port is not None and not 1 <= args.master_port <= 65535:
        parser.error('master-port must be between 1 and 65535')
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    configure_ipc_tmpdir()
    os.environ.setdefault('FASTWAM_BUNDLE_ROOT', str(ROOT))
    runtime = Path(os.environ.get('FASTWAM_RUNTIME_ROOT', str(ROOT / '.runtime'))).resolve()
    os.environ['FASTWAM_RUNTIME_ROOT'] = str(runtime)
    import torch
    from omegaconf import OmegaConf
    import fastwam.runtime as active_runtime
    torch.set_num_threads(1)
    packages = {name: importlib.metadata.version(name) for name in ('torch', 'accelerate', 'deepspeed', 'transformers')}
    expected = {'torch': '2.7.1', 'accelerate': '1.12.0', 'deepspeed': '0.18.5', 'transformers': '4.57.1'}
    mismatches = [f'{name}={packages[name]} (expected {version})' for name, version in expected.items()
                  if packages[name].split('+', 1)[0] != version]
    if mismatches or torch.version.cuda != '12.8':
        raise RuntimeError('Bundled environment differs from the reference: '
                           + ', '.join(mismatches + ([f'PyTorch CUDA={torch.version.cuda}, expected 12.8'] if torch.version.cuda != '12.8' else [])))
    if Path(active_runtime.__file__).resolve() != ROOT / 'src/fastwam/runtime.py':
        raise RuntimeError('A different checkout supplies fastwam.runtime: ' + active_runtime.__file__)
    active_runtime._configure_runtime_cache(OmegaConf.create({'runtime_cache_root': str(runtime / 'cache')}))
    checked_directories = [ROOT / 'runs', runtime,
                           runtime / 'cache/huggingface/datasets', runtime / 'cache/torch',
                           runtime / 'cache/triton']
    for directory in checked_directories:
        check_cache_files(directory)
    with Listener(family='AF_UNIX') as listener:
        socket_address = listener.address
    probe_workers = min(2, args.num_workers)
    report = dict(passed=False, runtime_source=active_runtime.__file__, python=sys.executable,
                  packages=packages, pytorch_cuda=torch.version.cuda,
                  tmpdir=os.environ['TMPDIR'], tempfile_directory=tempfile.gettempdir(),
                  unix_socket_address=socket_address, unix_socket_passed=True,
                  tensor_loader_workers=probe_workers,
                  tensor_loader_batches=check_loader(torch, TensorProbeDataset(), workers=probe_workers),
                  cache_write_rename_lock_checks=[str(path) for path in checked_directories],
                  cpu_only=args.cpu, checked_at=datetime.now(timezone.utc).isoformat())
    print('[check] source=' + report['runtime_source'], flush=True)
    print('[check] TMPDIR=' + report['tmpdir'] + f'; AF_UNIX and {probe_workers}-worker tensor loading passed', flush=True)
    if not args.cpu:
        if torch.cuda.device_count() < args.gpus:
            raise RuntimeError(f'Requested {args.gpus} GPUs, visible count is {torch.cuda.device_count()}')
        uuids = [str(getattr(torch.cuda.get_device_properties(i), 'uuid', '')) for i in range(args.gpus)]
        conflicts = conflicting_training_processes(ROOT, args.gpus, uuids if all(uuids) else None)
        if conflicts:
            runs = sorted({row['run_dir'] for row in conflicts})
            raise RuntimeError('Existing training overlaps these GPUs: ' + ', '.join(runs)
                               + '. Use bash stop_training.sh --list to inspect it.')
        gpu_info = []
        for index in range(args.gpus):
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f'GPU {index} does not support BF16')
                free, total = torch.cuda.mem_get_info()
                gpu_info.append(dict(index=index, name=torch.cuda.get_device_name(index),
                                     free_gib=round(free / 2**30, 2), total_gib=round(total / 2**30, 2)))
                if free < 24 * 2**30:
                    raise RuntimeError(f'GPU {index} has only {free / 2**30:.1f} GiB free; inspect existing jobs before loading the model')
                parameter = torch.nn.Parameter(torch.ones((16, 16), device='cuda'))
                optimizer = torch.optim.AdamW([parameter], lr=1e-5, fused=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = (parameter @ parameter).square().mean()
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                assert torch.isfinite(parameter).all()
                del parameter, optimizer, loss
                torch.cuda.empty_cache()
                gpu_info[-1]['bf16_forward_backward_fused_adamw_passed'] = True
        report['gpus'] = gpu_info
        if args.num_workers:
            shm = os.statvfs('/dev/shm')
            free = shm.f_bavail * shm.f_frsize
            sample_bytes = 3 * 9 * 384 * 320 * 4 + 128 * 4096 * 2 + 32 * 14 * 4
            minimum = args.gpus * args.microbatch * sample_bytes
            estimate = minimum * (4 * args.num_workers + 2)
            report['shared_memory'] = dict(free_gib=round(free / 2**30, 3),
                                         estimated_prefetch_gib=round(estimate / 2**30, 3))
            if free < estimate:
                raise RuntimeError(f'/dev/shm has {free / 2**30:.2f} GiB free; '
                                   f'reference prefetch may need {estimate / 2**30:.2f} GiB. '
                                   'Increase container shared memory or launch with --num-workers 0.')
    if args.master_port is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
            try:
                tcp.bind(('127.0.0.1', args.master_port))
            except OSError as error:
                raise RuntimeError(f'Master port {args.master_port} is unavailable; select another --master-port: {error}') from error
        report['master_port_available'] = args.master_port
    if args.real_data:
        from hydra.utils import instantiate
        from fastwam.utils import misc
        cfg = OmegaConf.load(ROOT / f'configs/portable_{args.task}.yaml')
        from training_modes import configure_instruction_mode
        configure_instruction_mode(cfg, args.instruction_mode)
        cfg.data.train.episode_indices = [cfg.data.train.episode_indices[0]]
        cfg.data.train.dataset_dirs = [str(args.dataset_root.resolve())]
        if args.instruction_mode == 'subtask':
            cfg.data.train.gt_subtask_text_embedding_cache_dir = str(args.text_cache.resolve())
        else:
            from prepare_full_instruction_text_cache import prepare as prepare_text
            prepare_text(args.task, cfg.data.train.text_embedding_cache_dir)
        cfg.data.train.pretrained_norm_stats = str(ROOT / f'data/task_norm_stats/{args.task}.json')
        if cfg.model.get('init_frame_kv_config') is not None:
            from precompute_init_frame_latents import prepare
            # A one-episode runtime probe must not replace the full training cache.
            probe_cache = runtime / 'checks/init_frame_latents' / args.task
            prepare(args.dataset_root, probe_cache, cfg.data.train.episode_indices,
                    'cpu' if args.cpu else 'cuda:0')
            cfg.data.train.event_anchor_latent_cache_root = str(probe_cache)
            for task, annotation in cfg.data.train.get('gt_subtask_annotation_files', {}).items():
                cfg.data.train.gt_subtask_annotation_files[task] = str(
                    args.raw_root / task / 'demo_clean' / Path(annotation).name)
        source = json.loads((ROOT / 'data/source_map_template.json').read_text())
        source['raw_root'] = str(args.raw_root.resolve())
        with tempfile.TemporaryDirectory(prefix='fastwam-real-data-') as temporary:
            temporary = Path(temporary)
            mapping = temporary / 'source_map.json'
            mapping.write_text(json.dumps(source))
            if args.instruction_mode == 'subtask':
                cfg.data.train.subtask_source_map = str(mapping)
            misc.register_work_dir(temporary / 'samples')
            dataset = instantiate(cfg.data.train)
            arrow_files = [item['filename'] for child in dataset.lerobot_dataset.multi_dataset._datasets
                           for item in child.hf_dataset.cache_files]
            expected_cache = Path(os.environ['HF_DATASETS_CACHE']).resolve()
            if not arrow_files or any(expected_cache not in Path(name).resolve().parents for name in arrow_files):
                raise RuntimeError('LeRobot Arrow files are outside the configured HF_DATASETS_CACHE')
            report['arrow_cache_files'] = arrow_files
            subset = torch.utils.data.Subset(dataset, [min(i, len(dataset) - 1) for i in (0, 1, 43, 44)])
            report['real_data_batches'] = check_loader(torch, subset, getattr(dataset, 'collate_fn', None), probe_workers)
            report['task'] = args.task
    report['passed'] = True
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('RUNTIME_CHECK_PASSED ' + json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('RUNTIME_CHECK_FAILED ' + type(error).__name__ + ': ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
