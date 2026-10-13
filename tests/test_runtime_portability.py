"""Regression checks for unsupported filesystems and training process ownership."""
import errno
import json
from multiprocessing.connection import Listener
import multiprocessing.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fastwam import runtime_paths
from training_processes import conflicting_training_processes, find_training_processes, still_same_process


class RuntimePathsTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''})
        self.environment.start()
        os.environ.pop('FASTWAM_IPC_TMPDIR', None)
        os.environ.pop('TMPDIR', None)
        self.cached = tempfile.tempdir
        tempfile.tempdir = None
        self.temporary = tempfile.TemporaryDirectory(prefix='fastwam-path-test-', dir='/tmp')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        # Finalize the test's pymp directories before deleting their parent.
        for finalizer in list(multiprocessing.util._finalizer_registry.values()):
            if finalizer._callback is multiprocessing.util._remove_temp_dir:
                directory = Path(finalizer._args[-1])
                if self.root in directory.parents:
                    finalizer()
        self.temporary.cleanup()
        self.environment.stop()
        tempfile.tempdir = self.cached

    def test_explicit_directory_resets_cached_tempdir_and_hosts_listener(self):
        directory = self.root / 'local'
        os.environ['FASTWAM_IPC_TMPDIR'] = str(directory)
        tempfile.tempdir = '/incorrect-cached-path'
        self.assertEqual(runtime_paths.configure_ipc_tmpdir(), directory)
        self.assertEqual(Path(tempfile.gettempdir()), directory)
        with Listener(family='AF_UNIX') as listener:
            self.assertTrue(str(listener.address).startswith(str(directory)))

    def test_unsupported_inherited_filesystem_falls_back(self):
        directory = self.root / 'cpfs'
        os.environ['TMPDIR'] = str(directory)
        native_probe = runtime_paths._probe_unix_sockets

        def probe(path):
            if path == directory:
                raise OSError(errno.EOPNOTSUPP, 'Operation not supported')
            return native_probe(path)

        with patch.object(runtime_paths, '_probe_unix_sockets', side_effect=probe):
            chosen = runtime_paths.configure_ipc_tmpdir()
        self.assertNotEqual(chosen, directory)
        with Listener(family='AF_UNIX') as listener:
            self.assertTrue(str(listener.address).startswith(str(chosen)))

    def test_invalid_explicit_path_fails_with_actionable_error(self):
        os.environ['FASTWAM_IPC_TMPDIR'] = str(self.root / 'cpfs')
        with patch.object(runtime_paths, '_probe_unix_sockets', side_effect=OSError(errno.EOPNOTSUPP, 'Operation not supported')):
            with self.assertRaisesRegex(RuntimeError, 'FASTWAM_IPC_TMPDIR.*local disk'):
                runtime_paths.configure_ipc_tmpdir()

    def test_overlong_inherited_socket_path_falls_back(self):
        directory = self.root / ('a' * 70) / ('b' * 70)
        os.environ['TMPDIR'] = str(directory)
        chosen = runtime_paths.configure_ipc_tmpdir()
        self.assertNotEqual(chosen, directory)
        with Listener(family='AF_UNIX'):
            pass

    def test_runtime_cache_configuration_preserves_local_ipc(self):
        from omegaconf import OmegaConf
        from fastwam.runtime import _configure_runtime_cache
        local = self.root / 'ipc'
        os.environ['FASTWAM_IPC_TMPDIR'] = str(local)
        cache = self.root / 'cpfs-cache'
        cached_mp = cache / 'tmp/pymp-old'
        cached_mp.mkdir(parents=True)
        __import__('multiprocessing').current_process()._config['tempdir'] = str(cached_mp)
        _configure_runtime_cache(OmegaConf.create({'runtime_cache_root': str(cache)}))
        self.assertEqual(Path(os.environ['TMPDIR']), local)
        self.assertEqual(Path(os.environ['HF_DATASETS_CACHE']), cache / 'huggingface/datasets')
        with Listener(family='AF_UNIX') as listener:
            self.assertTrue(str(listener.address).startswith(str(local)))

    def test_parquet_cache_honors_runtime_environment_after_early_import(self):
        from datasets import config
        import pyarrow as pa
        import pyarrow.parquet as pq
        from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset
        cache = self.root / 'configured-arrow-cache'
        self.assertNotEqual(Path(config.HF_DATASETS_CACHE), cache)
        os.environ['HF_DATASETS_CACHE'] = str(cache)
        raw = self.root / 'raw'
        (raw / 'data').mkdir(parents=True)
        pq.write_table(pa.table({'index': [0, 1], 'action': [[1.0, 2.0], [3.0, 4.0]]}), raw / 'data/episode_000000.parquet')
        for episodes in (None, (0,)):
            with self.subTest(episodes=episodes):
                dataset = LeRobotDataset.__new__(LeRobotDataset)
                dataset.root = raw
                dataset.episodes = episodes
                dataset.meta = SimpleNamespace(get_data_file_path=lambda episode: 'data/episode_000000.parquet')
                loaded = dataset.load_hf_dataset()
                self.assertEqual(len(loaded), 2)
                self.assertTrue(loaded.cache_files)
                self.assertTrue(all(cache in Path(row['filename']).parents for row in loaded.cache_files))
                self.assertEqual(loaded[0]['action'].tolist(), [1.0, 2.0])
                self.assertEqual(loaded[1]['action'].tolist(), [3.0, 4.0])


class ProcessOwnershipTests(unittest.TestCase):
    def test_only_checkout_training_is_selected_and_disjoint_gpus_are_allowed(self):
        run = ROOT / 'runs' / ('_process_test_' + uuid.uuid4().hex)
        run.mkdir(parents=True)
        (run / 'launch_command.json').write_text(json.dumps({'gpus': 2, 'CUDA_VISIBLE_DEVICES': '0,1'}))
        sleeper = 'import time; time.sleep(30)'
        owned = subprocess.Popen([sys.executable, '-c', sleeper, str(ROOT / 'scripts/train.py'), '--config-path', str(run)])
        unrelated = subprocess.Popen([sys.executable, '-c', sleeper, '/tmp/another-project/scripts/train.py', '--config-path', '/tmp/another-project/runs/run'])
        try:
            rows = find_training_processes(ROOT)
            row = next(item for item in rows if item['pid'] == owned.pid)
            self.assertNotIn(unrelated.pid, [item['pid'] for item in rows])
            self.assertTrue(still_same_process(row))
            self.assertFalse(still_same_process(dict(row, start_ticks='-1')))
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}):
                self.assertIn(owned.pid, [item['pid'] for item in conflicting_training_processes(ROOT, 2)])
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,3'}):
                self.assertNotIn(owned.pid, [item['pid'] for item in conflicting_training_processes(ROOT, 2)])
            subprocess.run([sys.executable, str(ROOT / 'scripts/stop_training.py'), '--run-dir', str(run), '--stop'], check=True, capture_output=True, timeout=15)
            owned.wait(timeout=3)
            self.assertIsNone(unrelated.poll())
        finally:
            for process in (owned, unrelated):
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=3)
            shutil.rmtree(run)


if __name__ == '__main__':
    unittest.main()
