"""Task isolation, complete-state selection, and launcher resume behavior."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from checkpoint_resume import inspect_training_state, latest_training_state
import train_task
from fastwam.datasets.lerobot.press_button_labels import ORDINALS
from training_modes import configure_instruction_mode


EXPECTED_HOLD_LOSS_EXCLUSIONS = {
    'put_back_block': ['Press the button.'],
    'rearrange_blocks': ['Press button.'],
    'blocks_ranking_try': ['Press the button.'],
    'press_button': [f'Press the {button} button for the {ordinal} time.'
                     for button in ('left', 'middle') for ordinal in ORDINALS[:9]],
}


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='fastwam-resume-',dir='/tmp')
        self.root=Path(self.temp.name)
        (self.root/'data').mkdir()
        shutil.copy2(ROOT/'data/task_manifest.json',self.root/'data/task_manifest.json')
        shutil.copytree(ROOT/'configs',self.root/'configs')
        self.env=patch.dict(os.environ,FASTWAM_BUNDLE_ROOT=str(self.root),FASTWAM_RUNTIME_ROOT=str(self.root/'runtime'))
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def state(self,task='put_back_block',run='run1',step=2000,stamp=100,world=4):
        p=self.root/'runs'/task/run/'checkpoints/state'/f'step_{step:06d}'
        (p/'pytorch_model').mkdir(parents=True)
        payload=dict(global_step=step,checkpoint_world_size=world,
                     checkpoint_batch_size=8,checkpoint_gradient_accumulation_steps=64//(world*8))
        marker=p/'trainer_state.json';marker.write_text(json.dumps(payload))
        (p/'scheduler.bin').write_bytes(b'scheduler')
        (p/'pytorch_model/mp_rank_00_model_states.pt').write_bytes(b'model')
        for i in range(world):
            (p/f'random_states_{i}.pkl').write_bytes(b'rng')
            (p/f'pytorch_model/bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt').write_bytes(b'optimizer')
        os.utime(marker,(stamp,stamp))
        return p

    def launch(self,*args):
        output=io.StringIO()
        with patch.object(train_task,'ROOT',self.root),patch.object(sys,'argv',['train_task.py',*args]),contextlib.redirect_stdout(output):
            code=train_task.main()
        return code,output.getvalue()

    def test_latest_complete_state_skips_partial_and_other_task(self):
        old=self.state(step=2000,stamp=100)
        current=self.state(step=4000,stamp=200)
        partial=self.state(step=6000,stamp=300)
        (partial/'pytorch_model/bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt').unlink()
        self.state(task='swap_blocks',step=8000,stamp=400)
        found,skipped=latest_training_state(self.root/'runs/put_back_block')
        self.assertEqual(found.path,current)
        self.assertEqual(len(skipped),1)
        self.assertEqual(found.world_size,4)

    def test_latest_means_recent_run_even_after_an_explicit_restart(self):
        self.state(run='old',step=200000,stamp=100)
        recent=self.state(run='new',step=2000,stamp=200)
        found,_=latest_training_state(self.root/'runs/put_back_block')
        self.assertEqual(found.path,recent)

    def test_missing_rng_or_empty_artifact_is_not_resumable(self):
        p=self.state()
        (p/'scheduler.bin').write_bytes(b'')
        with self.assertRaises(ValueError):inspect_training_state(p)
        (p/'scheduler.bin').write_bytes(b'scheduler')
        (p/'random_states_0.pkl').unlink()
        with self.assertRaises(ValueError):inspect_training_state(p)

    def test_weights_only_does_not_silently_start_from_step_zero(self):
        task=self.root/'runs/put_back_block'
        self.assertEqual(latest_training_state(task),(None,[]))
        weight=task/'run/checkpoints/weights/step_002000.pt'
        weight.parent.mkdir(parents=True);weight.write_bytes(b'weights')
        with self.assertRaisesRegex(ValueError,'weights-only'):
            latest_training_state(task)
        code,output=self.launch('put_back_block','--dry-run')
        self.assertEqual(code,0);self.assertIn('mode=fresh step=0',output)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
            self.launch('put_back_block','--resume','--dry-run')
        self.assertEqual(exc.exception.code,2)

    def test_launcher_restores_batch_and_preserves_existing_launch_in_dry_run(self):
        p=self.state();run=p.parent.parent.parent
        original=run/'launch_config.yaml'
        previous='num_workers: 2\nmodel:\n  mot_checkpoint_mixed_attn: true\n'
        original.write_text(previous)
        code,output=self.launch('put_back_block','--resume','--dry-run')
        self.assertEqual(code,0);self.assertIn('mode=full_state step=2000',output)
        self.assertEqual(original.read_text(),previous)
        record=json.loads(next(run.glob('dry_runs/*/launch_command.json')).read_text())
        self.assertEqual((record['gpus'],record['microbatch'],record['gradient_accumulation_steps']),(4,8,2))
        cfg=OmegaConf.load(next(run.glob('dry_runs/*/launch_config.yaml')))
        self.assertEqual(cfg.resume,str(p));self.assertEqual(cfg.output_dir,str(run))
        self.assertEqual(list(cfg.data.train.exclude_terminal_hold_loss_subtasks),['Press the button.'])
        self.assertEqual((cfg.save_every,cfg.max_checkpoints_to_keep),(2000,2))
        self.assertEqual(cfg.num_workers,2)
        self.assertTrue(cfg.model.mot_checkpoint_mixed_attn)

    def test_all_task_commands_start_fresh_even_when_checkpoint_exists(self):
        tasks=json.loads((self.root/'data/task_manifest.json').read_text())['tasks']
        for task in tasks:
            with self.subTest(task=task):
                saved=self.state(task=task)
                code,output=self.launch(task,'--dry-run')
                self.assertEqual(code,0);self.assertIn('mode=fresh step=0',output)
                record=json.loads(next((self.root/'runs'/task).glob('*/dry_runs/*/launch_command.json')).read_text())
                exclusions=record['exclude_terminal_hold_loss_subtasks']
                self.assertEqual(exclusions,EXPECTED_HOLD_LOSS_EXCLUSIONS.get(task,[]))
                self.assertIsNone(record['resume'])
                self.assertEqual(record['gpus'],8)
                self.assertNotEqual(Path(record['run_dir']),saved.parent.parent.parent)
                self.assertTrue((saved/'trainer_state.json').is_file())

    def test_completed_run_is_skipped_only_with_resume(self):
        self.state(step=200000)
        code,output=self.launch('put_back_block','--resume')
        self.assertEqual(code,0);self.assertIn('[complete]',output)
        for flags in ([],['--fresh']):
            code,output=self.launch('put_back_block',*flags,'--dry-run')
            self.assertEqual(code,0);self.assertIn('mode=fresh step=0',output)

    def test_full_instruction_launch_for_every_task(self):
        tasks=json.loads((self.root/'data/task_manifest.json').read_text())['tasks']
        for task in tasks:
            with self.subTest(task=task):
                code,output=self.launch(task,'--instruction-mode','full','--dry-run','--run-id','full_preview')
                self.assertEqual(code,0)
                run=self.root/'runs'/task/'full_instruction/full_preview'
                cfg=OmegaConf.load(next(run.glob('dry_runs/*/launch_config.yaml')))
                data=cfg.data.train
                self.assertEqual(cfg.instruction_mode,'full')
                self.assertEqual(cfg.output_dir,str(run))
                self.assertEqual((data.training_shuffle_unit,data.num_frames,data.action_video_freq_ratio),('frame',33,4))
                self.assertIn('FullInstructionRobotVideoDataset',data._target_)
                self.assertIn('full_instruction_text_embeds',data.text_embedding_cache_dir)
                self.assertFalse(data.text_embedding_use_true_mask)
                self.assertFalse(any(k.startswith(('gt_subtask_','subtask_')) for k in data))
                self.assertNotIn('exclude_terminal_hold_loss_subtasks',data)
                self.assertNotIn('press_button_label_mode',data)
                self.assertEqual((cfg.max_steps,cfg.save_every,cfg.max_checkpoints_to_keep),(200000,2000,2))
                baseline=OmegaConf.load(self.root/f'configs/portable_{task}.yaml')
                self.assertEqual(cfg.model,baseline.model)
                self.assertIn('mode=fresh step=0',output)

    def test_full_instruction_resume_is_isolated_and_rejects_other_mode(self):
        subtask=self.state(run='subtask',stamp=300)
        full=self.state(task='put_back_block/full_instruction',run='full',stamp=200)
        full_cfg=configure_instruction_mode(OmegaConf.load(self.root/'configs/portable_put_back_block.yaml'),'full')
        OmegaConf.save(full_cfg,full.parent.parent.parent/'launch_config.yaml')
        code,_=self.launch('put_back_block','--instruction-mode','full','--resume','--dry-run')
        self.assertEqual(code,0)
        cfg=OmegaConf.load(next(full.parent.parent.parent.glob('dry_runs/*/launch_config.yaml')))
        self.assertEqual(cfg.resume,str(full))
        code,_=self.launch('put_back_block','--resume','--dry-run')
        self.assertEqual(code,0)
        cfg=OmegaConf.load(next(subtask.parent.parent.parent.glob('dry_runs/*/launch_config.yaml')))
        self.assertEqual(cfg.resume,str(subtask))
        for flags in (['--instruction-mode','full','--resume',str(subtask)],['--resume',str(full)]):
            with self.subTest(flags=flags),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
                self.launch('put_back_block',*flags,'--dry-run')
            self.assertEqual(exc.exception.code,2)

    def test_full_instruction_rejects_subtask_segment_option(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
            self.launch('press_button','--instruction-mode','full','--subtask-segments','raw_annotation','--dry-run')
        self.assertEqual(exc.exception.code,2)

    def test_sequential_full_instruction_covers_all_tasks_and_passes_resume(self):
        with patch.object(sys,'argv',['train_tasks_sequential.py','--instruction-mode','full','--resume']), \
             patch('subprocess.run') as launch,contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT/'scripts/train_tasks_sequential.py'),run_name='__main__')
        self.assertEqual(launch.call_count,12)
        for call in launch.call_args_list:
            command=call.args[0]
            self.assertEqual(command[command.index('--instruction-mode')+1],'full')
            self.assertIn('--resume',command)

    def test_observe_init_resume_requires_matching_inputs_and_annotation_source(self):
        saved = self.state(task='observe_and_pickup')
        previous = saved.parent.parent.parent / 'launch_config.yaml'
        cfg = OmegaConf.load(self.root / 'configs/portable_observe_and_pickup.yaml')
        OmegaConf.save(cfg, previous)
        code, output = self.launch('observe_and_pickup', '--resume', '--dry-run')
        self.assertEqual(code, 0)
        self.assertIn('mode=full_state step=2000', output)
        for path, value in (
            ('model.init_frame_kv_config', None),
            ('data.train._target_', 'fastwam.datasets.lerobot.gt_subtask_video_dataset.GroundTruthSubtaskRobotVideoDataset'),
            ('data.train.gt_subtask_annotation_files', {}),
        ):
            changed = OmegaConf.create(OmegaConf.to_container(cfg))
            OmegaConf.update(changed, path, value, merge=False)
            OmegaConf.save(changed, previous)
            with self.subTest(path=path), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                self.launch('observe_and_pickup', '--resume', '--dry-run')
            self.assertEqual(exc.exception.code, 2)
        self.assertTrue((saved / 'trainer_state.json').is_file())

    def test_explicit_resume_path_selects_requested_state(self):
        saved=self.state(step=2000,stamp=100)
        self.state(step=4000,stamp=200)
        code,output=self.launch('put_back_block','--resume',str(saved),'--dry-run')
        self.assertEqual(code,0);self.assertIn('mode=full_state step=2000',output)

    def test_sequential_launcher_resumes_only_with_flag(self):
        for flags in ([],['--resume'],['--fresh']):
            with self.subTest(flags=flags), \
                 patch.object(sys,'argv',['train_tasks_sequential.py','--tasks','put_back_block','press_button',*flags]), \
                 patch('subprocess.run') as launch,contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT/'scripts/train_tasks_sequential.py'),run_name='__main__')
            self.assertEqual(launch.call_count,2)
            for call in launch.call_args_list:
                command=call.args[0]
                self.assertEqual('--resume' in command,'--resume' in flags)
                self.assertEqual('--fresh' in command,'--fresh' in flags)

    def test_incompatible_resume_gpu_count_is_rejected(self):
        self.state(world=4)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
            self.launch('put_back_block','--resume','--gpus','8','--dry-run')
        self.assertEqual(exc.exception.code,2)

    def test_active_run_is_not_relaunched_or_overwritten_even_on_disjoint_gpus(self):
        p=self.state();run=p.parent.parent.parent
        original=run/'launch_config.yaml';original.write_text('num_workers: 2\n')
        with patch('torch.cuda.device_count',return_value=4), \
             patch('torch.cuda.is_bf16_supported',return_value=True), \
             patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(uuid='fake')), \
             patch.object(train_task.subprocess,'call',return_value=0), \
             patch.object(train_task.subprocess,'Popen') as launch, \
             patch.object(train_task,'find_training_processes',return_value=[{'run_dir':str(run)}]), \
             patch.object(train_task,'conflicting_training_processes',return_value=[]), \
             contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
            self.launch('put_back_block','--resume')
        self.assertEqual(exc.exception.code,2)
        launch.assert_not_called()
        self.assertEqual(original.read_text(),'num_workers: 2\n')


if __name__=='__main__':unittest.main()
