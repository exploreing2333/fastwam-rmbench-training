"""Run the classify_blocks expert once without loading a policy checkpoint."""

from pathlib import Path
import argparse
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs import CONFIGS_PATH
from envs.classify_blocks import classify_blocks


def _load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100000)
    cli_args = parser.parse_args()
    args = _load_yaml("task_config/demo_clean.yml")
    embodiment_name = args["embodiment"][0]
    embodiment_configs = _load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")
    robot_file = embodiment_configs[embodiment_name]["file_path"]
    robot_config = _load_yaml(Path(robot_file) / "config.yml")
    camera_configs = _load_yaml(Path(CONFIGS_PATH) / "_camera_config.yml")
    head_camera = camera_configs[args["camera"]["head_camera_type"]]

    args.update(
        {
            "task_name": "classify_blocks",
            "task_config": "demo_clean",
            "eval_mode": True,
            "eval_video_save_dir": None,
            "eval_video_camera": "head_camera",
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "left_embodiment_config": robot_config,
            "right_embodiment_config": robot_config,
            "dual_arm_embodied": True,
            "head_camera_h": head_camera["h"],
            "head_camera_w": head_camera["w"],
        }
    )

    task = classify_blocks()
    try:
        task.setup_demo(now_ep_num=0, seed=cli_args.seed, is_test=True, **args)
        print(
            "[SMOKE] scene_initialized "
            f"class_counts={[len(blocks) for blocks in task.blocks_by_class]}",
            flush=True,
        )
        for class_index, blocks in enumerate(task.blocks_by_class):
            for block_index, block in enumerate(blocks):
                print(
                    f"[SMOKE] class={class_index} block={block_index} "
                    f"position={block.get_pose().p.tolist()}",
                    flush=True,
                )
        task.play_once()
        for class_index, blocks in enumerate(task.blocks_by_class):
            basket = task.baskets[class_index]
            for block_index, block in enumerate(blocks):
                print(
                    f"[SMOKE] final class={class_index} block={block_index} "
                    f"position={block.get_pose().p.tolist()} "
                    f"in_basket={task._block_in_basket(block, basket)}",
                    flush=True,
                )
        success = bool(task.plan_success and task.check_success())
        print(
            f"[SMOKE] plan_success={task.plan_success} "
            f"task_success={task.check_success()} max_reward={task.max_reward}",
            flush=True,
        )
        if not success:
            raise RuntimeError("classify_blocks expert smoke test failed")
    finally:
        task.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
