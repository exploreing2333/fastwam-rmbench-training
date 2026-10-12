import sys
import os
import re
import subprocess
import json

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
curobo_src = os.path.abspath("./envs/curobo/src")
if os.path.isdir(curobo_src) and curobo_src not in sys.path:
    sys.path.append(curobo_src)
extra_site_packages = os.environ.get(
    "RMBENCH_EXTRA_SITE_PACKAGES",
    "/home/cenzl/conda_env/RMBench/lib/python3.10/site-packages",
)
if os.path.isdir(extra_site_packages) and extra_site_packages not in sys.path:
    sys.path.append(extra_site_packages)
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    eval_seed_list = None
    eval_seed_list_file = usr_args.get("eval_seed_list_file")
    if eval_seed_list_file not in (None, ""):
        seed_list_path = Path(str(eval_seed_list_file)).resolve()
        with seed_list_path.open("r", encoding="utf-8") as stream:
            seed_payload = json.load(stream)
        if isinstance(seed_payload, dict):
            seed_payload = seed_payload.get("seeds")
        if not isinstance(seed_payload, list):
            raise ValueError(
                "`eval_seed_list_file` must contain a JSON list or an object "
                "with a `seeds` list."
            )
        eval_seed_list = [int(value) for value in seed_payload]
        if len(eval_seed_list) != eval_num_episodes:
            raise ValueError(
                "Explicit eval seed count does not match `eval_num_episodes`: "
                f"{len(eval_seed_list)} != {eval_num_episodes}"
            )
        if len(set(eval_seed_list)) != len(eval_seed_list):
            raise ValueError("Explicit eval seed list contains duplicates.")
    eval_output_dir = usr_args.get("eval_output_dir")
    resume_eval = bool(usr_args.get("resume_eval", False))
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    for key, value in usr_args.items():
        if key in args:
            args[key] = value
    args["eval_video_camera"] = usr_args.get(
        "eval_video_camera",
        args.get("eval_video_camera", "third_view"),
    )
    args["eval_video_fps"] = int(
        usr_args.get("eval_video_fps", args.get("eval_video_fps", 30))
    )
    args["debug_expert_errors"] = bool(
        usr_args.get("debug_expert_errors", args.get("debug_expert_errors", False))
    )
    args["max_expert_seed_failures"] = int(
        usr_args.get(
            "max_expert_seed_failures",
            args.get("max_expert_seed_failures", 1000),
        )
    )
    args["explicit_seed_precheck_retries"] = int(
        usr_args.get(
            "explicit_seed_precheck_retries",
            args.get("explicit_seed_precheck_retries", 3),
        )
    )
    if args["explicit_seed_precheck_retries"] < 0:
        raise ValueError("`explicit_seed_precheck_retries` must be >= 0")

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = save_dir / "eval_log.txt"
    args["log_file"] = str(log_file)
    args["resume_eval"] = resume_eval
    if not (resume_eval and log_file.is_file()):
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Eval log for {task_name} | {policy_name} | {task_config} | {ckpt_setting}\n")
            f.write(f"Timestamp: {current_time}\n\n")

    if args["eval_video_log"]:
        video_save_dir = save_dir
        eval_video_camera = str(args.get("eval_video_camera", "third_view"))
        if eval_video_camera == "third_view":
            video_size = "320x240"
        elif eval_video_camera == "head_camera":
            camera_config = get_camera_config(args["camera"]["head_camera_type"])
            video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        elif eval_video_camera in {"left_camera", "right_camera"}:
            camera_config = get_camera_config(args["camera"]["wrist_camera_type"])
            video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        else:
            raise ValueError(
                f"Unsupported eval_video_camera={eval_video_camera!r}; "
                "expected third_view, head_camera, left_camera, or right_camera."
            )
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = int(usr_args.get("start_seed", 100000 * (1 + seed)))
    suc_nums = []
    test_num = eval_num_episodes
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num, task_total_reward = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   explicit_seed_list=eval_seed_list)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, "_result.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")

        # Success Rate
        success_rates = (np.asarray(suc_nums, dtype=float) / float(test_num)).reshape(-1)
        for sr in success_rates:
            file.write(f"Success Rate: {sr}\n")

        file.write("\n")  # 分隔一下

        # Reward
        rewards = task_total_reward / test_num
        if np.isscalar(rewards):
            file.write(f"Reward: {float(rewards)}\n")
        else:
            rewards = np.asarray(rewards, dtype=float).reshape(-1)
            for r in rewards:
                file.write(f"Reward: {r}\n")

    print(f"Data has been saved to {file_path}")
    # return task_reward

def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                explicit_seed_list=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    if explicit_seed_list is not None:
        explicit_seed_list = [int(value) for value in explicit_seed_list]
        if len(explicit_seed_list) != test_num:
            raise ValueError(
                f"Explicit seed count {len(explicit_seed_list)} != test_num {test_num}"
            )
        print(
            "[EVAL_EXPLICIT_SEEDS] " + ",".join(map(str, explicit_seed_list)),
            flush=True,
        )
    task_total_reward = 0.0
    if bool(args.get("resume_eval", False)):
        log_file = args.get("log_file")
        completed = []
        if log_file is not None and os.path.isfile(log_file):
            pattern = re.compile(
                r"episode_id=(\d+), seed=(\d+), result=(Success|Fail)"
                r"(?:, max_reward=([-+0-9.eE]+))?"
            )
            with open(log_file, "r", encoding="utf-8") as stream:
                for line in stream:
                    match = pattern.search(line)
                    if match is not None:
                        completed.append(match)
        if len(completed) > test_num:
            raise ValueError(
                f"Resume log has {len(completed)} episodes, expected at most {test_num}."
            )
        if completed:
            now_id = len(completed)
            succ_seed = len(completed)
            suc_test_seed_list = [int(match.group(2)) for match in completed]
            TASK_ENV.suc = sum(match.group(3) == "Success" for match in completed)
            TASK_ENV.test_num = len(completed)
            task_total_reward = sum(
                float(match.group(4)) if match.group(4) is not None else 0.0
                for match in completed
            )
            if explicit_seed_list is not None:
                completed_seeds = [int(match.group(2)) for match in completed]
                expected_prefix = explicit_seed_list[:len(completed_seeds)]
                if completed_seeds != expected_prefix:
                    raise ValueError(
                        "Resume log seed prefix differs from explicit seed list: "
                        f"{completed_seeds} != {expected_prefix}"
                    )
            now_seed = int(completed[-1].group(2)) + 1
            print(
                f"[EVAL_RESUME] episodes={len(completed)}/{test_num} "
                f"successes={TASK_ENV.suc} next_seed={now_seed}",
                flush=True,
            )
    clear_cache_freq = args["clear_cache_freq"]
    skip_get_obs_within_replan = bool(args.get("skip_get_obs_within_replan", False))

    args["eval_mode"] = True
    expert_failures = 0
    max_expert_seed_failures = int(args.get("max_expert_seed_failures", 1000))
    debug_expert_errors = bool(args.get("debug_expert_errors", False))
    explicit_seed_precheck_retries = int(
        args.get("explicit_seed_precheck_retries", 3)
    )

    while succ_seed < test_num:
        if explicit_seed_list is not None:
            now_seed = explicit_seed_list[succ_seed]
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            explicit_attempt = 0
            retry_nonexplicit_seed = False
            while True:
                try:
                    TASK_ENV.setup_demo(
                        now_ep_num=now_id, seed=now_seed, is_test=True, **args
                    )
                    episode_info = TASK_ENV.play_once()
                    TASK_ENV.close_env()
                    break
                except Exception as e:
                    expert_failures += 1
                    if debug_expert_errors:
                        stack_trace = traceback.format_exc()
                        print(" -------------")
                        print("Error: ", e)
                        print(stack_trace)
                        print(" -------------")
                    TASK_ENV.close_env()
                    if (
                        explicit_seed_list is not None
                        and explicit_attempt < explicit_seed_precheck_retries
                    ):
                        explicit_attempt += 1
                        print(
                            f"[EVAL_EXPLICIT_SEED_PRECHECK_RETRY] seed={now_seed} "
                            f"retry={explicit_attempt}/{explicit_seed_precheck_retries} "
                            f"error={type(e).__name__}: {e}",
                            flush=True,
                        )
                        continue
                    if explicit_seed_list is not None:
                        raise RuntimeError(
                            f"Explicit eval seed {now_seed} failed expert precheck "
                            f"after {explicit_attempt + 1} attempts"
                        ) from e
                    now_seed += 1
                    args["render_freq"] = render_freq
                    print("error occurs !")
                    if expert_failures >= max_expert_seed_failures:
                        raise RuntimeError(
                            f"Expert seed precheck failed {expert_failures} consecutive times "
                            f"for task={task_name}; last_seed={now_seed - 1}"
                        ) from e
                    retry_nonexplicit_seed = True
                    break
            if retry_nonexplicit_seed:
                continue

        expert_seed_valid = bool(
            (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success())
        )
        if explicit_seed_list is not None and not expert_seed_valid:
            print(
                f"[EVAL_EXPLICIT_SEED_PRECHECK_MISMATCH] seed={now_seed} "
                "canonical panel accepted it; this worker's expert planner did not",
                flush=True,
            )
        if explicit_seed_list is not None or expert_seed_valid:
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    str(int(args.get("eval_video_fps", 30))),
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            need_obs = True
            if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                need_obs = bool(model.should_request_observation())
            if need_obs and hasattr(model, "capture_observation"):
                observation = model.capture_observation(TASK_ENV)
            else:
                observation = TASK_ENV.get_obs() if need_obs else None
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        if hasattr(model, "finish_episode"):
            model.finish_episode()
        task_total_reward += TASK_ENV.max_reward
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
        if hasattr(model, "finish_episode_artifacts"):
            model.finish_episode_artifacts(
                TASK_ENV,
                episode=TASK_ENV.test_num,
                task_name=task_name,
                instruction_type=str(instruction_type),
            )

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m", " | max reward:", TASK_ENV.max_reward)
            result_str = "Success"
        else:
            print("\033[91mFail!\033[0m", " | max reward:", TASK_ENV.max_reward)
            result_str = "Fail"

        log_file = args.get("log_file", None)
        if log_file is not None:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"episode_id={now_id}, "
                        f"seed={now_seed}, "
                        f"result={result_str}, "
                        f"max_reward={float(TASK_ENV.max_reward)}\n"
                    )
            except Exception as e:
                print(f"[Log Warning] Failed to write log: {e}")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, task_total_reward


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
