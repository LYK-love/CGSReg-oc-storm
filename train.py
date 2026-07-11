import argparse
import cv2
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import colorama
import random
import shutil
import os
import time
import warnings
import json
from pathlib import Path

from utils import tools
import importlib


def train_ratio_scheduling(episode_length, step, total_steps, min_train_ratio, max_train_ratio):
    """
    Train ratio scheduling
    """
    middle_step = step - episode_length / 2
    current_progress = middle_step / total_steps
    current_train_ratio = min_train_ratio + (max_train_ratio - min_train_ratio) * current_progress
    return current_train_ratio


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")


def optional_str2bool(v):
    if v is None:
        return True
    return str2bool(v)


def get_env_ram(env):
    ale = getattr(getattr(env, "unwrapped", env), "ale", None)
    if ale is None or not hasattr(ale, "getRAM"):
        return None
    return np.array(ale.getRAM(), dtype=np.uint8)


def make_like_replay_buffer(buffer, max_length, include_ram):
    kwargs = {
        "action_dim": buffer.action_dim,
        "num_envs": 1,
        "max_length": max_length,
        "warmup_length": 0,
        "store_on_gpu": True,
    }
    if hasattr(buffer, "state_buffer"):
        new_buffer = type(buffer)(
            state_shape=tuple(buffer.state_buffer.shape[1:]),
            obs_shape=tuple(buffer.obs_buffer.shape[1:]),
            **kwargs,
        )
    else:
        new_buffer = type(buffer)(
            obs_shape=tuple(buffer.obs_buffer.shape[1:]),
            **kwargs,
        )
    if include_ram:
        new_buffer.enable_ram()
    return new_buffer


@torch.no_grad()
def collect_eval_replay_buffer(
    *,
    env,
    feature_extractor,
    action_space,
    agent,
    params,
    replay_buffer_template,
    eval_buffer=None,
    episode_id_start=1,
    eval_episodes,
    max_steps,
    include_ram,
    logger=None,
    log_step=None,
):
    if eval_buffer is None:
        eval_buffer = make_like_replay_buffer(replay_buffer_template, max_steps, include_ram)
    if len(eval_buffer) >= max_steps:
        return eval_buffer, episode_id_start
    agent.eval()
    episode_count = episode_id_start
    episode_id_stop = episode_id_start + eval_episodes
    episode_return = 0
    episode_returns = []
    episode_lengths = []
    feature_extractor.reset()
    current_obs, current_info = env.reset()
    context_state = deque(maxlen=params.eval_context_length + 1)
    context_action = deque(maxlen=params.eval_context_length)

    while episode_count < episode_id_stop and len(eval_buffer) < max_steps:
        current_state, _ = feature_extractor.extract_features(current_obs)
        current_ram = get_env_ram(env) if include_ram else None
        context_state.append(current_state)
        if len(context_action) == 0:
            action = np.zeros(action_space.dim, dtype=np.int32)
        else:
            action = agent.sample_policy(context_state, context_action, greedy=True)
        context_action.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        eval_buffer.append(
            current_state,
            action,
            reward,
            terminated or info["life_loss"],
            episode_count,
            info={"ram": current_ram} if include_ram else None,
        )

        episode_return += reward
        current_obs = obs
        current_info = info

        if terminated or truncated:
            episode_length = current_info["episode_frame_number"] // params.frame_skip
            if logger is not None:
                logger.log("eval/episode_return", episode_return, step=log_step)
                logger.log("eval/episode_length", episode_length, step=log_step)
            episode_returns.append(float(episode_return))
            episode_lengths.append(float(episode_length))
            episode_count += 1
            episode_return = 0
            feature_extractor.reset()
            context_state.clear()
            context_action.clear()
            if episode_count < episode_id_stop:
                current_obs, current_info = env.reset()
    if logger is not None and episode_returns:
        returns = np.asarray(episode_returns, dtype=np.float32)
        lengths = np.asarray(episode_lengths, dtype=np.float32)
        logger.log("eval_real/score_mean", float(returns.mean()), step=log_step)
        logger.log("eval_real/score_std", float(returns.std()), step=log_step)
        logger.log("eval_real/score_min", float(returns.min()), step=log_step)
        logger.log("eval_real/score_max", float(returns.max()), step=log_step)
        logger.log("eval_real/length_mean", float(lengths.mean()), step=log_step)
        logger.log("eval_real/episodes", int(len(returns)), step=log_step)
    return eval_buffer, episode_count


def log_world_model_dataset_metrics(agent, replay_buffer, args, logger, prefix):
    for _ in range(args.eval_metrics_batches):
        agent.log_world_model_metrics(
            replay_buffer,
            args.eval_batch_size,
            args.eval_batch_length,
            logger,
            prefix=prefix,
        )


if __name__ == "__main__":
    # ignore warnings
    warnings.filterwarnings("ignore")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # detect if use visualiation, remote servers don't have graphical interface
    if "HKRL_LOCAL_DEVICE" in os.environ:
        opencv_visualization = True
    else:
        opencv_visualization = False

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_name", type=str, required=True)  # please pass "None" if not using dynamic importing
    parser.add_argument(
        "--size-config",
        "--size_config",
        dest="size_config",
        type=str,
        default="base",
        help="Model size preset. Use base for the original oc-storm repository size.",
    )
    parser.add_argument(
        "--ram",
        nargs="?",
        const=True,
        default=False,
        type=optional_str2bool,
        help="Collect Atari RAM at every replay step and store it under info['ram'].",
    )
    parser.add_argument(
        "--no-ram",
        dest="ram",
        action="store_false",
        help="Disable per-step RAM collection.",
    )
    parser.add_argument(
        "--include_ram",
        dest="ram",
        type=str2bool,
        help="Deprecated alias for --ram/--no-ram.",
    )
    parser.add_argument(
        "--save_dataset",
        type=str2bool,
        default=True,
        help="Export the collected replay buffer as an oc-storm offline dataset when training finishes.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="",
        help="Output split directory for exported episodes. Defaults to runs/<run_name>/dataset/train.",
    )
    parser.add_argument(
        "--save_eval_dataset",
        type=str2bool,
        default=True,
        help="Collect periodic online greedy eval rollouts and export them as runs/<run_name>/dataset/eval.",
    )
    parser.add_argument(
        "--eval_dataset_dir",
        type=str,
        default="",
        help="Output split directory for eval episodes. Defaults to runs/<run_name>/dataset/eval.",
    )
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_dataset_steps", type=int, default=100000)
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=0,
        help="Collect online eval episodes every N environment steps. Defaults to config save_every_steps.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_length", type=int, default=32)
    parser.add_argument("--eval_metrics_batches", type=int, default=1)
    parser.add_argument(
        "--max_sample_steps",
        "--max-sample-steps",
        dest="max_sample_steps",
        type=int,
        default=-1,
        help="Override params.max_sample_steps for extended diagnostics.",
    )
    parser.add_argument(
        "--freeze_wm_after_fraction",
        "--freeze-wm-after-fraction",
        dest="freeze_wm_after_fraction",
        type=float,
        default=-1.0,
        help="Freeze world-model updates after this fraction of max_sample_steps; policy updates continue.",
    )
    parser.add_argument(
        "--freeze_wm_after_step",
        "--freeze-wm-after-step",
        dest="freeze_wm_after_step",
        type=int,
        default=-1,
        help="Freeze world-model updates after this environment sample step; overrides the fraction when set.",
    )
    parser.add_argument(
        "--freeze_wm_original_step",
        "--freeze-wm-original-step",
        dest="freeze_wm_original_step",
        type=int,
        default=-1,
        help="Denominator for dyna/freeze_progress; defaults to params.max_sample_steps.",
    )
    args = parser.parse_args()
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    tools.seed_np_torch(seed=args.seed)
    # create log/ckpt folder
    logger = tools.Logger(
        run_name=args.run_name,
        config={
            "entrypoint": "train.py",
            "env_name": args.env_name,
            "seed": args.seed,
            "config_name": args.config_name,
            "size_config": args.size_config,
            "ram": args.ram,
            "save_dataset": args.save_dataset,
            "dataset_dir": args.dataset_dir,
            "save_eval_dataset": args.save_eval_dataset,
            "eval_dataset_dir": args.eval_dataset_dir,
            "eval_episodes": args.eval_episodes,
            "eval_dataset_steps": args.eval_dataset_steps,
            "eval_every_steps": args.eval_every_steps,
            "eval_metrics_batches": args.eval_metrics_batches,
            "eval_real_episodes": args.eval_episodes,
            "eval_real_every_steps": args.eval_every_steps,
            "eval_real_policy": "greedy",
            "eval_real_env": "ALE Pong real env",
            "freeze_wm_original_step": args.freeze_wm_original_step,
        },
    )  # tensorboard writer
    os.makedirs(f"runs/{args.run_name}/ckpt", exist_ok=True)  # ckpt dir

    # load config
    if args.config_name != "None":
        # pass
        os.environ["STORM_SIZE_CONFIG"] = args.size_config
        module_name = f"configs.{args.config_name}"
        config_module = importlib.import_module(module_name)
        build = getattr(config_module, "build")
        Params = getattr(config_module, "Params")
        print(colorama.Fore.RED + f"Using {args.config_name}.py" + colorama.Style.RESET_ALL)
        shutil.copy(f"configs/{args.config_name}.py", f"runs/{args.run_name}/config.py")
    else:
        # print(colorama.Fore.YELLOW + "[WARN] config_path not used, the importing is following the code" + colorama.Style.RESET_ALL)
        ###############################################################################################
        # mannual import, IDE friendly
        from configs.hollow_knight_vector_visual import build, Params

        print(colorama.Fore.RED + "Using hollow_knight_vector_visual.py" + colorama.Style.RESET_ALL)
        shutil.copy("configs/hollow_knight_vector_visual.py", f"runs/{args.run_name}/config.py")
        ###############################################################################################

    # build all components
    params, env, action_space, feature_extractor, replay_buffer, agent = build(env_name=args.env_name, seed=args.seed)
    if args.max_sample_steps > 0:
        params.max_sample_steps = int(args.max_sample_steps)
    if args.ram:
        replay_buffer.enable_ram()
    if args.eval_every_steps <= 0:
        args.eval_every_steps = params.save_every_steps
    freeze_wm_after_step = int(args.freeze_wm_after_step)
    if freeze_wm_after_step < 0 and args.freeze_wm_after_fraction >= 0:
        if not (0.0 < args.freeze_wm_after_fraction < 1.0):
            raise ValueError("--freeze-wm-after-fraction must be in (0, 1).")
        freeze_wm_after_step = int(params.max_sample_steps * args.freeze_wm_after_fraction)
    if freeze_wm_after_step >= 0:
        logger.log("dyna/freeze_wm_after_step", freeze_wm_after_step, step=0)
        logger.log("dyna/freeze_wm_after_fraction", args.freeze_wm_after_fraction, step=0)
    freeze_wm_original_step = int(args.freeze_wm_original_step)
    if freeze_wm_original_step <= 0:
        freeze_wm_original_step = int(params.max_sample_steps)
    eval_dataset_dir = args.eval_dataset_dir or f"runs/{args.run_name}/dataset/eval"
    eval_enabled = args.save_eval_dataset or args.eval_metrics_batches > 0
    eval_state = {"buffer": None, "last_step": None, "next_episode": 1}

    def run_online_eval(step_label):
        if not eval_enabled:
            return
        print(colorama.Fore.GREEN + f"Collecting online eval episodes at step {step_label}" + colorama.Style.RESET_ALL)
        eval_buffer = eval_state["buffer"] if args.save_eval_dataset else None
        before = len(eval_buffer) if eval_buffer is not None else 0
        before_next_episode = eval_state["next_episode"]
        eval_buffer, eval_state["next_episode"] = collect_eval_replay_buffer(
            env=env,
            feature_extractor=feature_extractor,
            action_space=action_space,
            agent=agent,
            params=params,
            replay_buffer_template=replay_buffer,
            eval_buffer=eval_buffer,
            episode_id_start=eval_state["next_episode"],
            eval_episodes=args.eval_episodes,
            max_steps=args.eval_dataset_steps,
            include_ram=args.ram,
            logger=logger,
            log_step=step_label,
        )
        if not args.save_eval_dataset and eval_state["next_episode"] != before_next_episode + args.eval_episodes:
            raise RuntimeError(
                f"Expected {args.eval_episodes} eval_real episodes, "
                f"got {eval_state['next_episode'] - before_next_episode}."
            )
        eval_state["buffer"] = eval_buffer
        eval_wm_frozen = int(freeze_wm_after_step >= 0 and step_label >= freeze_wm_after_step)
        logger.log("eval_real/collection_step", step_label, step=step_label)
        logger.log("eval_real/freeze_progress", step_label / float(freeze_wm_original_step), step=step_label)
        logger.log("eval_real/wm_frozen", eval_wm_frozen, step=step_label)
        logger.log("eval_real/freeze_wm_after_step", freeze_wm_after_step, step=step_label)
        logger.log("eval/collection_step", step_label, step=step_label)
        logger.log("eval/replay_buffer_length", len(eval_state["buffer"]), step=step_label)
        if len(eval_state["buffer"]) == before:
            print(colorama.Fore.YELLOW + "Eval replay buffer is full; skipped new eval collection." + colorama.Style.RESET_ALL)
        if args.eval_metrics_batches > 0:
            log_world_model_dataset_metrics(agent, replay_buffer, args, logger, prefix="train_dataset")
            log_world_model_dataset_metrics(agent, eval_state["buffer"], args, logger, prefix="eval")
        eval_state["last_step"] = step_label

    # train >>>
    # reset envs and variables
    episode_count = 1
    episode_return = 0
    current_obs, current_info = env.reset()
    context_state = deque(maxlen=params.eval_context_length + 1)  # +1 for the current_state
    context_action = deque(maxlen=params.eval_context_length)
    next_eval_step = args.eval_every_steps

    # sample and train
    for current_sample_step in tqdm(range(params.max_sample_steps)):
        current_state, visualization_obs = feature_extractor.extract_features(current_obs)
        current_ram = get_env_ram(env) if args.ram else None

        # policy part >>>
        if replay_buffer.ready():
            context_state.append(current_state)  # first append the current state
            if len(context_action) == 0:  # First step of the episode, no context
                action = np.zeros(action_space.dim, dtype=np.int32)
            else:
                if random.random() > 0.01:
                    action = agent.sample_policy(context_state, context_action, greedy=False)
                else:
                    action = action_space.sample()
            context_action.append(action)  # finally append the action
        else:  # warmup
            action = action_space.sample()
        # <<< policy part

        obs, reward, terminated, truncated, info = env.step(action)
        replay_buffer.append(
            current_state,
            action,
            reward,
            terminated or info["life_loss"],
            episode_count,
            info={"ram": current_ram} if args.ram else None,
        )

        # visualization >>>
        if opencv_visualization:
            cv2.imshow("visualization_obs", cv2.cvtColor(visualization_obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        # <<< visualization

        # update current_obs, current_info and episode_return
        episode_return += reward
        current_obs = obs
        current_info = info

        # save model for evaluation
        if current_sample_step % params.save_every_steps == 0 and current_sample_step > 0:
            print(
                colorama.Fore.GREEN
                + f"Saving log model at total steps {current_sample_step}"
                + colorama.Style.RESET_ALL
            )
            torch.save(agent.state_dict(), f"runs/{args.run_name}/ckpt/agent_{current_sample_step}.pth")

        # useful option when debugging
        # if terminated or truncated or current_sample_step > 80:
        #     replay_buffer.warmup_length = 0
        if terminated or truncated:
            # clear feature extractor memory
            feature_extractor.reset()
            # clear context for next episode
            context_state.clear()
            context_action.clear()

            # logs >>>
            episode_length = current_info["episode_frame_number"] // params.frame_skip
            print(colorama.Fore.YELLOW + f"\nEpisode {episode_count} done" + colorama.Style.RESET_ALL)
            print("Return: " + colorama.Fore.YELLOW + f"{episode_return}" + colorama.Style.RESET_ALL)
            episode_count += 1

            logger.log("sample/episode_return", episode_return)
            logger.log("sample/episode_length", episode_length)
            if "HollowKnight" in args.env_name:
                logger.log("sample/win_battle", current_info["win_battle"])
                logger.log("sample/health_remains", current_info["health"])

            logger.log("replay_buffer/length", len(replay_buffer))
            # <<< logs

            # reset episode_return for next episode
            episode_return = 0

            if replay_buffer.ready():
                print(colorama.Fore.CYAN + "\nEpisode done, start training" + colorama.Style.RESET_ALL)
                current_train_ratio = train_ratio_scheduling(
                    episode_length,
                    current_sample_step,
                    params.max_sample_steps,
                    params.min_train_ratio,
                    params.max_train_ratio,
                )
                logger.log("train/train_ratio", current_train_ratio)
                training_steps = int(current_train_ratio * episode_length)
                train_world_model = not (
                    freeze_wm_after_step >= 0 and current_sample_step >= freeze_wm_after_step
                )
                logger.log("dyna/wm_frozen", 0 if train_world_model else 1, step=current_sample_step)
                logger.log("dyna/collection_step", current_sample_step, step=current_sample_step)
                logger.log(
                    "dyna/freeze_progress",
                    current_sample_step / float(freeze_wm_original_step),
                    step=current_sample_step,
                )
                training_start_time = time.time()
                for _ in tqdm(range(training_steps)):
                    agent.update(
                        replay_buffer,
                        training_steps,
                        params.batch_size,
                        params.batch_length,
                        params.imagine_batch_size,
                        params.imagine_context_length,
                        params.imagine_batch_length,
                        logger,
                        train_world_model=train_world_model,
                    )
                training_time = time.time() - training_start_time
                if (
                    training_time < 6 and "HollowKnight" in args.env_name
                ):  # Hollow Knight, see the else branch, level 3 boss's episode length can be too short
                    if training_time < 6 + 8 and "HKPrime" in args.env_name:
                        # TODO: in HKPrime, there would be an extra 8sec post-swing if the boss is defeated
                        # here we sleep for 14 seconds despite win or lose, should be optimized later
                        time.sleep(6 + 8 - training_time)
                    else:
                        time.sleep(6 - training_time)
            else:
                print(colorama.Fore.BLUE + "\nBuffer not warmed up, skip training" + colorama.Style.RESET_ALL)
                if "HollowKnight" in args.env_name:
                    time.sleep(6)  # wait for game to load, for Hollow Knight

            if eval_enabled and current_sample_step >= next_eval_step:
                run_online_eval(current_sample_step)
                while next_eval_step <= current_sample_step:
                    next_eval_step += args.eval_every_steps

            # reset envs and variables
            print(colorama.Fore.BLUE + "\nTraining done, reset env" + colorama.Style.RESET_ALL)
            feature_extractor.reset()
            current_obs, current_info = env.reset()
            # save latest model every episode
            print(colorama.Fore.GREEN + f"Saving latest model at step {current_sample_step}" + colorama.Style.RESET_ALL)
            torch.save(agent.state_dict(), f"runs/{args.run_name}/ckpt/latest_agent.pth")

    # <<< train
    if "HollowKnight" in args.env_name:
        env.release_action()
        print(colorama.Fore.RED + "Execution done, release keyboard" + colorama.Style.RESET_ALL)
    eval_meta = None
    if eval_enabled and eval_state["last_step"] != params.max_sample_steps:
        run_online_eval(params.max_sample_steps)
    if eval_state["buffer"] is not None and args.save_eval_dataset:
        print(colorama.Fore.GREEN + f"Exporting online eval dataset to {eval_dataset_dir}" + colorama.Style.RESET_ALL)
        eval_meta = eval_state["buffer"].export_offline_dataset(eval_dataset_dir, include_ram=args.ram)
        eval_path = Path(eval_dataset_dir)
        if eval_path.name == "eval" and not args.save_dataset:
            manifest = {
                "format": "oc_storm_offline_dataset_v1",
                "source": f"online train.py run {args.run_name}",
                "ram": bool(args.ram),
                "eval_collection": {
                    "type": "periodic_online_current_policy",
                    "eval_every_steps": args.eval_every_steps,
                    "eval_episodes_per_collection": args.eval_episodes,
                    "max_eval_steps": args.eval_dataset_steps,
                },
                "splits": {"eval": eval_meta},
            }
            (eval_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.save_dataset:
        dataset_dir = args.dataset_dir or f"runs/{args.run_name}/dataset/train"
        print(colorama.Fore.GREEN + f"Exporting collected dataset to {dataset_dir}" + colorama.Style.RESET_ALL)
        meta = replay_buffer.export_offline_dataset(dataset_dir, include_ram=args.ram)
        dataset_path = Path(dataset_dir)
        if dataset_path.name == "train":
            manifest = {
                "format": "oc_storm_offline_dataset_v1",
                "source": f"online train.py run {args.run_name}",
                "ram": bool(args.ram),
                "splits": {"train": meta},
            }
            if eval_enabled:
                manifest["eval_collection"] = {
                    "type": "periodic_online_current_policy",
                    "eval_every_steps": args.eval_every_steps,
                    "eval_episodes_per_collection": args.eval_episodes,
                    "max_eval_steps": args.eval_dataset_steps,
                }
            if eval_meta is not None:
                manifest["splits"]["eval"] = eval_meta
            (dataset_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            colorama.Fore.GREEN
            + f"Exported {meta['num_episodes']} episodes / {meta['num_steps']} steps"
            + colorama.Style.RESET_ALL
        )
    logger.close()
