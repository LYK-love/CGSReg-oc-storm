from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.distributions import Categorical

from pixel_rl.actor_critic import ActorCritic, ActorCriticConfig
from pixel_rl.envs import BatchedStormWMPixelEnv


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def _import_config(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_random_checkpoint(
    *,
    config_path: Path,
    env_name: str,
    seed: int,
    device: torch.device,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        return output_path
    module = _import_config(config_path)
    _, agent = module.build_offline(env_name=env_name, seed=seed)
    agent = agent.to(device).eval()
    torch.save(agent.state_dict(), output_path)
    return output_path


@torch.no_grad()
def _act(
    model: ActorCritic,
    obs: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    logits, _, next_state = model.predict_act_value(obs, state)
    if deterministic:
        action = logits.argmax(dim=1).long()
    else:
        action = Categorical(logits=logits).sample().long()
    return action, next_state


@torch.no_grad()
def _warmup(env, model: ActorCritic, device: torch.device, steps: int, deterministic: bool) -> None:
    obs, _ = env.reset()
    hx = torch.zeros(env.num_envs, model.lstm_dim, device=device)
    cx = torch.zeros(env.num_envs, model.lstm_dim, device=device)
    for _ in range(int(steps)):
        action, (hx, cx) = _act(model, obs, (hx, cx), deterministic=deterministic)
        obs, _, end, trunc, _ = env.step(action)
        dead = torch.logical_or(end.bool(), trunc.bool())
        if dead.any():
            gate = (~dead).float().unsqueeze(1)
            hx = hx * gate
            cx = cx * gate
    _sync(device)


@torch.no_grad()
def _profile(env, model: ActorCritic, device: torch.device, steps: int, deterministic: bool) -> dict:
    obs, _ = env.reset()
    hx = torch.zeros(env.num_envs, model.lstm_dim, device=device)
    cx = torch.zeros(env.num_envs, model.lstm_dim, device=device)
    policy_sec = 0.0
    env_step_sec = 0.0

    _sync(device)
    total_start = time.perf_counter()
    for _ in range(int(steps)):
        _sync(device)
        t0 = time.perf_counter()
        action, (hx, cx) = _act(model, obs, (hx, cx), deterministic=deterministic)
        _sync(device)
        policy_sec += time.perf_counter() - t0

        _sync(device)
        t0 = time.perf_counter()
        obs, _, end, trunc, _ = env.step(action)
        _sync(device)
        env_step_sec += time.perf_counter() - t0

        dead = torch.logical_or(end.bool(), trunc.bool())
        if dead.any():
            gate = (~dead).float().unsqueeze(1)
            hx = hx * gate
            cx = cx * gate

    _sync(device)
    total_sec = time.perf_counter() - total_start
    transitions = int(steps) * int(env.num_envs)
    return {
        "total_sec": total_sec,
        "env_step_sec": env_step_sec,
        "policy_forward_sec": policy_sec,
        "env_step_pct": 100.0 * env_step_sec / total_sec,
        "policy_forward_pct": 100.0 * policy_sec / total_sec,
        "step_fps": transitions / env_step_sec,
        "rollout_fps": transitions / total_sec,
        "ms_per_env_step_call": 1000.0 * env_step_sec / max(1, int(steps)),
        "ms_per_transition_in_env_step": 1000.0 * env_step_sec / max(1, transitions),
        "steps": int(steps),
        "transitions": transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile random-init STORM WM env.step and pixel policy forward.")
    parser.add_argument("--config-path", type=Path, default=Path("configs/atari_visual.py"))
    parser.add_argument("--env-id", default="PongNoFrameskip-v4")
    parser.add_argument("--num-envs", type=int, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--wm-horizon", type=int, default=512)
    parser.add_argument("--wm-reward-quantize-threshold", type=float, default=0.5)
    parser.add_argument("--sample-mode", default="probs")
    parser.add_argument("--deterministic-policy", action="store_true")
    parser.add_argument("--size-config", default="base")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--random-checkpoint-dir", type=Path, default=Path("runs/profile_random_ckpts"))
    parser.add_argument("--jsonl-out", type=Path, default=None)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = _device(args.device)
    config_path = args.config_path.expanduser().resolve()
    os.environ["STORM_SIZE_CONFIG"] = args.size_config

    checkpoint = args.checkpoint
    checkpoint_status = "provided"
    if checkpoint is None:
        checkpoint_status = "random_init"
        checkpoint = args.random_checkpoint_dir / f"storm_{args.size_config}_{args.env_id.replace('/', '_')}_random.pth"
        checkpoint = _make_random_checkpoint(
            config_path=config_path,
            env_name=args.env_id,
            seed=args.seed,
            device=device,
            output_path=checkpoint.expanduser().resolve(),
        )
    checkpoint = checkpoint.expanduser().resolve()

    rows = []
    for num_envs in args.num_envs:
        env = BatchedStormWMPixelEnv(
            config_path=config_path,
            env_name=args.env_id,
            seed=args.seed,
            checkpoint=checkpoint,
            horizon=args.wm_horizon,
            num_envs=int(num_envs),
            device=device,
            reward_threshold=args.wm_reward_quantize_threshold,
            sample_mode=args.sample_mode,
            disable_kv_cache=True,
            respect_terminal=False,
            bootstrap_dataset=None,
        )
        obs, _ = env.reset()
        _, channels, height, width = obs.shape
        if height != width:
            raise ValueError(f"ActorCritic expects square image observations, got {height}x{width}.")
        model = ActorCritic(
            ActorCriticConfig(
                img_channels=int(channels),
                img_size=int(height),
                num_actions=int(env.num_actions),
            )
        ).to(device).eval()
        try:
            _warmup(env, model, device, args.warmup_steps, args.deterministic_policy)
            result = _profile(env, model, device, args.steps, args.deterministic_policy)
        finally:
            env.close()

        result.update(
            {
                "project": "STORM",
                "backend": "WM",
                "checkpoint": str(checkpoint),
                "checkpoint_label": f"{args.size_config}-{checkpoint_status}",
                "num_envs": int(num_envs),
                "device": str(device),
                "warmup_steps": int(args.warmup_steps),
                "wm_horizon": int(args.wm_horizon),
                "deterministic_policy": bool(args.deterministic_policy),
                "size_config": args.size_config,
                "config_path": str(config_path),
            }
        )
        rows.append(result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_out.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
