from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMON_SRC = ROOT / "third_party" / "rl-in-pixel-env" / "src"
if COMMON_SRC.is_dir():
    sys.path.insert(0, str(COMMON_SRC))

from rl_in_pixel_env import run_experiment
from rl_in_pixel_env.api import ExperimentConfig


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "t", "on"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OC-STORM adapter for the shared rl-in-pixel-env runner.")
    parser.add_argument("--backend", choices=["real", "wm"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--env-name", default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/atari_visual.py")
    parser.add_argument("--wm-checkpoint", default="")
    parser.add_argument("--bootstrap-dataset", default="")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--ac-updates", type=int, default=20000)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wm-horizon", type=int, default=512)
    parser.add_argument("--wm-sample-mode", choices=["probs", "mode", "random_sample"], default="probs")
    parser.add_argument("--wm-disable-kv-cache", type=str2bool, default=True)
    parser.add_argument("--wm-respect-terminal", type=str2bool, default=True)
    parser.add_argument("--wm-initial-source", choices=["real", "prior", "dataset"], default="real")
    parser.add_argument("--wm-reward-quantize-threshold", type=float, default=0.5)
    parser.add_argument("--backup-every", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--wm-video-every", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--checkpoint-keep", type=int, default=2)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--eval-real-every", type=int, default=2000)
    parser.add_argument("--eval-real-video-every", type=int, default=10000)
    parser.add_argument("--eval-real-eps", type=int, default=5)
    parser.add_argument("--video-max-frames", type=int, default=512)
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--lstm-dim", type=int, default=512)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 32, 64, 64])
    parser.add_argument("--down", type=int, nargs="+", default=[1, 1, 1, 1])
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--weight-value-loss", type=float, default=1.0)
    parser.add_argument("--weight-entropy-loss", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=100.0)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "rl-in-pixel-env-storm"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "ssl-lab"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-enabled", type=str2bool, default=str2bool(os.environ.get("WANDB_ENABLED", "1")))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = ExperimentConfig(
        framework="torch",
        backend=args.backend,
        run_name=args.run_name,
        env_name=args.env_name,
        seed=args.seed,
        device=args.device,
        steps=args.steps,
        ac_updates=args.ac_updates,
        envs=args.envs,
        wm_checkpoint=args.wm_checkpoint,
        wm_horizon=args.wm_horizon,
        wm_respect_terminal=args.wm_respect_terminal,
        wm_reward_quantize_threshold=args.wm_reward_quantize_threshold,
        backup_every=args.backup_every,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_keep=args.checkpoint_keep,
        resume=args.resume,
        eval_real_every=args.eval_real_every,
        eval_real_eps=args.eval_real_eps,
        eval_real_video_every=args.eval_real_video_every,
        video_every=args.wm_video_every,
        video_fps=args.video_fps,
        video_max_frames=args.video_max_frames,
        lstm_dim=args.lstm_dim,
        channels=tuple(args.channels),
        down=tuple(args.down),
        gamma=args.gamma,
        lambda_=args.lambda_,
        weight_value_loss=args.weight_value_loss,
        weight_entropy_loss=args.weight_entropy_loss,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_enabled=args.wandb_enabled,
        adapter_extra={
            "config_path": args.config,
            "sample_mode": args.wm_sample_mode,
            "disable_kv_cache": args.wm_disable_kv_cache,
            "wm_initial_source": args.wm_initial_source,
            "bootstrap_dataset": args.bootstrap_dataset or None,
        },
    )
    return run_experiment(cfg, adapter_spec="pixel_rl.adapter")


if __name__ == "__main__":
    main()
