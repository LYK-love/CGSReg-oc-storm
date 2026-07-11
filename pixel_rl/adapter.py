from __future__ import annotations

from .envs import make_real_pixel_envs, make_wm_pixel_envs


def make_torch_real_env(ctx):
    return make_real_pixel_envs(
        env_name=ctx.env_name,
        seed=ctx.seed,
        num_envs=ctx.num_envs,
        device=ctx.device,
    )


def make_torch_wm_env(ctx):
    return make_wm_pixel_envs(
        config_path=ctx.extra.get("config_path", "configs/atari_visual.py"),
        env_name=ctx.env_name,
        seed=ctx.seed,
        checkpoint=ctx.wm_checkpoint,
        horizon=ctx.wm_horizon,
        num_envs=ctx.num_envs,
        device=ctx.device,
        reward_threshold=ctx.wm_reward_quantize_threshold,
        sample_mode=ctx.extra.get("sample_mode", "probs"),
        disable_kv_cache=ctx.extra.get("disable_kv_cache", True),
        respect_terminal=ctx.wm_respect_terminal,
        initial_source=ctx.extra.get("wm_initial_source", "real"),
        bootstrap_dataset=ctx.extra.get("bootstrap_dataset"),
    )
