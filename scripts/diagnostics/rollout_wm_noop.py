#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from interactive.oc_storm_adapter import build_oc_storm_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out an oc-storm/STORM WM with noop actions outside the web UI.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wm-name", default="wm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--wm-sample-mode", choices=["probs", "mode", "random_sample"], default="probs")
    parser.add_argument("--wm-respect-terminal", dest="wm_respect_terminal", action="store_true", default=True)
    parser.add_argument("--wm-ignore-terminal", dest="wm_respect_terminal", action="store_false")
    parser.add_argument("--wm-disable-kv-cache", action="store_true")
    parser.add_argument("--wm-kv-cache-dtype", choices=["fp32", "amp"], default="fp32")
    parser.add_argument("--out-dir", default="debug_outputs/wm_noop_rollout")
    parser.add_argument("--gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    session = build_oc_storm_session(
        config_path=Path(args.config).expanduser().resolve(),
        env_name=args.env_name,
        seed=args.seed,
        checkpoint_args=[str(Path(args.checkpoint).expanduser().resolve())],
        wm_name_args=[args.wm_name],
        wm_sample_mode=args.wm_sample_mode,
        wm_respect_terminal=args.wm_respect_terminal,
        wm_disable_kv_cache=args.wm_disable_kv_cache,
        wm_kv_cache_dtype=args.wm_kv_cache_dtype,
    )
    try:
        session.current_backend_index = 1
        session.current_slot = session.wm_slots[0]
        session.reset()

        frames: list[np.ndarray] = []
        infos = []
        initial = np.asarray(session.current_obs, dtype=np.uint8)
        frames.append(initial)
        Image.fromarray(initial).save(frames_dir / "frame_0000.png")

        for step in range(1, args.steps + 1):
            result = session.step(0)
            frame = np.asarray(result.obs, dtype=np.uint8)
            frames.append(frame)
            infos.append(result.info)
            Image.fromarray(frame).save(frames_dir / f"frame_{step:04d}.png")
            if result.done or result.trunc:
                break

        arr = np.stack(frames, axis=0).astype(np.float32)
        diffs = np.abs(arr[1:] - arr[:-1])
        frame_mean_abs_diff = diffs.mean(axis=(1, 2, 3)) if len(frames) > 1 else np.asarray([], dtype=np.float32)
        frame_max_abs_diff = diffs.max(axis=(1, 2, 3)) if len(frames) > 1 else np.asarray([], dtype=np.float32)
        metrics = {
            "config": str(Path(args.config).expanduser()),
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "env_name": args.env_name,
            "seed": args.seed,
            "wm_sample_mode": args.wm_sample_mode,
            "wm_respect_terminal": args.wm_respect_terminal,
            "wm_disable_kv_cache": args.wm_disable_kv_cache,
            "wm_kv_cache_dtype": args.wm_kv_cache_dtype,
            "num_frames": len(frames),
            "num_steps": max(0, len(frames) - 1),
            "mean_abs_diff_mean": float(frame_mean_abs_diff.mean()) if frame_mean_abs_diff.size else 0.0,
            "mean_abs_diff_p50": float(np.percentile(frame_mean_abs_diff, 50)) if frame_mean_abs_diff.size else 0.0,
            "mean_abs_diff_p95": float(np.percentile(frame_mean_abs_diff, 95)) if frame_mean_abs_diff.size else 0.0,
            "max_abs_diff_max": float(frame_max_abs_diff.max()) if frame_max_abs_diff.size else 0.0,
            "done": bool(infos[-1].get("done", False)) if infos else False,
            "trunc": bool(infos[-1].get("trunc", False)) if infos else False,
            "terminal": bool(infos[-1].get("terminal", False)) if infos else False,
            "terminal_predicted_count": sum(
                1 for info in infos if bool(info.get("terminal_predicted", info.get("terminal", False)))
            ),
            "trunc_predicted_count": sum(
                1 for info in infos if bool(info.get("trunc_predicted", info.get("trunc", False)))
            ),
            "terminal_ignored_count": sum(1 for info in infos if bool(info.get("terminal_ignored", False))),
            "last_info": infos[-1] if infos else {},
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.save(out_dir / "frame_mean_abs_diff.npy", frame_mean_abs_diff)

        if args.gif:
            images = [Image.fromarray(frame) for frame in frames]
            images[0].save(
                out_dir / "rollout.gif",
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0,
            )

        print(json.dumps(metrics, indent=2))
        print(f"Wrote frames to {frames_dir}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
