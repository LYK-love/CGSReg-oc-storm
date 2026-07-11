from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _find_episode_files(split_dir: Path) -> list[Path]:
    return sorted(
        [p for p in split_dir.rglob("*.pt") if p.name != "info.pt"],
        key=lambda p: int(p.stem),
    )


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, weights_only=False, map_location="cpu")


def _obs_to_uint8_chw(obs: torch.Tensor, image_size: int | None) -> torch.Tensor:
    obs = torch.as_tensor(obs)
    if obs.dtype != torch.uint8:
        obs = ((obs.float() + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
    if obs.ndim != 4:
        raise ValueError(f"Expected obs shape (T,C,H,W), got {tuple(obs.shape)}")
    if image_size is not None and tuple(obs.shape[-2:]) != (image_size, image_size):
        obs = F.interpolate(obs.float(), size=(image_size, image_size), mode="bilinear", align_corners=False)
        obs = obs.clamp(0, 255).to(torch.uint8)
    return obs


def _mask_to_uint8(mask: Any, image_size: int | None) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = torch.as_tensor(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape (T,H,W), got {tuple(mask.shape)}")
    mask = (mask > 0).to(torch.uint8)
    if image_size is not None and tuple(mask.shape[-2:]) != (image_size, image_size):
        mask = F.interpolate(mask[:, None].float(), size=(image_size, image_size), mode="nearest")
        mask = mask[:, 0].to(torch.uint8)
    return mask


def _first_present(episode: dict[str, Any], keys: tuple[str, ...]):
    for key in keys:
        if key in episode:
            return episode[key]
    return None


def _convert_episode(
    episode: dict[str, Any],
    *,
    include_masks: bool,
    include_ram: bool,
    image_size: int | None,
) -> dict[str, Any]:
    obs = _obs_to_uint8_chw(episode["obs"], image_size)
    action = torch.as_tensor(_first_present(episode, ("action", "act"))).long()
    reward = torch.as_tensor(_first_present(episode, ("reward", "rew"))).float()
    termination = torch.as_tensor(_first_present(episode, ("termination", "end"))).float()

    out: dict[str, Any] = {
        "obs": obs,
        "action": action,
        "reward": reward,
        "termination": termination,
    }

    if include_masks:
        for key in ("mask1", "mask2", "mask3"):
            value = _mask_to_uint8(episode.get(key), image_size)
            if value is not None:
                out[key] = value

    if episode.get("important_event_indicator") is not None:
        out["important_event_indicator"] = torch.as_tensor(episode["important_event_indicator"]).to(torch.uint8)

    if include_ram:
        info = episode.get("info") or {}
        if "ram" not in info:
            raise KeyError("--include-ram was requested but an episode has no info['ram'] field.")
        out["info"] = {"ram": torch.as_tensor(info["ram"]).to(torch.uint8)}
    else:
        out["info"] = {}

    length = min(v.shape[0] for k, v in out.items() if k != "info" and torch.is_tensor(v))
    for key, value in list(out.items()):
        if key == "info":
            if "ram" in value:
                value["ram"] = value["ram"][:length]
        elif torch.is_tensor(value):
            out[key] = value[:length]
    return out


def _convert_split(
    input_dir: Path,
    output_dir: Path,
    *,
    include_masks: bool,
    include_ram: bool,
    image_size: int | None,
) -> dict[str, Any]:
    episode_files = _find_episode_files(input_dir)
    if not episode_files:
        raise ValueError(f"No DIAMOND episode .pt files found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    lengths: list[int] = []
    for out_id, path in enumerate(episode_files):
        converted = _convert_episode(
            _load(path),
            include_masks=include_masks,
            include_ram=include_ram,
            image_size=image_size,
        )
        out_path = output_dir / f"{out_id}.pt"
        torch.save(converted, out_path)
        lengths.append(int(converted["obs"].shape[0]))

    meta = {
        "num_episodes": len(lengths),
        "num_steps": int(sum(lengths)),
        "lengths": lengths,
        "format": "oc_storm_offline_episode_v1",
        "include_masks": include_masks,
        "include_ram": include_ram,
        "image_size": image_size,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Converted {len(lengths)} episodes / {sum(lengths)} steps: {input_dir} -> {output_dir}")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DIAMOND .pt replay episodes into oc-storm offline episodes.")
    parser.add_argument("--diamond-root", type=Path, required=True, help="DIAMOND dataset root containing train/ and optionally test/.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for oc-storm train/ and eval splits.")
    parser.add_argument("--include-masks", action="store_true", help="Preserve mask1/mask2/mask3.")
    parser.add_argument("--include-ram", "--ram", dest="include_ram", action="store_true", help="Preserve per-step RAM as info['ram'].")
    parser.add_argument("--no-ram", dest="include_ram", action="store_false", help="Do not preserve RAM fields.")
    parser.add_argument("--image-size", type=int, default=64, help="Resize obs and masks to this square size. Use 0 to preserve input size.")
    parser.add_argument("--force", action="store_true", help="Overwrite output root if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diamond_root = args.diamond_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    image_size = None if args.image_size == 0 else int(args.image_size)

    train_in = diamond_root / "train"
    test_in = diamond_root / "test"
    if not train_in.exists():
        raise FileNotFoundError(f"Missing DIAMOND train split: {train_in}")
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --force to overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source": str(diamond_root),
        "format": "oc_storm_offline_dataset_v1",
        "include_masks": bool(args.include_masks),
        "include_ram": bool(args.include_ram),
        "image_size": image_size,
        "splits": {},
    }
    manifest["splits"]["train"] = _convert_split(
        train_in,
        output_root / "train",
        include_masks=args.include_masks,
        include_ram=args.include_ram,
        image_size=image_size,
    )
    if test_in.exists():
        manifest["splits"]["eval"] = _convert_split(
            test_in,
            output_root / "eval",
            include_masks=args.include_masks,
            include_ram=args.include_ram,
            image_size=image_size,
        )

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
