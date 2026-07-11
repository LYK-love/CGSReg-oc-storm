from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (
    ROOT / "wm_play" / "src",
    ROOT / "src" / "wm_play" / "src",
    ROOT.parent / "wm-play-common" / "src",
):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from wm_play.api import PixelPolicy, PolicyAction


WM_EVAL_EVAL_SRC = Path.home() / "projects" / "wm-evaluation" / "scripts" / "eval"
if WM_EVAL_EVAL_SRC.is_dir():
    sys.path.insert(0, str(WM_EVAL_EVAL_SRC))

from sb3_atari_policy import load_sb3_atari_policy  # noqa: E402


FORMAT = "wm_eval_plain_sb3_atari_policy_v1"


def is_sb3_atari_policy_checkpoint(path: str | Path) -> bool:
    path = Path(path).expanduser()
    if not path.is_file() or path.suffix != ".pt":
        return False
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return isinstance(data, dict) and data.get("format") == FORMAT and "state_dict" in data


class SB3AtariPixelPolicy(PixelPolicy):
    def __init__(self, name: str, checkpoint: str | Path, *, device: torch.device | str, deterministic: bool = True):
        self.name = name
        self.policy = load_sb3_atari_policy(checkpoint, device=str(device), deterministic=deterministic)

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def act(self, obs: Any) -> PolicyAction:
        action, value, _ = self.policy.act(_extract_image(obs))
        return PolicyAction(
            action=int(action.reshape(-1)[0].detach().cpu().item()),
            info={
                "source": "sb3_atari_policy",
                "policy_name": self.name,
                "value": float(value.reshape(-1)[0].detach().cpu().item()),
            },
        )


def _extract_image(obs: Any) -> Any:
    if isinstance(obs, dict):
        if "image" in obs:
            return obs["image"]
        for key, value in obs.items():
            if str(key).endswith("image") or getattr(key, "name", None) == "image":
                return value
    return obs


def load_sb3_pixel_policy(
    checkpoint: str | Path,
    *,
    name: str,
    device: torch.device | str,
    deterministic: bool = True,
) -> SB3AtariPixelPolicy:
    return SB3AtariPixelPolicy(name, checkpoint, device=device, deterministic=deterministic)
