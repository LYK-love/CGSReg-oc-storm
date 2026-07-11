
from __future__ import annotations

import copy
import importlib.util
import math
import os
import re
import sys
import warnings
from collections import deque
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
warnings.filterwarnings(
    'ignore',
    message='pkg_resources is deprecated as an API.*',
    category=UserWarning,
)
import pygame
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from wm_play.api import PixelPolicy, PlaySession, PolicyAction, StepResult
from wm_play.status import play_status_lines

ROOT = Path(__file__).resolve().parents[1]
COMMON_SRC = ROOT / 'third_party' / 'rl-in-pixel-env' / 'src'
if COMMON_SRC.is_dir():
    sys.path.insert(0, str(COMMON_SRC))

from diamond_rl_env.actor_critic import ActorCriticConfig
from diamond_rl_env.policy import ActorCriticPolicy, load_actor_critic_policy
from interactive.sb3_atari_policy import is_sb3_atari_policy_checkpoint, load_sb3_pixel_policy


ATARI_ACTION_NAMES = [
    'noop', 'fire', 'up', 'right', 'left', 'down',
    'upright', 'upleft', 'downright', 'downleft',
    'upfire', 'rightfire', 'leftfire', 'downfire',
    'uprightfire', 'upleftfire', 'downrightfire', 'downleftfire',
]
ATARI_KEYMAP = {
    (pygame.K_SPACE,): 1,
    (pygame.K_w,): 2,
    (pygame.K_d,): 3,
    (pygame.K_a,): 4,
    (pygame.K_s,): 5,
    (pygame.K_w, pygame.K_d): 6,
    (pygame.K_w, pygame.K_a): 7,
    (pygame.K_s, pygame.K_d): 8,
    (pygame.K_s, pygame.K_a): 9,
    (pygame.K_w, pygame.K_SPACE): 10,
    (pygame.K_d, pygame.K_SPACE): 11,
    (pygame.K_a, pygame.K_SPACE): 12,
    (pygame.K_s, pygame.K_SPACE): 13,
    (pygame.K_w, pygame.K_d, pygame.K_SPACE): 14,
    (pygame.K_w, pygame.K_a, pygame.K_SPACE): 15,
    (pygame.K_s, pygame.K_d, pygame.K_SPACE): 16,
    (pygame.K_s, pygame.K_a, pygame.K_SPACE): 17,
}


def _import_config_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to import config from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_state_dict(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _load_checkpoint(path: Path, device: torch.device | str = 'cpu') -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _policy_checkpoint_sort_key(path: Path) -> tuple[int, int, float, str]:
    match = re.search(r'(?:update|epoch)_?0*(\d+)', path.stem)
    if match:
        return (1, int(match.group(1)), path.stat().st_mtime, path.name)
    return (0, -1, path.stat().st_mtime, path.name)


def resolve_policy_checkpoint_path(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if root.is_file():
        return root.resolve()
    if not root.exists():
        raise FileNotFoundError(f'Policy checkpoint path does not exist: {root}')
    if not root.is_dir():
        raise ValueError(f'Policy checkpoint path is neither file nor directory: {root}')

    for rel in (
        'latest.pt',
        'pixel_rl_ckpt/latest.pt',
        'policy_ckpt/latest.pt',
        'ckpt/latest.pt',
        'checkpoints/latest.pt',
    ):
        candidate = root / rel
        if candidate.is_file():
            return candidate.resolve()

    candidates: list[Path] = []
    for child in ('pixel_rl_ckpt', 'policy_ckpt', 'ckpt', 'checkpoints'):
        candidate_dir = root / child
        if candidate_dir.is_dir():
            candidates.extend(p for p in candidate_dir.glob('*.pt') if p.is_file())
    if not candidates:
        candidates = [p for p in root.glob('*.pt') if p.is_file()]
    if not candidates:
        candidates = [p for p in root.rglob('*.pt') if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f'No .pt policy checkpoint found under: {root}')
    return max(candidates, key=_policy_checkpoint_sort_key).resolve()


def _variant_from_agent(agent) -> str:
    module = agent.__class__.__module__
    if 'vector_visual' in module:
        return 'vector_visual'
    if 'visual' in module:
        return 'visual'
    if 'vector' in module:
        return 'vector'
    raise ValueError(f'Unsupported agent module: {module}')


def _validate_initial_source(value: str) -> str:
    source = str(value).lower()
    if source not in {'real', 'prior', 'dataset'}:
        raise ValueError(f'Unsupported wm_initial_source={value!r}')
    return source


def _infer_action_names(env) -> list[str]:
    meanings = getattr(env.unwrapped, 'get_action_meanings', None)
    if callable(meanings):
        return [str(x).lower() for x in meanings()]
    action_space = getattr(env, 'action_space', None)
    if action_space is not None and hasattr(action_space, 'n'):
        return [f'a{i}' for i in range(action_space.n)]
    return ['noop']


def _build_keymap(action_names: list[str]) -> dict[tuple[int, ...], int]:
    keymap: dict[tuple[int, ...], int] = {}
    for keys, idx in ATARI_KEYMAP.items():
        if idx < len(ATARI_ACTION_NAMES):
            action_name = ATARI_ACTION_NAMES[idx]
            if action_name in action_names:
                keymap[keys] = action_names.index(action_name)
    return keymap


def _resize_frame(frame: np.ndarray, size: int = 64) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)


def _frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float().to(device)
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor


def _ensure_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, np.ndarray):
        if value.ndim == 3 and value.shape[-1] == 3:
            return _frame_to_tensor(value, device)
        return torch.from_numpy(value).float().to(device=device)
    return torch.as_tensor(value, device=device).float()


class _OfflineVisualFeatureExtractor:
    """Lightweight RGB extractor for offline-trained visual STORM play."""

    def __init__(self, state_resolution=(64, 64), input_channels: int = 3, device: str = 'cuda') -> None:
        self.state_resolution = tuple(state_resolution)
        self.input_channels = int(input_channels)
        self.device = torch.device(device)

    def reset(self):
        return None

    def extract_features(self, frame) -> tuple[torch.Tensor, np.ndarray]:
        resized = _resize_frame(np.asarray(frame), self.state_resolution[0])
        state = _frame_to_tensor(resized, self.device)
        if self.input_channels > 3:
            masks = torch.zeros(
                (self.input_channels - 3, *state.shape[-2:]),
                dtype=state.dtype,
                device=state.device,
            )
            state = torch.cat([state, masks], dim=0)
        return state, resized


class _OfflineVisualBootstrapDataset:
    def __init__(self, root: str | Path, device: torch.device, input_channels: int) -> None:
        root = Path(root).expanduser()
        if root.is_file():
            files = [root]
        else:
            files = sorted(
                p for suffix in ('*.pt', '*.pth', '*.npz')
                for p in root.rglob(suffix)
                if p.name != 'info.pt')
        if not files:
            raise FileNotFoundError(f'No offline bootstrap episodes found under {root}')
        self.files = files
        self.device = device
        self.input_channels = int(input_channels)

    @staticmethod
    def _first_present(data: dict[str, Any], names: tuple[str, ...]):
        for name in names:
            if name in data:
                return data[name]
        return None

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if path.suffix == '.npz':
            data = np.load(path)
            return {key: torch.as_tensor(data[key]) for key in data.files}
        loaded = torch.load(path, map_location='cpu')
        if not isinstance(loaded, dict):
            raise ValueError(f'Offline bootstrap episode must be dict-like: {path}')
        return loaded

    @staticmethod
    def _obs_float01(obs: torch.Tensor) -> torch.Tensor:
        obs = torch.as_tensor(obs)
        if obs.dtype == torch.uint8:
            return obs.float() / 255.0
        obs = obs.float()
        if obs.numel() and obs.min() < 0:
            obs = (obs + 1.0) / 2.0
        return obs.clamp(0.0, 1.0)

    @staticmethod
    def _mask_float(mask: torch.Tensor) -> torch.Tensor:
        mask = torch.as_tensor(mask)
        if mask.ndim == 3:
            mask = mask[:, None]
        return mask.float()

    def _state(self, data: dict[str, Any]) -> torch.Tensor:
        obs = self._first_present(data, ('obs', 'image', 'frames', 'observation'))
        if obs is None:
            raise KeyError('Offline bootstrap episode has no obs/image/frames key.')
        state = self._obs_float01(obs)
        if state.ndim != 4:
            raise ValueError(f'Expected bootstrap obs shape (T,C,H,W), got {tuple(state.shape)}')
        if state.shape[1] < self.input_channels:
            masks = self._first_present(data, ('masks', 'mask'))
            mask_parts = []
            if masks is not None:
                mask_parts.append(self._mask_float(masks))
            else:
                for key in ('mask1', 'mask2', 'mask3'):
                    value = self._first_present(data, (key,))
                    if value is not None:
                        mask_parts.append(self._mask_float(value))
            if mask_parts:
                state = torch.cat([state, torch.cat(mask_parts, dim=1)], dim=1)
        if state.shape[1] < self.input_channels:
            pad = torch.zeros(
                state.shape[0],
                self.input_channels - state.shape[1],
                *state.shape[-2:],
                dtype=state.dtype)
            state = torch.cat([state, pad], dim=1)
        return state[:, :self.input_channels].contiguous()

    def sample(self, length: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        for _ in range(max(8, len(self.files))):
            path = self.files[int(torch.randint(len(self.files), ()).item())]
            data = self._load(path)
            state = self._state(data)
            actions = self._first_present(data, ('action', 'act', 'actions'))
            if actions is None:
                actions = torch.zeros((state.shape[0], 1), dtype=torch.int32)
            actions = torch.as_tensor(actions).long()
            if actions.ndim == 1:
                actions = actions[:, None]
            usable = min(state.shape[0], actions.shape[0])
            if usable >= 2:
                length = min(max(2, int(length)), usable)
                start = int(torch.randint(usable - length + 1, ()).item())
                state_seq = state[start:start + length].to(self.device, non_blocking=True)
                action_seq = actions[start:start + length - 1].to(self.device, dtype=torch.int32, non_blocking=True)
                frame = state_seq[-1, :3].detach().float().cpu().permute(1, 2, 0).numpy()
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                return state_seq, action_seq, frame
        raise ValueError(f'No bootstrap episode in {self.files[0].parent} is long enough.')


def _tensor_to_frame(tensor: Any) -> np.ndarray:
    if tensor is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if isinstance(tensor, np.ndarray):
        frame = tensor
    elif isinstance(tensor, torch.Tensor):
        frame = tensor.detach().float().cpu()
        while frame.ndim > 3:
            frame = frame[0]
        if frame.ndim == 3 and frame.shape[0] >= 3 and frame.shape[-1] != 3:
            frame = frame.permute(1, 2, 0)
        frame = frame.numpy()
    else:
        frame = np.asarray(tensor)
    if frame.ndim != 3:
        frame = np.zeros((64, 64, 3), dtype=np.float32)
    if frame.shape[-1] >= 3:
        frame = frame[..., :3]
    elif frame.shape[0] >= 3:
        frame = np.transpose(frame[:3], (1, 2, 0))
    elif frame.shape[-1] != 3:
        frame = np.repeat(frame[..., :1], 3, axis=-1)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.5:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _fmt_scalar(value: Any) -> str:
    if value is None:
        return '-'
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f'{value:.2f}'


def _action_to_index(action: Any) -> int:
    arr = np.asarray(action).reshape(-1)
    if arr.size == 0:
        return 0
    return int(arr[0])


def _extract_prefixed_state_dict(state_dict: dict[str, Any], module_name: str) -> dict[str, Any]:
    prefix = f'{module_name}.'
    if module_name in state_dict and isinstance(state_dict[module_name], dict):
        return state_dict[module_name]
    return {
        k.split('.', 1)[1]: v
        for k, v in state_dict.items()
        if isinstance(k, str) and k.startswith(prefix)
    }


def _diamond_project_root() -> Path:
    candidates = [
        os.environ.get('DIAMOND_ROOT'),
        Path(__file__).resolve().parents[2] / 'diamond',
        str(Path.home() / 'projects' / 'diamond'),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / 'src' / 'models' / 'actor_critic.py').exists():
            return Path(candidate).resolve()
    raise RuntimeError(
        'Could not find DIAMOND checkout. Set DIAMOND_ROOT to the DIAMOND project root.'
    )


@contextmanager
def _diamond_import_context():
    root = _diamond_project_root()
    src = str(root / 'src')
    old_path = list(sys.path)
    module_prefixes = (
        'coroutines',
        'data',
        'envs',
        'models',
        'utils',
    )
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name in module_prefixes or name.startswith(tuple(f'{p}.' for p in module_prefixes))
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    sys.path.insert(0, src)
    try:
        yield root
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name in module_prefixes or name.startswith(tuple(f'{p}.' for p in module_prefixes)):
                sys.modules.pop(name, None)
        for name in saved_modules:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


class DiamondActorCriticPixelPolicy(PixelPolicy):
    """PixelPolicy wrapper for DIAMOND actor-critic checkpoints."""

    def __init__(self, name: str, actor_critic: torch.nn.Module, device: torch.device) -> None:
        self.name = name
        self.actor_critic = actor_critic
        self.device = device
        self.hx_cx = None

    def reset(self) -> None:
        self.hx_cx = None

    def _obs_tensor(self, obs: Any) -> torch.Tensor:
        frame = _tensor_to_frame(obs)
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float()
        tensor = tensor.to(self.device, non_blocking=True) / 127.5 - 1.0
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def act(self, obs: Any) -> PolicyAction:
        policy_obs = self._obs_tensor(obs)
        logits, value, self.hx_cx = self.actor_critic.predict_act_value(policy_obs, self.hx_cx)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        entropy = dist.entropy() / math.log(2)
        return PolicyAction(
            action=int(action.reshape(-1)[0].item()),
            info={
                'entropy': float(entropy.item()),
                'value': float(value.item()),
                'source': 'diamond_policy',
                'policy_slot_name': self.name,
            },
        )


class RLInPixelEnvPolicy(PixelPolicy):
    """PixelPolicy wrapper for rl-in-pixel-env actor-critic checkpoints."""

    def __init__(self, name: str, policy: ActorCriticPolicy) -> None:
        self.name = name
        self.policy = policy

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def act(self, obs: Any) -> PolicyAction:
        frame = _tensor_to_frame(obs)
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float()
        tensor = tensor.to(self.policy.device, non_blocking=True)
        if tensor.numel() and tensor.max() > 1.5:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0, 1).mul(2).sub(1).unsqueeze(0)
        action, value, _ = self.policy.act(tensor)
        return PolicyAction(
            action=int(action.reshape(-1)[0].detach().cpu().item()),
            info={
                'source': 'rl_in_pixel_env_policy',
                'policy_slot_name': self.name,
                'value': float(value.reshape(-1)[0].detach().cpu().item()),
            },
        )


class _DiamondGroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(max(1, in_channels // 32), in_channels, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class _DiamondSmallResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.f = nn.Sequential(
            _DiamondGroupNorm(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.skip_projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip_projection(x) + self.f(x)


class _DiamondActorCriticEncoder(nn.Module):
    def __init__(self, img_channels: int, channels: list[int], down: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(img_channels, channels[0], kernel_size=3, stride=1, padding=1)
        ]
        for i in range(len(channels)):
            layers.append(_DiamondSmallResBlock(channels[max(0, i - 1)], channels[i]))
            if down[i]:
                layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _DiamondActorCritic(nn.Module):
    def __init__(
        self,
        *,
        num_actions: int,
        img_channels: int = 3,
        img_size: int = 64,
        channels: list[int] | None = None,
        down: list[int] | None = None,
        lstm_dim: int = 512,
    ) -> None:
        super().__init__()
        channels = channels or [32, 32, 64, 64]
        down = down or [1, 1, 1, 1]
        self.encoder = _DiamondActorCriticEncoder(img_channels, channels, down)
        self.lstm_dim = lstm_dim
        input_dim_lstm = channels[-1] * (img_size // 2 ** sum(down)) ** 2
        self.lstm = nn.LSTMCell(input_dim_lstm, lstm_dim)
        self.critic_linear = nn.Linear(lstm_dim, 1)
        self.actor_linear = nn.Linear(lstm_dim, num_actions)

    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh.device

    def predict_act_value(self, obs: torch.Tensor, hx_cx):
        if hx_cx is None:
            batch_size = obs.shape[0]
            hx = torch.zeros(batch_size, self.lstm_dim, device=self.device)
            cx = torch.zeros(batch_size, self.lstm_dim, device=self.device)
            hx_cx = (hx, cx)
        x = self.encoder(obs).flatten(start_dim=1)
        hx, cx = self.lstm(x, hx_cx)
        return self.actor_linear(hx), self.critic_linear(hx).squeeze(dim=1), (hx, cx)


def _build_diamond_pixel_policy(
    *,
    name: str,
    checkpoint: Path,
    device: torch.device,
    num_actions: int,
) -> DiamondActorCriticPixelPolicy:
    state_dict = _load_checkpoint(checkpoint, device=device)
    actor_state = _extract_prefixed_state_dict(state_dict, 'actor_critic')
    if not actor_state:
        actor_state = state_dict
    ckpt_num_actions = actor_state.get('actor_linear.weight')
    if ckpt_num_actions is not None and int(ckpt_num_actions.shape[0]) != int(num_actions):
        raise RuntimeError(
            f'DIAMOND policy action count {ckpt_num_actions.shape[0]} does not match env action count {num_actions}.'
        )
    actor_critic = _DiamondActorCritic(num_actions=int(num_actions)).to(device).eval()
    actor_critic.load_state_dict(actor_state)
    return DiamondActorCriticPixelPolicy(name, actor_critic, device)


def _build_rl_in_pixel_env_policy(
    *,
    name: str,
    checkpoint: Path,
    device: torch.device,
    num_actions: int,
) -> RLInPixelEnvPolicy:
    checkpoint = resolve_policy_checkpoint_path(checkpoint)
    cfg = ActorCriticConfig(num_actions=int(num_actions))
    policy = load_actor_critic_policy(
        checkpoint,
        cfg=cfg,
        device=device,
        deterministic=True,
        module_name='policy',
    )
    return RLInPixelEnvPolicy(name, policy)


@dataclass
class WMSlot:
    name: str
    checkpoint: Path
    agent: Any
    variant: str
    policy_name: str
    current_state: Any = None
    current_frame: np.ndarray | None = None
    current_state_latent: Any = None
    current_obs_latent: Any = None
    current_hidden: Any = None
    state_latent_hist: deque = field(default_factory=deque)
    obs_latent_hist: deque = field(default_factory=deque)
    wm_action_hist: deque = field(default_factory=deque)
    kv_cache_rebuilds: int = 0
    step: int = 0
    reward: float = 0.0
    total_return: float = 0.0
    done: bool = False
    trunc: bool = False
    decode_warning: str | None = None
    decode_frame_max: int | None = None
    decode_frame_mean: float | None = None


@dataclass
class PolicySlot:
    name: str
    checkpoint: Path
    agent: Any = None
    variant: str = 'pixel'
    pixel_policy: PixelPolicy | None = None
    state_hist: deque = field(default_factory=deque)
    obs_hist: deque = field(default_factory=deque)
    action_hist: deque = field(default_factory=deque)


class OCStormLatentPixelPolicy(PixelPolicy):
    """Pixel-space adapter for oc-storm latent actor-critic policies."""

    def __init__(self, session: 'OCStormPlaySession', name: str = 'policy') -> None:
        self.session = session
        self.name = name

    def reset(self) -> None:
        self.session.last_policy_info = {}

    def act(self, obs: Any) -> PolicyAction:
        session = self.session
        policy_slot = session._selected_policy_slot()
        if policy_slot is None:
            return PolicyAction(session._format_action(0), {'source': 'bootstrap'})

        if policy_slot.pixel_policy is not None:
            result = policy_slot.pixel_policy.act(obs)
            action = session._format_action(result.action)
            info = dict(result.info or {})
            info.setdefault('policy_slot_name', policy_slot.name)
            info.setdefault('backend', 'real' if session.current_backend_index == 0 else 'wm')
            return PolicyAction(action, info)

        policy_input = session._pixel_policy_input(policy_slot, obs)
        if policy_input is None:
            info = {
                'entropy': None,
                'value': None,
                'source': 'bootstrap',
                'policy_slot_name': policy_slot.name,
                'backend': 'real' if session.current_backend_index == 0 else 'wm',
            }
            return PolicyAction(session._format_action(0), info)
        action_idx, entropy, value = session._policy_action_and_stats(policy_slot, policy_input, greedy=False)
        action = session._format_action(action_idx)
        session._record_policy_action(policy_slot, action)
        info = {
            'entropy': entropy,
            'value': value,
            'source': 'policy',
            'policy_slot_name': policy_slot.name,
            'backend': 'real' if session.current_backend_index == 0 else 'wm',
        }
        return PolicyAction(action, info)


class OCStormPlaySession(PlaySession):
    def __init__(
        self,
        config_path: Path,
        env_name: str,
        seed: int,
        checkpoints: list[str],
        wm_names: list[str],
        policy_names: list[str] | None = None,
        policy_checkpoints: list[str] | None = None,
        controller: str = 'human',
        bootstrap_dataset: str | None = None,
        wm_sample_mode: str = 'probs',
        wm_respect_terminal: bool = True,
        wm_disable_kv_cache: bool = False,
        wm_kv_cache_dtype: str = 'fp32',
        wm_initial_source: str = 'real',
    ) -> None:
        if wm_sample_mode not in {'probs', 'mode', 'random_sample'}:
            raise ValueError(f'Unsupported wm_sample_mode={wm_sample_mode!r}')
        if wm_kv_cache_dtype not in {'fp32', 'amp'}:
            raise ValueError(f'Unsupported wm_kv_cache_dtype={wm_kv_cache_dtype!r}')
        module = _import_config_module(config_path)
        build = getattr(module, 'build')
        build_offline = getattr(module, 'build_offline', None)
        replay_buffer = None
        feature_extractor_source = 'config.build'
        if callable(build_offline):
            params, agent_template = build_offline(env_name=env_name, seed=seed)
            variant = _variant_from_agent(agent_template)
            if variant == 'visual':
                from envs.atari.build_env import build_single_atari_env

                env, action_space = build_single_atari_env(env_name, seed)
                input_channels = int(agent_template.world_model.encoder.backbone[0].in_channels)
                feature_extractor = _OfflineVisualFeatureExtractor(
                    state_resolution=(64, 64),
                    input_channels=input_channels,
                    device=next(agent_template.parameters()).device,
                )
                feature_extractor_source = 'build_offline+rgb_zero_masks'
            else:
                if callable(getattr(agent_template, 'close', None)):
                    agent_template.close()
                params, env, action_space, feature_extractor, replay_buffer, agent_template = build(
                    env_name=env_name, seed=seed)
        else:
            params, env, action_space, feature_extractor, replay_buffer, agent_template = build(env_name=env_name, seed=seed)
        self.params = params
        self.env = env
        self.env_name = env_name
        self.seed = seed
        self.action_space = action_space
        self.feature_extractor = feature_extractor
        self.feature_extractor_source = feature_extractor_source
        self.replay_buffer = replay_buffer
        self.base_agent = agent_template
        self.device = next(agent_template.parameters()).device
        self.variant = _variant_from_agent(agent_template)
        self.bootstrap_dataset = None
        self.bootstrap_dataset_path = bootstrap_dataset
        self.action_names = _infer_action_names(env)
        self.keymap = _build_keymap(self.action_names)
        self.controller = controller if controller in {'human', 'policy'} else 'human'
        self.wm_sample_mode = wm_sample_mode
        self.wm_respect_terminal = bool(wm_respect_terminal)
        self.wm_disable_kv_cache = bool(wm_disable_kv_cache)
        self.wm_kv_cache_dtype = wm_kv_cache_dtype
        self.wm_initial_source = _validate_initial_source(wm_initial_source)
        if self.wm_initial_source == 'dataset':
            if not bootstrap_dataset:
                raise ValueError('wm_initial_source=dataset requires --bootstrap-dataset.')
            if self.variant != 'visual':
                raise ValueError('dataset initial source is currently supported only for visual STORM checkpoints.')
            input_channels = int(agent_template.world_model.encoder.backbone[0].in_channels)
            self.bootstrap_dataset = _OfflineVisualBootstrapDataset(
                bootstrap_dataset,
                self.device,
                input_channels,
            )
        self.policy_context_length = int(getattr(params, 'eval_context_length', 8))
        self.real_state_hist = deque(maxlen=self.policy_context_length + 1)
        self.real_frame_hist = deque(maxlen=self.policy_context_length + 1)
        self.real_action_hist = deque(maxlen=self.policy_context_length)
        self.real_state = None
        self.real_obs = None
        self.real_frame = None
        self.real_return = 0.0
        self.real_step = 0
        self.current_backend_index = 0
        self.policy_slot_index = 0
        self.current_obs = None
        self.last_policy_info: dict[str, Any] = {}
        wm_policy_names = [] if policy_checkpoints else (policy_names or [])
        self.wm_slots = self._build_slots(checkpoints, wm_names, wm_policy_names)
        self.policy_slots = self._build_policy_slots(policy_checkpoints or [], policy_names or [])
        if self.controller == 'policy':
            self.controller = self.policy_slots[0].name if self.policy_slots else 'human'
        self.pixel_policy = OCStormLatentPixelPolicy(self)
        self.current_slot: WMSlot | None = None
        self._bootstrap_zero = True

    def _build_slots(self, checkpoints: list[str], wm_names: list[str], policy_names: list[str]) -> list[WMSlot]:
        slots: list[WMSlot] = []
        for idx, ckpt in enumerate(checkpoints):
            agent = copy.deepcopy(self.base_agent)
            state_dict = _load_state_dict(Path(ckpt).expanduser())
            agent.load_state_dict(state_dict)
            agent.eval()
            name = wm_names[idx] if idx < len(wm_names) and wm_names[idx] else self._infer_name_from_checkpoint(ckpt)
            policy_name = policy_names[idx] if idx < len(policy_names) and policy_names[idx] else name
            slots.append(WMSlot(name=name, checkpoint=Path(ckpt).expanduser(), agent=agent, variant=self.variant, policy_name=policy_name))
        return slots

    def _build_policy_slots(self, policy_checkpoints: list[str], policy_names: list[str]) -> list[PolicySlot]:
        slots: list[PolicySlot] = []
        if policy_checkpoints:
            for idx, ckpt in enumerate(policy_checkpoints):
                name = policy_names[idx] if idx < len(policy_names) and policy_names[idx] else self._infer_name_from_checkpoint(ckpt)
                ckpt_path = resolve_policy_checkpoint_path(ckpt)
                if is_sb3_atari_policy_checkpoint(ckpt_path):
                    sb3_policy = load_sb3_pixel_policy(
                        ckpt_path,
                        name=name,
                        device=self.device,
                    )
                    slots.append(PolicySlot(
                        name=name,
                        checkpoint=ckpt_path,
                        variant='sb3-atari',
                        pixel_policy=sb3_policy,
                    ))
                    continue
                try:
                    agent = copy.deepcopy(self.base_agent)
                    state_dict = _load_state_dict(ckpt_path)
                    agent.load_state_dict(state_dict)
                    agent.eval()
                    slot = PolicySlot(name=name, checkpoint=ckpt_path, agent=agent, variant=_variant_from_agent(agent))
                except Exception as oc_error:
                    try:
                        diamond_policy = _build_diamond_pixel_policy(
                            name=name,
                            checkpoint=ckpt_path,
                            device=self.device,
                            num_actions=len(self.action_names),
                        )
                    except Exception as diamond_error:
                        try:
                            rl_policy = _build_rl_in_pixel_env_policy(
                                name=name,
                                checkpoint=ckpt_path,
                                device=self.device,
                                num_actions=len(self.action_names),
                            )
                        except Exception as rl_error:
                            raise RuntimeError(
                                f'Failed to load policy checkpoint {ckpt_path} as '
                                f'oc-storm/STORM, DIAMOND actor-critic, or '
                                f'rl-in-pixel-env policy.'
                            ) from rl_error
                        slot = PolicySlot(
                            name=name,
                            checkpoint=ckpt_path,
                            variant='rl-in-pixel-env',
                            pixel_policy=rl_policy,
                        )
                    else:
                        slot = PolicySlot(
                            name=name,
                            checkpoint=ckpt_path,
                            variant='diamond',
                            pixel_policy=diamond_policy,
                        )
                self._reset_policy_slot(slot)
                slots.append(slot)
            return slots

        return slots

    @staticmethod
    def _infer_name_from_checkpoint(ckpt: str) -> str:
        path = Path(ckpt).expanduser()
        if path.parent.name in {'ckpt', 'checkpoints', 'checkpoint'} and path.parent.parent.name:
            return path.parent.parent.name
        stem = path.stem
        stem = re.sub(r'(_?\d+p?\d*|_?latest_agent)$', '', stem).strip('_-')
        return stem or path.parent.name or path.name

    def _maybe_resize_policy_context(self, new_len: int) -> None:
        new_len = max(1, int(new_len))
        if new_len == self.policy_context_length:
            return
        self.policy_context_length = new_len
        self.real_state_hist = deque(self.real_state_hist, maxlen=self.policy_context_length + 1)
        self.real_frame_hist = deque(self.real_frame_hist, maxlen=self.policy_context_length + 1)
        self.real_action_hist = deque(self.real_action_hist, maxlen=self.policy_context_length)
        for slot in self.policy_slots:
            slot.state_hist = deque(slot.state_hist, maxlen=self.policy_context_length + 1)
            slot.obs_hist = deque(slot.obs_hist, maxlen=self.policy_context_length + 1)
            slot.action_hist = deque(slot.action_hist, maxlen=self.policy_context_length)

    def _reset_policy_slot(self, slot: PolicySlot) -> None:
        slot.state_hist = deque(maxlen=self.policy_context_length + 1)
        slot.obs_hist = deque(maxlen=self.policy_context_length + 1)
        slot.action_hist = deque(maxlen=self.policy_context_length)
        reset = getattr(slot.pixel_policy, 'reset', None)
        if callable(reset):
            reset()

    def _record_policy_action(self, slot: PolicySlot, action: Any) -> None:
        slot.action_hist.append(self._format_action(action).copy())

    def _zero_hidden(self, slot: WMSlot):
        hidden_dim = slot.agent.world_model.transformer_hidden_dim
        device = self.device
        dtype = slot.agent.world_model.amp_tensor_dtype
        if slot.variant == 'vector_visual':
            obj = slot.agent.world_model.num_objects + 1
            return torch.zeros((1, 1, obj, hidden_dim), device=device, dtype=dtype)
        if slot.variant == 'vector':
            obj = slot.agent.world_model.num_objects
            return torch.zeros((1, 1, obj, hidden_dim), device=device, dtype=dtype)
        return torch.zeros((1, 1, hidden_dim), device=device, dtype=dtype)

    def _kv_cache_max_length(self, slot: WMSlot) -> int:
        transformer = slot.agent.world_model.storm_transformer
        position_encoding = getattr(transformer, 'position_encoding', None)
        return int(getattr(position_encoding, 'max_length', 64))

    def _wm_history_max_length(self, slot: WMSlot) -> int:
        if self.wm_disable_kv_cache:
            return max(1, self._kv_cache_max_length(slot) - 1)
        context_len = max(
            int(getattr(self.params, 'imagine_context_length', 4)),
            int(getattr(self.params, 'eval_context_length', 8)),
        )
        return max(1, min(self._kv_cache_max_length(slot) - 1, context_len))

    def _reset_wm_histories(self, slot: WMSlot) -> None:
        maxlen = self._wm_history_max_length(slot)
        slot.state_latent_hist = deque(maxlen=maxlen)
        slot.obs_latent_hist = deque(maxlen=maxlen)
        slot.wm_action_hist = deque(maxlen=maxlen)

    def _remember_wm_transition(
        self,
        slot: WMSlot,
        state_latent: torch.Tensor,
        obs_latent: torch.Tensor | None,
        action: torch.Tensor,
    ) -> None:
        slot.state_latent_hist.append(state_latent.detach().clone())
        if obs_latent is not None:
            slot.obs_latent_hist.append(obs_latent.detach().clone())
        elif slot.obs_latent_hist:
            slot.obs_latent_hist.clear()
        slot.wm_action_hist.append(action.detach().clone())

    def _reset_slot_kv_cache(self, slot: WMSlot) -> None:
        transformer = slot.agent.world_model.storm_transformer
        dtype = self._wm_cache_dtype(slot)
        try:
            transformer.reset_kv_cache_list(1, self._kv_cache_max_length(slot), dtype=dtype)
        except TypeError:
            transformer.reset_kv_cache_list(1, dtype=dtype)

    def _rewarm_slot_kv_cache(self, slot: WMSlot) -> None:
        self._reset_slot_kv_cache(slot)
        if not slot.wm_action_hist or not slot.state_latent_hist:
            return
        steps = min(len(slot.wm_action_hist), len(slot.state_latent_hist))
        if slot.variant == 'vector_visual':
            steps = min(steps, len(slot.obs_latent_hist))
        if steps <= 0:
            return
        state_items = list(slot.state_latent_hist)[-steps:]
        obs_items = list(slot.obs_latent_hist)[-steps:] if slot.variant == 'vector_visual' else []
        action_items = list(slot.wm_action_hist)[-steps:]
        last_hidden = None
        with torch.no_grad(), self._wm_kv_precision(slot), self._wm_autocast(slot):
            for idx in range(steps):
                if slot.variant == 'vector_visual':
                    last_hidden, _state_hidden, _obs_hidden = slot.agent.world_model.storm_transformer.forward_with_kv_cache(
                        state_items[idx],
                        obs_items[idx],
                        action_items[idx],
                    )
                else:
                    last_hidden = slot.agent.world_model.storm_transformer.forward_with_kv_cache(
                        state_items[idx],
                        action_items[idx],
                    )
        if last_hidden is not None:
            slot.current_hidden = last_hidden
        slot.kv_cache_rebuilds += 1

    def _ensure_slot_kv_cache(self, slot: WMSlot) -> None:
        transformer = slot.agent.world_model.storm_transformer
        if not hasattr(transformer, 'kv_cache_list'):
            self._reset_slot_kv_cache(slot)
            return
        step_memory = getattr(transformer, 'kv_cache_step_memory_list', None)
        if step_memory is not None:
            if not step_memory or int(step_memory[0]) >= self._kv_cache_max_length(slot):
                self._rewarm_slot_kv_cache(slot)
            return
        cache = getattr(transformer, 'kv_cache_list', None)
        if not cache or int(cache[0].shape[1]) >= self._kv_cache_max_length(slot):
            self._rewarm_slot_kv_cache(slot)

    def _wm_autocast(self, slot: WMSlot):
        world_model = slot.agent.world_model
        dtype = getattr(world_model, 'amp_tensor_dtype', torch.bfloat16)
        enabled = bool(getattr(world_model, 'use_amp', False)) and self.device.type == 'cuda'
        return torch.autocast(device_type='cuda', dtype=dtype, enabled=enabled)

    def _wm_cache_dtype(self, slot: WMSlot) -> torch.dtype:
        if self.wm_kv_cache_dtype == 'fp32':
            return torch.float32
        return getattr(slot.agent.world_model, 'amp_tensor_dtype', torch.bfloat16)

    @contextmanager
    def _wm_kv_precision(self, slot: WMSlot):
        world_model = slot.agent.world_model
        old_use_amp = bool(getattr(world_model, 'use_amp', False))
        if self.wm_kv_cache_dtype == 'fp32':
            world_model.use_amp = False
        try:
            yield
        finally:
            world_model.use_amp = old_use_amp

    def _causal_mask(self, slot: WMSlot, latent: torch.Tensor) -> torch.Tensor:
        module = importlib.import_module(slot.agent.world_model.__class__.__module__)
        return module.get_causal_mask(latent)

    def _wm_context_tensors(
        self,
        slot: WMSlot,
        current_state_latent: torch.Tensor,
        current_obs_latent: torch.Tensor | None,
        current_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        state_items = list(slot.state_latent_hist) + [current_state_latent]
        action_items = list(slot.wm_action_hist) + [current_action]
        state_seq = torch.cat(state_items, dim=1)
        action_seq = torch.cat(action_items, dim=1)
        obs_seq = None
        if slot.variant == 'vector_visual':
            obs_items = list(slot.obs_latent_hist) + [current_obs_latent]
            obs_seq = torch.cat(obs_items, dim=1)
        return state_seq, obs_seq, action_seq

    def _wm_predict_next_no_cache(self, slot: WMSlot, action_t: torch.Tensor):
        wm = slot.agent.world_model
        with self._wm_autocast(slot):
            if slot.variant == 'vector_visual':
                state_seq, obs_seq, action_seq = self._wm_context_tensors(
                    slot, slot.current_state_latent, slot.current_obs_latent, action_t)
                dist_feat, state_dist_feat, obs_dist_feat = wm.storm_transformer(
                    state_seq, obs_seq, action_seq, self._causal_mask(slot, state_seq))
                dist_feat = dist_feat[:, -1:]
                state_dist_feat = state_dist_feat[:, -1:]
                obs_dist_feat = obs_dist_feat[:, -1:]
                state_prior_logits = wm.state_dist_head.forward_prior(state_dist_feat)
                obs_prior_logits = wm.visual_dist_head.forward_prior(obs_dist_feat)
                state_prior_sample = wm.straight_through_gradient(
                    state_prior_logits, sample_mode=self.wm_sample_mode)
                next_state_latent = torch.flatten(state_prior_sample, start_dim=3)
                obs_prior_sample = wm.straight_through_gradient(
                    obs_prior_logits, sample_mode=self.wm_sample_mode)
                next_obs_latent = torch.flatten(obs_prior_sample, start_dim=2)
                obs_hat = wm.visual_decoder(next_obs_latent)
                reward_hat = wm.symlog_twohot_loss_func.decode(wm.reward_decoder(dist_feat))
                term_hat = wm.termination_decoder(dist_feat) > 0
                return obs_hat, reward_hat, term_hat, next_state_latent, next_obs_latent, dist_feat

            state_seq, _obs_seq, action_seq = self._wm_context_tensors(
                slot, slot.current_state_latent, None, action_t)
            dist_feat = wm.storm_transformer(state_seq, action_seq, self._causal_mask(slot, state_seq))
            dist_feat = dist_feat[:, -1:]
            prior_logits = wm.dist_head.forward_prior(dist_feat)
            prior_sample = wm.straight_through_gradient(prior_logits, sample_mode=self.wm_sample_mode)
            next_latent = wm.flatten_sample(prior_sample)
            obs_hat = wm.state_decoder(next_latent)
            reward_hat = wm.symlog_twohot_loss_func.decode(wm.reward_decoder(dist_feat))
            term_hat = wm.termination_decoder(dist_feat) > 0
            return obs_hat, reward_hat, term_hat, next_latent, dist_feat

    def _record_decode_frame_stats(self, slot: WMSlot, frame: np.ndarray, source: str) -> None:
        frame = np.asarray(frame)
        if frame.size == 0:
            return
        slot.decode_frame_max = int(frame.max())
        slot.decode_frame_mean = float(frame.mean())
        if slot.decode_frame_max <= 16 and slot.decode_frame_mean <= 8.0:
            slot.decode_warning = f'decoder-dark:{source}'
        elif slot.decode_warning and slot.decode_warning.startswith('decoder-dark'):
            slot.decode_warning = None

    def _decode_current_posterior_frame(self, slot: WMSlot) -> np.ndarray | None:
        if slot.variant == 'vector_visual' and slot.current_obs_latent is not None:
            decoded = slot.agent.world_model.visual_decoder(slot.current_obs_latent)
        elif slot.variant == 'visual' and slot.current_state_latent is not None:
            decoded = slot.agent.world_model.state_decoder(slot.current_state_latent)
        else:
            return None
        return _tensor_to_frame(decoded)

    def _set_slot_frame_from_posterior_decode(self, slot: WMSlot) -> None:
        try:
            with torch.no_grad():
                frame = self._decode_current_posterior_frame(slot)
            if frame is None:
                return
            self._record_decode_frame_stats(slot, frame, 'posterior')
            slot.current_frame = frame
        except Exception as exc:
            slot.decode_warning = f'decode-check-failed:{exc.__class__.__name__}'

    def _zero_visual_latent(self, slot: WMSlot, batch_size: int = 1) -> torch.Tensor:
        stoch_dim = int(getattr(slot.agent.world_model.dist_head, 'stoch_dim', 32))
        return torch.zeros(batch_size, 1, stoch_dim * stoch_dim, device=self.device, dtype=torch.float32)

    def _zero_vector_latent(self, slot: WMSlot, batch_size: int = 1) -> torch.Tensor:
        wm = slot.agent.world_model
        latent_width = int(getattr(wm, 'latent_width', 16))
        return torch.zeros(
            batch_size,
            1,
            int(getattr(wm, 'num_objects', 1)),
            latent_width * latent_width,
            device=self.device,
            dtype=torch.float32,
        )

    def _bootstrap_slot_from_prior(self, slot: WMSlot) -> None:
        action_t = torch.zeros(1, 1, 1, device=self.device, dtype=torch.int32)
        slot.current_hidden = self._zero_hidden(slot)
        self._reset_wm_histories(slot)
        self._reset_slot_kv_cache(slot)
        with torch.no_grad():
            if slot.variant == 'vector_visual':
                slot.current_state_latent = self._zero_vector_latent(slot)
                slot.current_obs_latent = self._zero_visual_latent(slot)
                obs_hat, reward_hat, term_hat, next_state_latent, next_obs_latent, dist_feat = (
                    self._wm_predict_next_no_cache(slot, action_t))
                slot.current_state_latent = next_state_latent.detach()
                slot.current_obs_latent = next_obs_latent.detach()
            elif slot.variant == 'visual':
                slot.current_state_latent = self._zero_visual_latent(slot)
                slot.current_obs_latent = None
                obs_hat, reward_hat, term_hat, next_latent, dist_feat = (
                    self._wm_predict_next_no_cache(slot, action_t))
                slot.current_state_latent = next_latent.detach()
            else:
                slot.current_state_latent = self._zero_vector_latent(slot)
                slot.current_obs_latent = None
                obs_hat, reward_hat, term_hat, next_latent, dist_feat = (
                    self._wm_predict_next_no_cache(slot, action_t))
                slot.current_state_latent = next_latent.detach()
            slot.current_hidden = dist_feat.detach()
        del reward_hat, term_hat
        frame = _tensor_to_frame(obs_hat)
        self._record_decode_frame_stats(slot, frame, 'zero-prior')
        slot.current_frame = frame
        slot.step = 0
        slot.reward = 0.0
        slot.total_return = 0.0
        slot.done = False
        slot.trunc = False
        slot.kv_cache_rebuilds = 0

    def _seed_visual_slot_context(self, slot: WMSlot, obs_seq: torch.Tensor, action_seq: torch.Tensor | None) -> None:
        obs_seq = obs_seq.unsqueeze(0)
        with torch.no_grad():
            latent = slot.agent.world_model.encode_obs(obs_seq, sample_mode=self.wm_sample_mode)
            slot.current_state_latent = latent[:, -1:]
            slot.current_obs_latent = None
            slot.current_hidden = self._zero_hidden(slot)
            self._reset_wm_histories(slot)
            self._reset_slot_kv_cache(slot)
            if action_seq is None or action_seq.numel() == 0 or latent.shape[1] < 2:
                return
            action_seq = action_seq.reshape(1, -1, action_seq.shape[-1]).to(self.device, dtype=torch.int32)
            steps = min(latent.shape[1] - 1, action_seq.shape[1])
            last_hidden = None
            with self._wm_kv_precision(slot), self._wm_autocast(slot):
                for idx in range(steps):
                    last_hidden = slot.agent.world_model.storm_transformer.forward_with_kv_cache(
                        latent[:, idx:idx + 1],
                        action_seq[:, idx:idx + 1],
                    )
                    self._remember_wm_transition(
                        slot,
                        latent[:, idx:idx + 1],
                        None,
                        action_seq[:, idx:idx + 1],
                    )
            if last_hidden is not None:
                slot.current_hidden = last_hidden

    def _bootstrap_visual_slot_from_dataset(self, slot: WMSlot) -> bool:
        if self.bootstrap_dataset is None:
            return False
        obs_seq, action_seq, frame = self.bootstrap_dataset.sample(self.policy_context_length + 1)
        slot.current_frame = frame
        self._seed_visual_slot_context(slot, obs_seq, action_seq)
        slot.step = 0
        slot.reward = 0.0
        slot.total_return = 0.0
        slot.done = False
        slot.trunc = False
        slot.kv_cache_rebuilds = 0
        self._set_slot_frame_from_posterior_decode(slot)
        return True

    def _bootstrap_slot_from_real(self, slot: WMSlot) -> None:
        if self.real_frame is None:
            return
        if slot.variant != 'visual' and self.real_state is None:
            return
        slot.current_frame = np.array(self.real_frame, copy=True)
        action_seq = None
        if len(self.real_action_hist) > 0:
            action_seq = torch.as_tensor(
                np.stack(list(self.real_action_hist), axis=0),
                device=self.device,
                dtype=torch.int32,
            ).unsqueeze(0)
        if slot.variant == 'vector_visual':
            if len(self.real_state_hist) > 0 and len(self.real_frame_hist) > 0:
                state_seq = torch.stack(list(self.real_state_hist), dim=0).unsqueeze(0)
                obs_seq = torch.stack(list(self.real_frame_hist), dim=0).unsqueeze(0)
            else:
                state_seq = self.real_state.unsqueeze(0).unsqueeze(0)
                obs_source = self.real_obs if self.real_obs is not None else _frame_to_tensor(self.real_frame, self.device)
                obs_seq = obs_source.unsqueeze(0).unsqueeze(0)
            state_latent, obs_latent = slot.agent.world_model.encode_obs(
                state_seq, obs_seq, sample_mode=self.wm_sample_mode)
            slot.current_state_latent = state_latent[:, -1:]
            slot.current_obs_latent = obs_latent[:, -1:]
            slot.current_hidden = self._zero_hidden(slot)
            self._reset_wm_histories(slot)
            self._reset_slot_kv_cache(slot)
            if action_seq is not None and state_latent.shape[1] >= 2:
                steps = min(state_latent.shape[1] - 1, action_seq.shape[1])
                last_hidden = None
                with self._wm_kv_precision(slot), self._wm_autocast(slot):
                    for idx in range(steps):
                        last_hidden, _state_hidden, _obs_hidden = slot.agent.world_model.storm_transformer.forward_with_kv_cache(
                            state_latent[:, idx:idx + 1],
                            obs_latent[:, idx:idx + 1],
                            action_seq[:, idx:idx + 1],
                        )
                        self._remember_wm_transition(
                            slot,
                            state_latent[:, idx:idx + 1],
                            obs_latent[:, idx:idx + 1],
                            action_seq[:, idx:idx + 1],
                        )
                if last_hidden is not None:
                    slot.current_hidden = last_hidden
        elif slot.variant == 'visual':
            if len(self.real_frame_hist) > 0:
                obs_seq = torch.stack(list(self.real_frame_hist), dim=0).unsqueeze(0)
            else:
                obs_source = self.real_obs if self.real_obs is not None else _frame_to_tensor(self.real_frame, self.device)
                obs_seq = obs_source.unsqueeze(0).unsqueeze(0)
            latent = slot.agent.world_model.encode_obs(obs_seq, sample_mode=self.wm_sample_mode)
            slot.current_state_latent = latent[:, -1:]
            slot.current_obs_latent = None
            slot.current_hidden = self._zero_hidden(slot)
            self._reset_wm_histories(slot)
            self._reset_slot_kv_cache(slot)
            if action_seq is not None and latent.shape[1] >= 2:
                steps = min(latent.shape[1] - 1, action_seq.shape[1])
                last_hidden = None
                with self._wm_kv_precision(slot), self._wm_autocast(slot):
                    for idx in range(steps):
                        last_hidden = slot.agent.world_model.storm_transformer.forward_with_kv_cache(
                            latent[:, idx:idx + 1],
                            action_seq[:, idx:idx + 1],
                        )
                        self._remember_wm_transition(
                            slot,
                            latent[:, idx:idx + 1],
                            None,
                            action_seq[:, idx:idx + 1],
                        )
                if last_hidden is not None:
                    slot.current_hidden = last_hidden
        else:
            if len(self.real_state_hist) > 0:
                state_seq = torch.stack(list(self.real_state_hist), dim=0).unsqueeze(0)
            else:
                state_seq = self.real_state.unsqueeze(0).unsqueeze(0)
            latent = slot.agent.world_model.encode_obs(state_seq, sample_mode=self.wm_sample_mode)
            slot.current_state_latent = latent[:, -1:]
            slot.current_obs_latent = None
            slot.current_hidden = self._zero_hidden(slot)
            self._reset_wm_histories(slot)
            self._reset_slot_kv_cache(slot)
            if action_seq is not None and latent.shape[1] >= 2:
                steps = min(latent.shape[1] - 1, action_seq.shape[1])
                last_hidden = None
                with self._wm_kv_precision(slot), self._wm_autocast(slot):
                    for idx in range(steps):
                        last_hidden = slot.agent.world_model.storm_transformer.forward_with_kv_cache(
                            latent[:, idx:idx + 1],
                            action_seq[:, idx:idx + 1],
                        )
                        self._remember_wm_transition(
                            slot,
                            latent[:, idx:idx + 1],
                            None,
                            action_seq[:, idx:idx + 1],
                        )
                if last_hidden is not None:
                    slot.current_hidden = last_hidden
        slot.step = 0
        slot.reward = 0.0
        slot.total_return = 0.0
        slot.done = False
        slot.trunc = False
        slot.kv_cache_rebuilds = 0
        self._set_slot_frame_from_posterior_decode(slot)

    def _reset_real_env_for_wm_context(self) -> None:
        if callable(getattr(self.feature_extractor, 'reset', None)):
            self.feature_extractor.reset()
        raw_obs, _ = self.env.reset()
        state, obs, _display = self._extract_real_state_and_obs(raw_obs)
        frame = _resize_frame(raw_obs, 64)
        self.real_state = state
        self.real_obs = obs
        self.real_frame = frame
        self.real_return = 0.0
        self.real_step = 0
        self._reset_real_histories()
        self._record_real_context(
            state,
            obs if obs is not None else _frame_to_tensor(frame, self.device),
        )
        context_frames = max(1, int(getattr(self.params, 'imagine_context_length', 4)))
        noop = self._format_action(0)
        while len(self.real_frame_hist) < context_frames:
            next_raw_obs, rew, end, trunc, info = self.env.step(noop)
            next_state, next_obs, _ = self._extract_real_state_and_obs(next_raw_obs)
            next_frame = _resize_frame(next_raw_obs, 64)
            self.real_state = next_state
            self.real_obs = next_obs
            self.real_frame = next_frame
            self.real_return += float(np.asarray(rew).item() if hasattr(np.asarray(rew), 'item') else float(rew))
            self.real_step += 1
            self.real_action_hist.append(noop.copy())
            self._record_real_context(
                next_state,
                next_obs if next_obs is not None else _frame_to_tensor(next_frame, self.device),
            )
            if end or trunc:
                raw_obs, _ = self.env.reset()
                state, obs, _display = self._extract_real_state_and_obs(raw_obs)
                frame = _resize_frame(raw_obs, 64)
                self.real_state = state
                self.real_obs = obs
                self.real_frame = frame
                self.real_return = 0.0
                self.real_step = 0
                self._reset_real_histories()
                self._record_real_context(
                    state,
                    obs if obs is not None else _frame_to_tensor(frame, self.device),
                )

    def _reset_real_histories(self) -> None:
        self.real_state_hist.clear()
        self.real_frame_hist.clear()
        self.real_action_hist.clear()

    def _record_real_context(self, state, frame) -> None:
        if state is not None:
            self.real_state_hist.append(state)
        self.real_frame_hist.append(frame)

    def _extract_real_state_and_obs(self, raw_obs):
        extracted, display = self.feature_extractor.extract_features(raw_obs)
        obs = None
        state = extracted
        if self.variant == 'vector_visual':
            if not isinstance(extracted, (tuple, list)) or len(extracted) != 2:
                raise RuntimeError(
                    'vector_visual feature extractor must return '
                    '((object_features, visual_obs), display_obs)'
                )
            state, obs = extracted
        elif self.variant == 'visual':
            obs = extracted
            state = None
        state = _ensure_tensor(state, self.device) if state is not None else None
        obs = _ensure_tensor(obs, self.device) if obs is not None else None
        return state, obs, display

    def _current_real_policy_input(self, slot: WMSlot):
        if self.variant == 'visual':
            if len(self.real_frame_hist) == 0:
                return None
        elif len(self.real_state_hist) == 0:
            return None
        if self.variant == 'vector_visual':
            state_seq = torch.stack(list(self.real_state_hist), dim=0).unsqueeze(0)
            frame_seq = torch.stack(list(self.real_frame_hist), dim=0).unsqueeze(0)
            state_latent, obs_latent = slot.agent.world_model.encode_obs(
                state_seq, frame_seq, sample_mode=self.wm_sample_mode)
            current_state_latent = state_latent[:, -1:]
            current_obs_latent = obs_latent[:, -1:]
            if len(self.real_action_hist) == 0:
                hidden = self._zero_hidden(slot)
            else:
                action_seq = torch.as_tensor(np.stack(list(self.real_action_hist), axis=0), device=self.device, dtype=torch.int32).unsqueeze(0)
                hidden = slot.agent.world_model.calc_last_hidden(state_latent[:, :-1], obs_latent[:, :-1], action_seq)
            policy_input = torch.cat([
                torch.flatten(current_state_latent, start_dim=2),
                current_obs_latent,
                torch.flatten(hidden, start_dim=2),
            ], dim=-1)
            return policy_input, hidden
        if self.variant == 'visual':
            frame_seq = torch.stack(list(self.real_frame_hist), dim=0).unsqueeze(0)
            latent = slot.agent.world_model.encode_obs(frame_seq, sample_mode=self.wm_sample_mode)
            current_latent = latent[:, -1:]
            if len(self.real_action_hist) == 0:
                hidden = self._zero_hidden(slot)
            else:
                action_seq = torch.as_tensor(np.stack(list(self.real_action_hist), axis=0), device=self.device, dtype=torch.int32).unsqueeze(0)
                hidden = slot.agent.world_model.calc_last_hidden(latent[:, :-1], action_seq)
            policy_input = torch.cat([current_latent, hidden], dim=-1)
            return policy_input, hidden
        state_seq = torch.stack(list(self.real_state_hist), dim=0).unsqueeze(0)
        latent = slot.agent.world_model.encode_obs(state_seq, sample_mode=self.wm_sample_mode)
        current_latent = latent[:, -1:]
        if len(self.real_action_hist) == 0:
            hidden = self._zero_hidden(slot)
        else:
            action_seq = torch.as_tensor(np.stack(list(self.real_action_hist), axis=0), device=self.device, dtype=torch.int32).unsqueeze(0)
            hidden = slot.agent.world_model.calc_last_hidden(latent[:, :-1], action_seq)
        policy_input = torch.cat([current_latent, hidden], dim=-1)
        return policy_input, hidden

    def _pixel_policy_input(self, slot: PolicySlot, obs: Any):
        frame = _tensor_to_frame(obs)
        extracted, _display = self.feature_extractor.extract_features(frame)

        if slot.variant == 'vector_visual':
            if not isinstance(extracted, (tuple, list)) or len(extracted) != 2:
                raise RuntimeError(
                    'vector_visual pixel policy requires a feature extractor '
                    'that returns (object_features, visual_obs).'
                )
            state, visual_obs = extracted
            slot.state_hist.append(_ensure_tensor(state, self.device))
            slot.obs_hist.append(_ensure_tensor(visual_obs, self.device))
            if len(slot.state_hist) == 0 or len(slot.obs_hist) == 0:
                return None
            state_seq = torch.stack(list(slot.state_hist), dim=0).unsqueeze(0)
            obs_seq = torch.stack(list(slot.obs_hist), dim=0).unsqueeze(0)
            state_latent, obs_latent = slot.agent.world_model.encode_obs(
                state_seq, obs_seq, sample_mode=self.wm_sample_mode)
            current_state_latent = state_latent[:, -1:]
            current_obs_latent = obs_latent[:, -1:]
            if len(slot.action_hist) == 0:
                hidden = self._zero_hidden(slot)
            else:
                action_seq = torch.as_tensor(np.stack(list(slot.action_hist), axis=0), device=self.device, dtype=torch.int32).unsqueeze(0)
                hidden = slot.agent.world_model.calc_last_hidden(state_latent[:, :-1], obs_latent[:, :-1], action_seq)
            return torch.cat([
                torch.flatten(current_state_latent, start_dim=2),
                current_obs_latent,
                torch.flatten(hidden, start_dim=2),
            ], dim=-1)

        state = _ensure_tensor(extracted, self.device)
        slot.state_hist.append(state)
        if len(slot.state_hist) == 0:
            return None
        state_seq = torch.stack(list(slot.state_hist), dim=0).unsqueeze(0)
        latent = slot.agent.world_model.encode_obs(state_seq, sample_mode=self.wm_sample_mode)
        current_latent = latent[:, -1:]
        if len(slot.action_hist) == 0:
            hidden = self._zero_hidden(slot)
        else:
            action_seq = torch.as_tensor(np.stack(list(slot.action_hist), axis=0), device=self.device, dtype=torch.int32).unsqueeze(0)
            hidden = slot.agent.world_model.calc_last_hidden(latent[:, :-1], action_seq)
        return torch.cat([current_latent, hidden], dim=-1)

    def _wm_policy_input(self, slot: WMSlot):
        if slot.current_state_latent is None:
            return None
        if slot.variant == 'vector_visual':
            return torch.cat([
                torch.flatten(slot.current_state_latent, start_dim=2),
                slot.current_obs_latent,
                torch.flatten(slot.current_hidden, start_dim=2),
            ], dim=-1)
        if slot.variant == 'visual':
            return torch.cat([slot.current_state_latent, slot.current_hidden], dim=-1)
        return torch.cat([slot.current_state_latent, slot.current_hidden], dim=-1)

    def _policy_action_and_stats(self, slot: WMSlot, policy_input, greedy: bool = False):
        with torch.no_grad():
            logits = slot.agent.actor_critic.policy(policy_input)
            dist = torch.distributions.Categorical(logits=logits)
            if greedy:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            entropy = float((dist.entropy().sum(dim=-1) / math.log(2)).item())
            value = float(slot.agent.actor_critic.value(policy_input).item())
        action_np = action.detach().cpu().numpy().reshape(-1)
        action_idx = int(action_np[0]) if action_np.size else 0
        return action_idx, entropy, value

    def _selected_policy_slot(self) -> PolicySlot | None:
        if not self.policy_slots:
            return None
        idx = min(max(self.policy_slot_index, 0), len(self.policy_slots) - 1)
        return self.policy_slots[idx]

    def _current_backend_label(self) -> str:
        if self.current_backend_index == 0:
            return 'real'
        slot = self.wm_slots[self.current_backend_index - 1]
        return f'{self.current_backend_index}/{len(self.wm_slots)} ({slot.name})'

    def _current_wm_label(self) -> str:
        if not self.wm_slots:
            return ''
        idx = self.current_backend_index - 1 if self.current_backend_index > 0 else 0
        idx = min(max(idx, 0), len(self.wm_slots) - 1)
        return f'{idx + 1}/{len(self.wm_slots)} ({self.wm_slots[idx].name})'

    @property
    def horizon(self) -> int | None:
        if self.current_backend_index == 0:
            return None
        return self.policy_context_length

    def reset(self) -> None:
        self.policy_slot_index = min(max(self.policy_slot_index, 0), len(self.policy_slots) - 1) if self.policy_slots else -1
        self.last_policy_info = {}
        self.pixel_policy.reset()
        for slot in self.policy_slots:
            self._reset_policy_slot(slot)
        needs_real_context = self.current_backend_index == 0 or self.wm_initial_source == 'real'
        if needs_real_context:
            self._reset_real_env_for_wm_context()
        for slot in self.wm_slots:
            if self.wm_initial_source == 'prior':
                self._bootstrap_slot_from_prior(slot)
            elif self.wm_initial_source == 'dataset':
                if not self._bootstrap_visual_slot_from_dataset(slot):
                    raise RuntimeError('Failed to initialize STORM WM from bootstrap dataset.')
            else:
                self._bootstrap_slot_from_real(slot)
        if self.current_backend_index > 0 and self.wm_slots:
            self.current_slot = self.wm_slots[self.current_backend_index - 1]
            self.current_obs = np.array(self.current_slot.current_frame, copy=True)
        else:
            self.current_backend_index = 0
            self.current_slot = None
            frame = self.real_frame if self.real_frame is not None else np.zeros((64, 64, 3), dtype=np.uint8)
            self.current_obs = np.array(frame, copy=True)

    def bootstrap_from_observation(self, obs: Any):
        frame = _resize_frame(np.asarray(obs), 64)
        state, visual_obs, _display = self._extract_real_state_and_obs(frame)
        self.real_state = state
        self.real_obs = visual_obs
        self.real_frame = frame
        self.real_state_hist.clear()
        self.real_frame_hist.clear()
        self.real_action_hist.clear()
        self._record_real_context(
            state,
            visual_obs if visual_obs is not None else _frame_to_tensor(frame, self.device),
        )
        for slot in self.wm_slots:
            self._bootstrap_slot_from_real(slot)
        if self.current_backend_index > 0 and self.wm_slots:
            self.current_slot = self.wm_slots[self.current_backend_index - 1]
            self.current_obs = np.array(self.current_slot.current_frame, copy=True)
        else:
            self.current_obs = np.array(frame, copy=True)
        return self.current_obs, {
            'backend': 'wm' if self.current_backend_index > 0 else 'real',
            'wm_initial_source': 'observation',
        }

    def switch_backend(self, direction: int) -> None:
        count = 1 + len(self.wm_slots)
        if count == 1:
            self.reset()
            return
        self.current_backend_index = (self.current_backend_index + direction) % count
        if self.current_backend_index > 0:
            self.current_slot = self.wm_slots[self.current_backend_index - 1]
        else:
            self.current_slot = None
        self.reset()
        if self.current_backend_index > 0:
            self.current_slot = self.wm_slots[self.current_backend_index - 1]

    def switch_controller(self) -> None:
        if self.controller == 'human' and not self.policy_slots:
            return
        controllers = ['human'] + [slot.name for slot in self.policy_slots]
        index = controllers.index(self.controller) if self.controller in controllers else 0
        index = (index + 1) % len(controllers)
        self.controller = controllers[index]
        if index > 0:
            self.policy_slot_index = index - 1

    def switch_policy(self, direction: int) -> None:
        if not self.policy_slots:
            return
        self.policy_slot_index = (self.policy_slot_index + int(direction)) % len(self.policy_slots)

    def adjust_horizon(self, delta: int) -> None:
        current = self.horizon
        if current is None:
            return
        self.set_horizon(current + delta)

    def set_horizon(self, horizon: int) -> None:
        if self.current_backend_index == 0:
            return
        self._maybe_resize_policy_context(horizon)

    def _format_action(self, action: Any) -> np.ndarray:
        idx = _action_to_index(action)
        return np.array([idx], dtype=np.int32)

    def _action_name(self, action: Any) -> str:
        idx = _action_to_index(action)
        if 0 <= idx < len(self.action_names):
            return self.action_names[idx]
        return str(idx)

    def choose_action(self, human_action: int) -> Any:
        if self.controller == 'human' or not self.policy_slots:
            self.last_policy_info = {'entropy': None, 'value': None, 'source': 'human'}
            return self._format_action(human_action)

        result = self.pixel_policy.act(self.current_obs)
        self.last_policy_info = dict(result.info or {})
        return result.action

    def _real_step(self, action: Any) -> StepResult:
        env_action = self._format_action(action)
        next_raw_obs, rew, end, trunc, info = self.env.step(env_action)
        if callable(getattr(self.feature_extractor, 'reset', None)) and (end or trunc):
            pass
        next_state, next_obs, _ = self._extract_real_state_and_obs(next_raw_obs)
        next_frame = _resize_frame(next_raw_obs, 64)
        self.real_state = next_state
        self.real_obs = next_obs
        self.real_frame = next_frame
        self.current_obs = np.array(next_frame, copy=True)
        self.real_return += float(np.asarray(rew).item() if hasattr(np.asarray(rew), 'item') else float(rew))
        self.real_step += 1
        self._record_real_context(
            next_state,
            next_obs if next_obs is not None else _frame_to_tensor(next_frame, self.device),
        )
        self.real_action_hist.append(env_action.copy())
        result_info = {
            'backend': 'real',
            'wm_label': 'real',
            'controller': self.controller,
            'device': str(self.device),
            'step': self.real_step,
            'return': self.real_return,
            'reward': float(np.asarray(rew).item() if hasattr(np.asarray(rew), 'item') else float(rew)),
            'action_name': self._action_name(env_action),
            'entropy': self.last_policy_info.get('entropy'),
            'value': self.last_policy_info.get('value'),
            'policy_context_length': self.policy_context_length,
            'slot_name': self.last_policy_info.get(
                'policy_slot_name',
                self._selected_policy_slot().name if self._selected_policy_slot() else None),
            'done': bool(end),
            'trunc': bool(trunc),
            'last': bool(end or trunc),
            'terminal': bool(end),
            'source': 'real Atari env',
        }
        return StepResult(obs=self.current_obs, reward=result_info['reward'], done=bool(end), trunc=bool(trunc), info=result_info)

    def _wm_step(self, action: Any) -> StepResult:
        slot = self.wm_slots[self.current_backend_index - 1]
        if not self.wm_disable_kv_cache:
            self._ensure_slot_kv_cache(slot)
        env_action = self._format_action(action)
        action_t = torch.as_tensor(env_action, device=self.device, dtype=torch.int32).reshape(1, 1, -1)
        with torch.no_grad():
            if slot.variant == 'vector_visual':
                prev_state_latent = slot.current_state_latent
                prev_obs_latent = slot.current_obs_latent
                if self.wm_disable_kv_cache:
                    obs_hat, reward_hat, term_hat, next_state_latent, next_obs_latent, dist_feat = (
                        self._wm_predict_next_no_cache(slot, action_t))
                else:
                    with self._wm_kv_precision(slot):
                        obs_hat, reward_hat, term_hat, next_state_latent, next_obs_latent, dist_feat = slot.agent.world_model.predict_next(
                            slot.current_state_latent, slot.current_obs_latent, action_t,
                            log_video=True, sample_mode=self.wm_sample_mode)
                slot.current_state_latent = next_state_latent
                slot.current_obs_latent = next_obs_latent
                slot.current_hidden = dist_feat
                self._remember_wm_transition(slot, prev_state_latent, prev_obs_latent, action_t)
            elif slot.variant == 'visual':
                prev_latent = slot.current_state_latent
                if self.wm_disable_kv_cache:
                    obs_hat, reward_hat, term_hat, next_latent, dist_feat = (
                        self._wm_predict_next_no_cache(slot, action_t))
                else:
                    with self._wm_kv_precision(slot):
                        obs_hat, reward_hat, term_hat, next_latent, dist_feat = slot.agent.world_model.predict_next(
                            slot.current_state_latent, action_t, log_video=True, sample_mode=self.wm_sample_mode)
                slot.current_state_latent = next_latent
                slot.current_hidden = dist_feat
                self._remember_wm_transition(slot, prev_latent, None, action_t)
            else:
                prev_latent = slot.current_state_latent
                if self.wm_disable_kv_cache:
                    obs_hat, reward_hat, term_hat, next_latent, dist_feat = (
                        self._wm_predict_next_no_cache(slot, action_t))
                else:
                    with self._wm_kv_precision(slot):
                        obs_hat, reward_hat, term_hat, next_latent, dist_feat = slot.agent.world_model.predict_next(
                            slot.current_state_latent, action_t, log_video=True, sample_mode=self.wm_sample_mode)
                slot.current_state_latent = next_latent
                slot.current_hidden = dist_feat
                self._remember_wm_transition(slot, prev_latent, None, action_t)
        reward = float(reward_hat.item()) if hasattr(reward_hat, 'item') else float(reward_hat)
        predicted_terminal = bool(term_hat.item()) if hasattr(term_hat, 'item') else bool(term_hat)
        slot.reward = reward
        slot.total_return += reward
        slot.step += 1
        predicted_trunc = bool(slot.step >= self.policy_context_length)
        done = predicted_terminal if self.wm_respect_terminal else False
        trunc = predicted_trunc
        slot.done = done
        slot.trunc = trunc
        frame = _tensor_to_frame(obs_hat)
        self._record_decode_frame_stats(slot, frame, 'prior')
        slot.current_frame = frame
        self.current_obs = np.array(frame, copy=True)
        result_info = {
            'backend': 'wm',
            'wm_label': f'{self.current_backend_index}/{len(self.wm_slots)} ({slot.name})',
            'controller': self.controller,
            'device': str(self.device),
            'step': slot.step,
            'return': slot.total_return,
            'reward': reward,
            'action_name': self._action_name(env_action),
            'entropy': self.last_policy_info.get('entropy'),
            'value': self.last_policy_info.get('value'),
            'policy_context_length': self.policy_context_length,
            'slot_name': self.last_policy_info.get(
                'policy_slot_name',
                self._selected_policy_slot().name if self._selected_policy_slot() else None),
            'done': done,
            'trunc': trunc,
            'last': bool(done or trunc),
            'terminal': done,
            'terminal_predicted': predicted_terminal,
            'trunc_predicted': predicted_trunc,
            'terminal_ignored': bool(predicted_terminal and not self.wm_respect_terminal),
            'decode_warning': slot.decode_warning,
            'decode_frame_max': slot.decode_frame_max,
            'decode_frame_mean': slot.decode_frame_mean,
            'kv_cache_rebuilds': slot.kv_cache_rebuilds,
            'wm_cache': 'off' if self.wm_disable_kv_cache else f'kv-{self.wm_kv_cache_dtype}',
            'source': 'imagined world model',
        }
        return StepResult(obs=self.current_obs, reward=reward, done=done, trunc=trunc, info=result_info)

    def step(self, action: Any) -> StepResult:
        if self.current_backend_index == 0:
            return self._real_step(action)
        return self._wm_step(action)

    def header(self, action: Any, info: dict[str, Any]) -> list[str]:
        if not isinstance(info, dict):
            info = {}
        backend = info.get('backend', 'real')
        controller = info.get('controller', self.controller)
        step = info.get('step', 0)
        action_name = info.get('action_name', self._action_name(action))
        policy_slot = info.get('policy_slot_name') or info.get('slot_name')
        terminal_predicted = info.get('terminal_predicted')
        trunc_predicted = info.get('trunc_predicted')
        decode_warning = info.get('decode_warning')
        env_name = 'real' if backend == 'real' else self.wm_slots[self.current_backend_index - 1].name
        status = {
            'env_name': env_name,
            'env_kind': 'real' if backend == 'real' else 'model',
            'control': controller,
            'step': step,
            'reward': info.get('reward'),
            'return': info.get('return'),
            'action_name': action_name,
            'done': bool(info.get('done', False)),
            'trunc': bool(info.get('trunc', False)),
        }
        extras = []
        if self.policy_slots:
            idx = min(max(self.policy_slot_index, 0), len(self.policy_slots) - 1)
            name = policy_slot or self.policy_slots[idx].name
            extras.append(('Policy', f'{idx + 1}/{len(self.policy_slots)} ({name})'))
        if backend != 'real':
            extras.append(('KV cache', info.get('wm_cache', 'off' if self.wm_disable_kv_cache else f'kv-{self.wm_kv_cache_dtype}')))
        if terminal_predicted is not None:
            extras.append(('Pred term', bool(terminal_predicted)))
        if trunc_predicted is not None:
            extras.append(('Pred trunc', bool(trunc_predicted)))
        if decode_warning:
            extras.append(('WM warn', decode_warning))
        if backend == 'wm' and info.get('kv_cache_rebuilds'):
            extras.append(('KV rebuild', info.get('kv_cache_rebuilds')))
        return play_status_lines(status, extras)

    def render_frame(self, size: int, header_lines: list[str]):
        frame = self.current_obs if self.current_obs is not None else np.zeros((64, 64, 3), dtype=np.uint8)
        frame = _tensor_to_frame(frame)
        return Image.fromarray(frame).resize((size, size), resample=Image.NEAREST)

    def record_metadata(self) -> dict[str, Any]:
        backend = 'real'
        if self.current_backend_index > 0 and self.wm_slots:
            backend = self.wm_slots[self.current_backend_index - 1].name
        policy = ''
        if self.policy_slots and self.policy_slot_index >= 0:
            policy = self.policy_slots[self.policy_slot_index].name
        return {
            'project': 'oc-storm/STORM',
            'env_name': self.env_name,
            'backend': backend,
            'backend_index': int(self.current_backend_index),
            'controller': self.controller,
            'policy_index': int(self.policy_slot_index),
            'policy_label': policy,
            'wm_initial_source': self.wm_initial_source,
            'wm_sample_mode': self.wm_sample_mode,
            'wm_cache': 'off' if self.wm_disable_kv_cache else f'kv-{self.wm_kv_cache_dtype}',
        }

    def close(self) -> None:
        close = getattr(self.env, 'close', None)
        if callable(close):
            close()
        for slot in self.wm_slots:
            close_slot = getattr(slot.agent, 'close', None)
            if callable(close_slot):
                try:
                    close_slot()
                except Exception:
                    pass


def build_oc_storm_session(
    config_path: Path,
    env_name: str,
    seed: int,
    checkpoint_args: list[str],
    wm_name_args: list[str],
    policy_name_args: list[str] | None = None,
    policy_checkpoint_args: list[str] | None = None,
    controller: str = 'human',
    bootstrap_dataset: str | None = None,
    wm_sample_mode: str = 'probs',
    wm_respect_terminal: bool = True,
    wm_disable_kv_cache: bool = False,
    wm_kv_cache_dtype: str = 'fp32',
    wm_initial_source: str = 'real',
) -> OCStormPlaySession:
    return OCStormPlaySession(
        config_path=config_path,
        env_name=env_name,
        seed=seed,
        checkpoints=checkpoint_args,
        wm_names=wm_name_args,
        policy_names=policy_name_args,
        policy_checkpoints=policy_checkpoint_args,
        controller=controller,
        bootstrap_dataset=bootstrap_dataset,
        wm_sample_mode=wm_sample_mode,
        wm_respect_terminal=wm_respect_terminal,
        wm_disable_kv_cache=wm_disable_kv_cache,
        wm_kv_cache_dtype=wm_kv_cache_dtype,
        wm_initial_source=wm_initial_source,
    )
