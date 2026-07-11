import numpy as np
import random
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil


def _ordered_indices(buffer):
    if buffer.length <= 0:
        return torch.empty((0,), dtype=torch.long, device=buffer.episode_buffer.device)
    if buffer.length < buffer.max_length:
        return torch.arange(buffer.length, device=buffer.episode_buffer.device)
    start = (buffer.last_pointer + 1) % buffer.max_length
    return torch.cat(
        [
            torch.arange(start, buffer.max_length, device=buffer.episode_buffer.device),
            torch.arange(0, start, device=buffer.episode_buffer.device),
        ]
    )


def _episode_ranges(episode_ids: torch.Tensor):
    if episode_ids.numel() == 0:
        return []
    episode_ids = episode_ids.detach().cpu().long()
    ranges = []
    start = 0
    current = int(episode_ids[0].item())
    for index in range(1, int(episode_ids.numel())):
        value = int(episode_ids[index].item())
        if value != current:
            ranges.append((current, start, index))
            start = index
            current = value
    ranges.append((current, start, int(episode_ids.numel())))
    return ranges


def _gather_cpu(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tensor.index_select(0, indices).detach().cpu().contiguous()


def _obs_for_export(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.detach().cpu()
    if obs.dtype == torch.uint8:
        return obs.contiguous()
    obs = obs.float()
    if obs.numel() and obs.max() <= 1.5:
        obs = obs * 255.0
    return obs.clamp(0, 255).to(torch.uint8).contiguous()


def _export_offline_dataset(buffer, output_dir, *, include_ram=True, include_state=False):
    output_dir = Path(output_dir).expanduser()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = _ordered_indices(buffer)
    if indices.numel() == 0:
        raise RuntimeError("Cannot export an empty replay buffer.")

    episode_ids = _gather_cpu(buffer.episode_buffer, indices).long()
    obs = _obs_for_export(buffer.obs_buffer.index_select(0, indices))
    action = _gather_cpu(buffer.action_buffer, indices).long()
    reward = _gather_cpu(buffer.reward_buffer, indices).float()
    termination = _gather_cpu(buffer.termination_buffer, indices).float()
    ram = None
    if include_ram and "ram" in buffer.info_buffer:
        ram = _gather_cpu(buffer.info_buffer["ram"], indices).to(torch.uint8)
    state = None
    if include_state and hasattr(buffer, "state_buffer"):
        state = _gather_cpu(buffer.state_buffer, indices).float()

    lengths = []
    for out_id, (_, start, end) in enumerate(_episode_ranges(episode_ids)):
        episode = {
            "obs": obs[start:end],
            "action": action[start:end],
            "reward": reward[start:end],
            "termination": termination[start:end],
            "info": {},
        }
        if state is not None:
            episode["state"] = state[start:end]
        if ram is not None:
            episode["info"]["ram"] = ram[start:end]
        torch.save(episode, output_dir / f"{out_id}.pt")
        lengths.append(int(end - start))

    meta = {
        "format": "oc_storm_offline_episode_v1",
        "num_episodes": len(lengths),
        "num_steps": int(sum(lengths)),
        "lengths": lengths,
        "include_ram": bool(ram is not None),
        "include_state": bool(state is not None),
        "step_semantics": (
            "100k environment/interaction steps are 100k saved "
            "transitions/observations; Atari frame skip 4 advances about "
            "400k raw emulator frames."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class ReplayBuffer:
    def __init__(
        self, obs_shape, action_dim, num_envs, max_length=int(1e6), warmup_length=50000, store_on_gpu=False
    ) -> None:
        self.store_on_gpu = store_on_gpu
        self.action_dim = action_dim

        assert num_envs == 1, "only support num_envs=1"
        assert self.store_on_gpu, "This version forces to store on GPU"

        # if an index is visited outside of current length, the empty may lead to unexpected behavior like idx_dim >= 0 && idx_dim < index_size
        self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.float32, device="cuda")
        self.action_buffer = torch.empty((max_length, action_dim), dtype=torch.int32, device="cuda")
        self.reward_buffer = torch.empty((max_length,), dtype=torch.float32, device="cuda")
        self.termination_buffer = torch.empty((max_length,), dtype=torch.int32, device="cuda")
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None
        self.info_buffer = {}

    def enable_ram(self, ram_shape=(128,)):
        self.info_buffer["ram"] = torch.empty((self.max_length, *ram_shape), dtype=torch.uint8, device="cuda")

    def ready(self):
        return self.length > self.warmup_length

    def sample_indices(self, batch_size, sample_limit):
        # power decay sample >>>
        logits = self.episode_buffer[:sample_limit] - torch.max(self.episode_buffer[:sample_limit])
        prob = torch.exp(logits * torch.log(torch.tensor(1.25)))
        prob = prob / torch.sum(prob)
        # mix uniform sample
        prob = 0.5 * prob + 0.5 / sample_limit
        # <<< power decay sample

        indices = torch.multinomial(prob, batch_size, replacement=True)

        return indices.cpu().numpy()  # otherwise the "for idx in indices" later will be very slow

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)

        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])

        return obs, action, reward, termination

    @torch.no_grad()
    def sample_with_ram(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"
        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])
        if "ram" not in self.info_buffer:
            return obs, action, reward, termination, None
        ram = torch.stack([self.info_buffer["ram"][idx : idx + batch_length] for idx in indices])
        return obs, action, reward, termination, ram

    def append(self, obs, action, reward, termination, episode, info=None, ram=None):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        self.obs_buffer[self.last_pointer] = obs
        self.action_buffer[self.last_pointer] = torch.from_numpy(action).cuda()
        self.reward_buffer[self.last_pointer] = reward
        self.termination_buffer[self.last_pointer] = termination
        self.episode_buffer[self.last_pointer] = episode
        if "ram" in self.info_buffer:
            ram = ram if ram is not None else (info or {}).get("ram")
            if ram is None:
                self.info_buffer["ram"][self.last_pointer].zero_()
            else:
                self.info_buffer["ram"][self.last_pointer] = torch.as_tensor(ram, dtype=torch.uint8, device="cuda")

        if len(self) < self.max_length:
            self.length += 1

    @torch.no_grad()
    def dry_sample(self, batch_size, batch_length):
        """
        For testing only
        """
        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        return indices

    def dry_append(self, episode):
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length

    def export_offline_dataset(self, output_dir, *, include_ram=True):
        return _export_offline_dataset(
            self, output_dir, include_ram=include_ram, include_state=False
        )


class VisualReplayBuffer:
    def __init__(
        self, obs_shape, action_dim, num_envs, max_length=int(1e6), warmup_length=50000, store_on_gpu=False
    ) -> None:
        self.store_on_gpu = store_on_gpu
        self.action_dim = action_dim

        assert num_envs == 1, "only support num_envs=1"
        assert self.store_on_gpu, "This version forces to store on GPU"

        # if an index is visited outside of current length, the empty may lead to unexpected behavior like idx_dim >= 0 && idx_dim < index_size
        self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.uint8, device="cuda")
        self.action_buffer = torch.empty((max_length, action_dim), dtype=torch.int32, device="cuda")
        self.reward_buffer = torch.empty((max_length,), dtype=torch.float32, device="cuda")
        self.termination_buffer = torch.empty((max_length,), dtype=torch.int32, device="cuda")
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None
        self.info_buffer = {}

    def enable_ram(self, ram_shape=(128,)):
        self.info_buffer["ram"] = torch.empty((self.max_length, *ram_shape), dtype=torch.uint8, device="cuda")

    def ready(self):
        return self.length > self.warmup_length

    def sample_indices(self, batch_size, sample_limit):
        # power decay sample >>>
        logits = self.episode_buffer[:sample_limit] - torch.max(self.episode_buffer[:sample_limit])
        prob = torch.exp(logits * torch.log(torch.tensor(1.25)))
        prob = prob / torch.sum(prob)
        # mix uniform sample
        prob = 0.5 * prob + 0.5 / sample_limit
        # <<< power decay sample

        indices = torch.multinomial(prob, batch_size, replacement=True)

        return indices.cpu().numpy()  # otherwise the "for idx in indices" later will be very slow

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)

        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])

        # convert uint8 obs to float32
        obs = obs.to(torch.float32) / 255

        return obs, action, reward, termination

    @torch.no_grad()
    def sample_with_ram(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"
        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])
        obs = obs.to(torch.float32) / 255
        if "ram" not in self.info_buffer:
            return obs, action, reward, termination, None
        ram = torch.stack([self.info_buffer["ram"][idx : idx + batch_length] for idx in indices])
        return obs, action, reward, termination, ram

    def append(self, obs, action, reward, termination, episode, info=None, ram=None):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        # convert float32 obs to uint8
        obs = obs * 255
        obs = obs.to(torch.uint8)

        self.obs_buffer[self.last_pointer] = obs
        self.action_buffer[self.last_pointer] = torch.from_numpy(action).cuda()
        self.reward_buffer[self.last_pointer] = reward
        self.termination_buffer[self.last_pointer] = termination
        self.episode_buffer[self.last_pointer] = episode
        if "ram" in self.info_buffer:
            ram = ram if ram is not None else (info or {}).get("ram")
            if ram is None:
                self.info_buffer["ram"][self.last_pointer].zero_()
            else:
                self.info_buffer["ram"][self.last_pointer] = torch.as_tensor(ram, dtype=torch.uint8, device="cuda")

        if len(self) < self.max_length:
            self.length += 1

    @torch.no_grad()
    def dry_sample(self, batch_size, batch_length):
        """
        For testing only
        """
        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        return indices

    def dry_append(self, episode):
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length

    def export_offline_dataset(self, output_dir, *, include_ram=True):
        return _export_offline_dataset(
            self, output_dir, include_ram=include_ram, include_state=False
        )


class ReplayBufferVectorPlusVisual:
    """
    Cutie object vector + visual observation
    """

    def __init__(
        self, state_shape, obs_shape, action_dim, num_envs, max_length=int(1e6), warmup_length=50000, store_on_gpu=False
    ) -> None:
        self.store_on_gpu = store_on_gpu
        self.action_dim = action_dim

        assert num_envs == 1, "only support num_envs=1"
        assert self.store_on_gpu, "This version forces to store on GPU"

        # if an index is visited outside of current length, the empty may lead to unexpected behavior like idx_dim >= 0 && idx_dim < index_size
        self.state_buffer = torch.empty((max_length, *state_shape), dtype=torch.float32, device="cuda")
        self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.uint8, device="cuda")
        self.action_buffer = torch.empty((max_length, action_dim), dtype=torch.int32, device="cuda")
        self.reward_buffer = torch.empty((max_length,), dtype=torch.float32, device="cuda")
        self.termination_buffer = torch.empty((max_length,), dtype=torch.int32, device="cuda")
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None
        self.info_buffer = {}

    def enable_ram(self, ram_shape=(128,)):
        self.info_buffer["ram"] = torch.empty((self.max_length, *ram_shape), dtype=torch.uint8, device="cuda")

    def ready(self):
        return self.length > self.warmup_length

    def sample_indices(self, batch_size, sample_limit):
        # power decay sample >>>
        logits = self.episode_buffer[:sample_limit] - torch.max(self.episode_buffer[:sample_limit])
        prob = torch.exp(logits * torch.log(torch.tensor(1.25)))
        prob = prob / torch.sum(prob)
        prob = 0.5 * prob + 0.5 / sample_limit  # mix uniform
        # <<< power decay sample

        indices = torch.multinomial(prob, batch_size, replacement=True)

        return indices.cpu().numpy()  # otherwise the "for idx in indices" later will be very slow

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)

        state = torch.stack([self.state_buffer[idx : idx + batch_length] for idx in indices])
        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])

        # convert uint8 obs to float32
        obs = obs.to(torch.float32) / 255

        return state, obs, action, reward, termination

    @torch.no_grad()
    def sample_with_ram(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"
        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        state = torch.stack([self.state_buffer[idx : idx + batch_length] for idx in indices])
        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])
        obs = obs.to(torch.float32) / 255
        if "ram" not in self.info_buffer:
            return state, obs, action, reward, termination, None
        ram = torch.stack([self.info_buffer["ram"][idx : idx + batch_length] for idx in indices])
        return state, obs, action, reward, termination, ram

    def append(self, state_obs, action, reward, termination, episode, info=None, ram=None):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        state, obs = state_obs  # tuple
        # convert float32 obs to uint8
        obs = obs * 255
        obs = obs.to(torch.uint8)

        self.state_buffer[self.last_pointer] = state
        self.obs_buffer[self.last_pointer] = obs
        self.action_buffer[self.last_pointer] = torch.from_numpy(action).cuda()
        self.reward_buffer[self.last_pointer] = reward
        self.termination_buffer[self.last_pointer] = termination
        self.episode_buffer[self.last_pointer] = episode
        if "ram" in self.info_buffer:
            ram = ram if ram is not None else (info or {}).get("ram")
            if ram is None:
                self.info_buffer["ram"][self.last_pointer].zero_()
            else:
                self.info_buffer["ram"][self.last_pointer] = torch.as_tensor(ram, dtype=torch.uint8, device="cuda")

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length

    def export_offline_dataset(self, output_dir, *, include_ram=True):
        return _export_offline_dataset(
            self, output_dir, include_ram=include_ram, include_state=True
        )
