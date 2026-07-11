from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch

from .rewards import quantize_pong_reward


def _validate_initial_source(value: str) -> str:
    source = str(value).lower()
    if source not in {"real", "prior", "dataset"}:
        raise ValueError(f"Unknown wm initial source {value!r}; expected 'real', 'prior', or 'dataset'.")
    return source


class GymAtariPixelEnv:

    def __init__(self, env, num_actions: int):
        self.env = env
        self.num_actions = int(num_actions)

    def reset(self):
        obs, info = self.env.reset()
        return self._obs(obs, reward=0.0, first=True, last=False, terminal=False), info

    def step(self, action: int):
        action = np.asarray([int(action)], dtype=np.int32)
        obs, rew, end, trunc, info = self.env.step(action)
        return (
            self._obs(obs, reward=rew, first=False, last=end or trunc, terminal=end),
            float(rew),
            bool(end),
            bool(trunc),
            info,
        )

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _obs(obs, reward, first, last, terminal):
        frame = cv2.resize(np.asarray(obs), (64, 64), interpolation=cv2.INTER_LINEAR)
        return {
            "image": frame.astype(np.uint8, copy=False),
            "reward": np.float32(reward),
            "is_first": bool(first),
            "is_last": bool(last),
            "is_terminal": bool(terminal),
        }


class VectorPixelEnv:

    def __init__(self, envs, device, raw_reward_threshold=0.5):
        self.envs = list(envs)
        self.device = torch.device(device)
        self.raw_reward_threshold = float(raw_reward_threshold)
        self.num_envs = len(self.envs)
        self.num_actions = self.envs[0].num_actions
        self._completed = []
        self._raw_rewards = []
        self._raw_agent_rewards = []
        self._raw_opponent_rewards = []
        self._episode_scores = [0.0 for _ in self.envs]
        self._episode_agent_scores = [0.0 for _ in self.envs]
        self._episode_opponent_scores = [0.0 for _ in self.envs]
        self._episode_raw_scores = [0.0 for _ in self.envs]
        self._episode_raw_agent_scores = [0.0 for _ in self.envs]
        self._episode_raw_opponent_scores = [0.0 for _ in self.envs]
        self._episode_has_raw_reward = [False for _ in self.envs]
        self._episode_lengths = [0 for _ in self.envs]

    def reset(self):
        obs = []
        infos = []
        for env in self.envs:
            ob, info = env.reset()
            obs.append(ob)
            infos.append(info)
        return self._obs_to_tensor(obs), {"raw": infos}

    def step(self, actions):
        actions = actions.detach().cpu().numpy().astype(np.int32)
        next_obs = []
        rewards = []
        ends = []
        truncs = []
        final_obs = []
        raw_infos = []
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            ob, rew, end, trunc, info = env.step(int(action))
            rewards.append(rew)
            ends.append(end)
            truncs.append(trunc)
            raw_infos.append(info)
            self._record_transition(index, rew, info)
            if end or trunc:
                self._complete_episode(index, ob)
                final_obs.append(ob)
                ob, _ = env.reset()
            next_obs.append(ob)
        info = {"raw": raw_infos}
        if final_obs:
            info["final_observation"] = self._obs_to_tensor(final_obs)
        return (
            self._obs_to_tensor(next_obs),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(ends, dtype=torch.float32, device=self.device),
            torch.as_tensor(truncs, dtype=torch.float32, device=self.device),
            info,
        )

    def pop_episode_stats(self):
        completed = self._completed
        self._completed = []
        return completed

    def pop_raw_reward_stats(self):
        if not self._raw_rewards:
            return {}
        values = np.asarray(self._raw_rewards, dtype=np.float32)
        agent_values = np.asarray(self._raw_agent_rewards, dtype=np.float32)
        opponent_values = np.asarray(self._raw_opponent_rewards, dtype=np.float32)
        self._raw_rewards = []
        self._raw_agent_rewards = []
        self._raw_opponent_rewards = []
        return {
            "raw_reward_mean": float(values.mean()),
            "raw_reward_abs_mean": float(np.abs(values).mean()),
            "raw_reward_nonzero_rate": float((np.abs(values) >= self.raw_reward_threshold).mean()),
            "raw_agent_reward_mean": float(agent_values.mean()),
            "raw_opponent_reward_mean": float(opponent_values.mean()),
            "raw_agent_reward_nonzero_rate": float((values >= self.raw_reward_threshold).mean()),
            "raw_opponent_reward_nonzero_rate": float((values <= -self.raw_reward_threshold).mean()),
        }

    def pop_episode_videos(self, max_videos=1):
        return []

    def close(self):
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _record_transition(self, index, rew, info):
        rew = float(rew)
        self._episode_scores[index] += rew
        agent_score, opponent_score = _pong_score_parts(rew)
        self._episode_agent_scores[index] += agent_score
        self._episode_opponent_scores[index] += opponent_score
        self._episode_lengths[index] += 1
        if isinstance(info, dict) and "raw_reward" in info:
            raw_reward = float(info["raw_reward"])
            raw_agent_reward, raw_opponent_reward = _pong_score_parts(raw_reward)
            self._episode_raw_scores[index] += raw_reward
            self._episode_raw_agent_scores[index] += raw_agent_reward
            self._episode_raw_opponent_scores[index] += raw_opponent_reward
            self._episode_has_raw_reward[index] = True
            self._raw_rewards.append(raw_reward)
            self._raw_agent_rewards.append(raw_agent_reward)
            self._raw_opponent_rewards.append(raw_opponent_reward)

    def _complete_episode(self, index, ob):
        completed = {
            "score": self._episode_scores[index],
            "agent_score": self._episode_agent_scores[index],
            "opponent_score": self._episode_opponent_scores[index],
            "length": self._episode_lengths[index],
        }
        if self._episode_has_raw_reward[index]:
            completed["raw_score"] = self._episode_raw_scores[index]
            completed["raw_agent_score"] = self._episode_raw_agent_scores[index]
            completed["raw_opponent_score"] = self._episode_raw_opponent_scores[index]
        self._completed.append(completed)
        self._episode_scores[index] = 0.0
        self._episode_agent_scores[index] = 0.0
        self._episode_opponent_scores[index] = 0.0
        self._episode_raw_scores[index] = 0.0
        self._episode_raw_agent_scores[index] = 0.0
        self._episode_raw_opponent_scores[index] = 0.0
        self._episode_has_raw_reward[index] = False
        self._episode_lengths[index] = 0

    def _obs_to_tensor(self, obs):
        images = [np.asarray(x["image"]) for x in obs]
        arr = np.stack(images)
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        return tensor.div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()


class QuantizedStormWMEnv:

    def __init__(
        self,
        *,
        config_path,
        env_name,
        seed,
        checkpoint,
        horizon,
        threshold,
        sample_mode="probs",
        disable_kv_cache=True,
        respect_terminal=True,
        initial_source="real",
        bootstrap_dataset=None,
    ):
        from interactive.oc_storm_adapter import build_oc_storm_session

        if not checkpoint:
            raise ValueError("pixel RL backend=wm requires --wm-checkpoint.")
        self.horizon = int(horizon)
        self.threshold = float(threshold)
        self._step = 0
        self.session = build_oc_storm_session(
            config_path=Path(config_path).expanduser().resolve(),
            env_name=env_name,
            seed=int(seed),
            checkpoint_args=[str(Path(checkpoint).expanduser().resolve())],
            wm_name_args=["storm_wm"],
            policy_checkpoint_args=[],
            policy_name_args=[],
            controller="human",
            bootstrap_dataset=bootstrap_dataset,
            wm_sample_mode=sample_mode,
            wm_respect_terminal=respect_terminal,
            wm_disable_kv_cache=disable_kv_cache,
            wm_kv_cache_dtype="fp32",
            wm_initial_source=initial_source,
        )
        if self.session.variant != "visual":
            raise ValueError(
                "RL in WM env is restricted to STORM visual checkpoints. "
                f"Got variant={self.session.variant!r}; OC-STORM/vector variants are not supported here."
            )
        self.session.current_backend_index = 1
        self.session.set_horizon(self.horizon)
        self.num_actions = len(self.session.action_names)

    def reset(self):
        self._step = 0
        self.session.current_backend_index = 1
        self.session.reset()
        return self._obs(self.session.current_obs, 0.0, True, False, False), {}

    def step(self, action: int):
        result = self.session.step(int(action))
        self._step += 1
        raw_reward = float(result.reward)
        reward = quantize_pong_reward(raw_reward, self.threshold)
        done = bool(result.done)
        trunc = bool(result.trunc or self._step >= self.horizon)
        obs = self._obs(result.obs, reward, False, done or trunc, done)
        info = dict(result.info or {})
        info["raw_reward"] = raw_reward
        info["trunc_predicted"] = trunc
        return obs, reward, done, trunc, info

    def close(self):
        close = getattr(self.session.env, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _obs(frame, reward, first, last, terminal):
        return {
            "image": np.asarray(frame).astype(np.uint8, copy=False),
            "reward": np.float32(reward),
            "is_first": bool(first),
            "is_last": bool(last),
            "is_terminal": bool(terminal),
        }


class BatchedStormWMPixelEnv:

    def __init__(
        self,
        *,
        config_path,
        env_name,
        seed,
        checkpoint,
        horizon,
        num_envs,
        device,
        reward_threshold,
        sample_mode="probs",
        disable_kv_cache=True,
        respect_terminal=True,
        initial_source="real",
        bootstrap_dataset=None,
    ):
        if not disable_kv_cache:
            raise ValueError(
                "RL in STORM WM requires --wm-disable-kv-cache=True. "
                "Long RL rollouts exceed the training context length and this "
                "project does not have a robust over-context KV-cache policy."
            )
        self.device = torch.device(device)
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.threshold = float(reward_threshold)
        self.sample_mode = str(sample_mode)
        self.respect_terminal = bool(respect_terminal)
        self.initial_source = _validate_initial_source(initial_source)
        self.env_name = env_name
        self.seed = int(seed)
        self._step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        module = _import_config_module(Path(config_path).expanduser().resolve())
        build_offline = getattr(module, "build_offline", None)
        if not callable(build_offline):
            raise ValueError("RL in STORM WM requires a config with build_offline().")
        params, agent = build_offline(env_name=env_name, seed=seed)
        if "visual" not in agent.__class__.__module__ or "vector" in agent.__class__.__module__:
            raise ValueError("RL in WM env is restricted to STORM visual checkpoints.")
        state_dict = _load_state_dict(Path(checkpoint).expanduser().resolve())
        agent.load_state_dict(state_dict)
        agent.eval()
        self.params = params
        self.agent = agent.to(self.device)
        self.num_actions = int(agent.action_dims[0])
        self.context_frames = max(1, int(getattr(params, "imagine_context_length", 4)))
        self.max_context = max(1, int(getattr(params.world_model, "transformer_max_length", 64)) - 1)
        self.bootstrap_dataset = None
        if self.initial_source == "dataset":
            if not bootstrap_dataset:
                raise ValueError("STORM dataset initial source requires --bootstrap-dataset.")
            from interactive.oc_storm_adapter import _OfflineVisualBootstrapDataset

            input_channels = int(agent.world_model.encoder.backbone[0].in_channels)
            self.bootstrap_dataset = _OfflineVisualBootstrapDataset(
                bootstrap_dataset,
                self.device,
                input_channels,
            )

        self.real_envs = []
        if self.initial_source == "real":
            from envs.atari.build_env import build_single_atari_env

            self.real_envs = [
                build_single_atari_env(env_name, self.seed + index)[0]
                for index in range(self.num_envs)
            ]
        self._completed = []
        self._raw_rewards = []
        self._raw_agent_rewards = []
        self._raw_opponent_rewards = []
        self._episode_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_agent_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_opponent_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_raw_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_raw_agent_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_raw_opponent_scores = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_has_raw_reward = np.zeros(self.num_envs, dtype=bool)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.current_latent = None
        self.current_hidden = None
        self.current_frames = None
        self.latent_hist = []
        self.action_hist = []

    def reset(self):
        self._reset_indices(torch.arange(self.num_envs, device=self.device))
        return self._frames_to_obs_tensor(self.current_frames), {}

    def step(self, actions):
        actions = actions.to(device=self.device, dtype=torch.int32).reshape(self.num_envs, 1, 1)
        with torch.no_grad(), self._autocast():
            latent_seq = torch.cat(self.latent_hist + [self.current_latent], dim=1)
            action_seq = torch.cat(self.action_hist + [actions], dim=1)
            mask = self._causal_mask(latent_seq)
            dist_feat = self.agent.world_model.storm_transformer(latent_seq, action_seq, mask)
            dist_feat = dist_feat[:, -1:]
            prior_logits = self.agent.world_model.dist_head.forward_prior(dist_feat)
            prior_sample = self.agent.world_model.straight_through_gradient(
                prior_logits, sample_mode=self.sample_mode)
            next_latent = self.agent.world_model.flatten_sample(prior_sample)
            obs_hat = self.agent.world_model.state_decoder(next_latent)
            reward_hat = self.agent.world_model.symlog_twohot_loss_func.decode(
                self.agent.world_model.reward_decoder(dist_feat))
            term_hat = self.agent.world_model.termination_decoder(dist_feat) > 0

        raw_rewards = reward_hat.reshape(-1).float()
        rewards = torch.zeros_like(raw_rewards)
        rewards[raw_rewards >= self.threshold] = 1.0
        rewards[raw_rewards <= -self.threshold] = -1.0
        predicted_terms = term_hat.reshape(-1).to(dtype=torch.bool)
        ends = predicted_terms.to(dtype=torch.float32) if self.respect_terminal else torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device)
        self._step += 1
        truncs = (self._step >= self.horizon).to(dtype=torch.float32)
        frames = _tensor_to_uint8_frames(obs_hat)

        self._record_batch(rewards.detach().cpu().numpy(), raw_rewards.detach().cpu().numpy())
        self.latent_hist.append(self.current_latent.detach())
        self.action_hist.append(actions.detach())
        if len(self.latent_hist) > self.max_context - 1:
            self.latent_hist = self.latent_hist[-(self.max_context - 1):]
            self.action_hist = self.action_hist[-(self.max_context - 1):]
        self.current_latent = next_latent.detach()
        self.current_hidden = dist_feat.detach()
        self.current_frames = frames

        info = {"raw": [
            {
                "terminal_predicted": bool(value),
                "trunc_predicted": bool(trunc_value),
                "terminal_ignored": bool(value and not self.respect_terminal),
            }
            for value, trunc_value in zip(
                predicted_terms.detach().cpu().numpy().tolist(),
                truncs.bool().detach().cpu().numpy().tolist(),
            )
        ]}
        dead = torch.logical_or(ends.bool(), truncs.bool())
        if dead.any():
            dead_indices = dead.nonzero(as_tuple=False).flatten()
            final_frames = self.current_frames[dead_indices.detach().cpu().numpy()]
            info["final_observation"] = self._frames_to_obs_tensor(final_frames)
            for index in dead_indices.detach().cpu().numpy().tolist():
                self._complete_episode(index)
            self._reset_indices(dead_indices)

        return (
            self._frames_to_obs_tensor(self.current_frames),
            rewards,
            ends,
            truncs,
            info,
        )

    def pop_episode_stats(self):
        completed = self._completed
        self._completed = []
        return completed

    def pop_episode_videos(self, max_videos=1):
        return []

    def pop_raw_reward_stats(self):
        if not self._raw_rewards:
            return {}
        values = np.asarray(self._raw_rewards, dtype=np.float32)
        agent_values = np.asarray(self._raw_agent_rewards, dtype=np.float32)
        opponent_values = np.asarray(self._raw_opponent_rewards, dtype=np.float32)
        self._raw_rewards = []
        self._raw_agent_rewards = []
        self._raw_opponent_rewards = []
        return {
            "raw_reward_mean": float(values.mean()),
            "raw_reward_abs_mean": float(np.abs(values).mean()),
            "raw_reward_nonzero_rate": float((np.abs(values) >= self.threshold).mean()),
            "raw_agent_reward_mean": float(agent_values.mean()),
            "raw_opponent_reward_mean": float(opponent_values.mean()),
            "raw_agent_reward_nonzero_rate": float((values >= self.threshold).mean()),
            "raw_opponent_reward_nonzero_rate": float((values <= -self.threshold).mean()),
        }

    def close(self):
        for env in self.real_envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _reset_indices(self, indices):
        if self.initial_source == "prior":
            self._reset_indices_from_prior(indices)
            return
        if self.initial_source == "dataset":
            self._reset_indices_from_dataset(indices)
            return
        self._reset_indices_from_real(indices)

    def _reset_indices_from_prior(self, indices):
        indices_cpu = indices.detach().cpu().numpy().astype(np.int64)
        batch_size = len(indices_cpu)
        zero_latent = self._zero_latent(batch_size)
        noop = torch.zeros(batch_size, 1, 1, dtype=torch.int32, device=self.device)
        with torch.no_grad(), self._autocast():
            dist_feat = self.agent.world_model.storm_transformer(zero_latent, noop, self._causal_mask(zero_latent))
            dist_feat = dist_feat[:, -1:]
            prior_logits = self.agent.world_model.dist_head.forward_prior(dist_feat)
            prior_sample = self.agent.world_model.straight_through_gradient(
                prior_logits, sample_mode=self.sample_mode)
            latent = self.agent.world_model.flatten_sample(prior_sample)
            obs_hat = self.agent.world_model.state_decoder(latent)
        frames = _tensor_to_uint8_frames(obs_hat)
        if self.current_frames is None:
            self.current_frames = np.zeros((self.num_envs, 64, 64, 3), dtype=np.uint8)
            self.current_latent = torch.zeros(
                self.num_envs, 1, latent.shape[-1], dtype=latent.dtype, device=self.device)
            self.latent_hist = []
            self.action_hist = []
        self.current_frames[indices_cpu] = frames
        self.current_latent[indices] = latent.detach()
        for hist_latent in self.latent_hist:
            hist_latent[indices] = 0.0
        for hist_action in self.action_hist:
            hist_action[indices] = 0
        self._step[indices] = 0

    def _reset_indices_from_real(self, indices):
        indices_cpu = indices.detach().cpu().numpy().astype(np.int64)
        frames = []
        actions = []
        for index in indices_cpu:
            env = self.real_envs[int(index)]
            raw_obs, _ = env.reset()
            seq = [_resize_frame(raw_obs)]
            act_seq = []
            noop = np.array([0], dtype=np.int32)
            while len(seq) < self.context_frames:
                raw_obs, _rew, end, trunc, _info = env.step(noop)
                if end or trunc:
                    raw_obs, _ = env.reset()
                    seq = [_resize_frame(raw_obs)]
                    act_seq = []
                    continue
                seq.append(_resize_frame(raw_obs))
                act_seq.append(noop.copy())
            frames.append(np.stack(seq))
            actions.append(np.stack(act_seq) if act_seq else np.zeros((0, 1), dtype=np.int32))

        obs_seq = torch.as_tensor(np.stack(frames), dtype=torch.float32, device=self.device)
        obs_seq = obs_seq.div(255).permute(0, 1, 4, 2, 3).contiguous()
        with torch.no_grad():
            latent = self.agent.world_model.encode_obs(obs_seq, sample_mode=self.sample_mode)

        if self.current_frames is None:
            self.current_frames = np.zeros((self.num_envs, 64, 64, 3), dtype=np.uint8)
            self.current_latent = torch.zeros(
                self.num_envs, 1, latent.shape[-1], dtype=latent.dtype, device=self.device)
            self.latent_hist = []
            self.action_hist = []

        self.current_frames[indices_cpu] = frames_to_current(frames)
        self.current_latent[indices] = latent[:, -1:]
        self._step[indices] = 0

        if not self.latent_hist:
            hist_len = max(0, latent.shape[1] - 1)
            self.latent_hist = [
                torch.zeros(self.num_envs, 1, latent.shape[-1], dtype=latent.dtype, device=self.device)
                for _ in range(hist_len)
            ]
            self.action_hist = [
                torch.zeros(self.num_envs, 1, 1, dtype=torch.int32, device=self.device)
                for _ in range(hist_len)
            ]
        for pos in range(min(len(self.latent_hist), latent.shape[1] - 1)):
            self.latent_hist[pos][indices] = latent[:, pos:pos + 1]
            action_values = torch.as_tensor(
                np.stack([x[pos] for x in actions]), dtype=torch.int32, device=self.device).reshape(-1, 1, 1)
            self.action_hist[pos][indices] = action_values

    def _reset_indices_from_dataset(self, indices):
        if self.bootstrap_dataset is None:
            raise RuntimeError("STORM bootstrap dataset is not initialized.")
        indices_cpu = indices.detach().cpu().numpy().astype(np.int64)
        states = []
        frames = []
        actions = []
        for _index in indices_cpu:
            state_seq, action_seq, frame = self.bootstrap_dataset.sample(self.context_frames)
            states.append(state_seq)
            frames.append(frame)
            actions.append(action_seq.detach().cpu().numpy().astype(np.int32))

        obs_seq = torch.stack(states, dim=0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            latent = self.agent.world_model.encode_obs(obs_seq, sample_mode=self.sample_mode)

        if self.current_frames is None:
            self.current_frames = np.zeros((self.num_envs, 64, 64, 3), dtype=np.uint8)
            self.current_latent = torch.zeros(
                self.num_envs, 1, latent.shape[-1], dtype=latent.dtype, device=self.device)
            self.latent_hist = []
            self.action_hist = []

        self.current_frames[indices_cpu] = np.stack(frames).astype(np.uint8, copy=False)
        self.current_latent[indices] = latent[:, -1:]
        self._step[indices] = 0

        if not self.latent_hist:
            hist_len = max(0, latent.shape[1] - 1)
            self.latent_hist = [
                torch.zeros(self.num_envs, 1, latent.shape[-1], dtype=latent.dtype, device=self.device)
                for _ in range(hist_len)
            ]
            self.action_hist = [
                torch.zeros(self.num_envs, 1, 1, dtype=torch.int32, device=self.device)
                for _ in range(hist_len)
            ]
        for pos in range(min(len(self.latent_hist), latent.shape[1] - 1)):
            self.latent_hist[pos][indices] = latent[:, pos:pos + 1]
            if actions and pos < actions[0].shape[0]:
                action_values = torch.as_tensor(
                    np.stack([x[pos] for x in actions]), dtype=torch.int32, device=self.device).reshape(-1, 1, 1)
            else:
                action_values = torch.zeros(indices.numel(), 1, 1, dtype=torch.int32, device=self.device)
            self.action_hist[pos][indices] = action_values

    def _zero_latent(self, batch_size):
        stoch_dim = int(getattr(self.agent.world_model.dist_head, "stoch_dim", 32))
        return torch.zeros(
            int(batch_size),
            1,
            stoch_dim * stoch_dim,
            dtype=torch.float32,
            device=self.device,
        )

    def _record_batch(self, rewards, raw_rewards):
        self._episode_scores += rewards
        self._episode_agent_scores += np.maximum(rewards, 0.0)
        self._episode_opponent_scores += np.maximum(-rewards, 0.0)
        self._episode_lengths += 1
        self._episode_raw_scores += raw_rewards
        self._episode_raw_agent_scores += np.maximum(raw_rewards, 0.0)
        self._episode_raw_opponent_scores += np.maximum(-raw_rewards, 0.0)
        self._episode_has_raw_reward[:] = True
        self._raw_rewards.extend(raw_rewards.tolist())
        self._raw_agent_rewards.extend(np.maximum(raw_rewards, 0.0).tolist())
        self._raw_opponent_rewards.extend(np.maximum(-raw_rewards, 0.0).tolist())

    def _complete_episode(self, index):
        completed = {
            "score": float(self._episode_scores[index]),
            "agent_score": float(self._episode_agent_scores[index]),
            "opponent_score": float(self._episode_opponent_scores[index]),
            "length": int(self._episode_lengths[index]),
        }
        if self._episode_has_raw_reward[index]:
            completed["raw_score"] = float(self._episode_raw_scores[index])
            completed["raw_agent_score"] = float(self._episode_raw_agent_scores[index])
            completed["raw_opponent_score"] = float(self._episode_raw_opponent_scores[index])
        self._completed.append(completed)
        self._episode_scores[index] = 0.0
        self._episode_agent_scores[index] = 0.0
        self._episode_opponent_scores[index] = 0.0
        self._episode_raw_scores[index] = 0.0
        self._episode_raw_agent_scores[index] = 0.0
        self._episode_raw_opponent_scores[index] = 0.0
        self._episode_has_raw_reward[index] = False
        self._episode_lengths[index] = 0

    def _frames_to_obs_tensor(self, frames):
        tensor = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        return tensor.div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()

    def _autocast(self):
        wm = self.agent.world_model
        enabled = bool(getattr(wm, "use_amp", False)) and self.device.type == "cuda"
        dtype = getattr(wm, "amp_tensor_dtype", torch.bfloat16)
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)

    def _causal_mask(self, latent):
        module = importlib.import_module(self.agent.world_model.__class__.__module__)
        return module.get_causal_mask(latent)


def _pong_score_parts(reward):
    reward = float(reward)
    return max(reward, 0.0), max(-reward, 0.0)


def make_real_pixel_envs(env_name, seed, num_envs, device):
    from envs.atari.build_env import build_single_atari_env

    envs = []
    for index in range(num_envs):
        env, action_space = build_single_atari_env(env_name, int(seed) + index)
        envs.append(GymAtariPixelEnv(env, action_space.choices_per_dim))
    return VectorPixelEnv(envs, device)


def make_wm_pixel_envs(
    *,
    config_path,
    env_name,
    seed,
    checkpoint,
    horizon,
    num_envs,
    device,
    reward_threshold,
    sample_mode,
    disable_kv_cache,
    respect_terminal,
    initial_source="real",
    bootstrap_dataset=None,
):
    if int(num_envs) > 1:
        return BatchedStormWMPixelEnv(
            config_path=config_path,
            env_name=env_name,
            seed=seed,
            checkpoint=checkpoint,
            horizon=horizon,
            num_envs=num_envs,
            device=device,
            reward_threshold=reward_threshold,
            sample_mode=sample_mode,
            disable_kv_cache=disable_kv_cache,
            respect_terminal=respect_terminal,
            initial_source=initial_source,
            bootstrap_dataset=bootstrap_dataset,
        )
    envs = [
        QuantizedStormWMEnv(
            config_path=config_path,
            env_name=env_name,
            seed=int(seed) + index,
            checkpoint=checkpoint,
            horizon=horizon,
            threshold=reward_threshold,
            sample_mode=sample_mode,
            disable_kv_cache=disable_kv_cache,
            respect_terminal=respect_terminal,
            initial_source=initial_source,
            bootstrap_dataset=bootstrap_dataset,
        )
        for index in range(num_envs)
    ]
    return VectorPixelEnv(envs, device, raw_reward_threshold=reward_threshold)


def _import_config_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_state_dict(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resize_frame(frame):
    return cv2.resize(np.asarray(frame), (64, 64), interpolation=cv2.INTER_LINEAR).astype(np.uint8, copy=False)


def _tensor_to_uint8_frames(tensor):
    if tensor.ndim == 5 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    tensor = tensor.detach().float().clamp(0, 1).mul(255).to(torch.uint8)
    return tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def frames_to_current(frames):
    return np.stack([x[-1] for x in frames]).astype(np.uint8, copy=False)
