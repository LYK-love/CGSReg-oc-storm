import argparse
import importlib
import os
import shutil
import warnings
from pathlib import Path

import colorama
import numpy as np
import torch
from tqdm import tqdm

from utils import tools


class PrefixedLogger:
    def __init__(self, logger, prefix: str):
        self.logger = logger
        self.prefix = prefix.strip("/")

    def log(self, tag: str, value):
        self.logger.log(f"{self.prefix}/{tag}", value)


def weight_slug(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def mask_weights_from_preset(mask_preset: str) -> dict[str, float]:
    if mask_preset in {"mask1", "ball", "pong"}:
        return {"mask1": 1.0, "mask2": 0.0, "mask3": 0.0}
    if mask_preset in {"mask1_mask3", "ball_player", "pong_player"}:
        return {"mask1": 1.0, "mask2": 0.0, "mask3": 1.0}
    raise ValueError(f"Unknown mask preset: {mask_preset}")


def _load_file(path: Path) -> dict:
    if path.suffix == ".npz":
        data = np.load(path)
        return {k: torch.as_tensor(data[k]) for k in data.files}
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        return loaded
    raise ValueError(f"Unsupported dataset file format in {path}: expected dict-like data.")


def _first_present(data: dict, names: tuple[str, ...]):
    for name in names:
        if name in data:
            return data[name]
    return None


def _as_float01(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.dtype == torch.uint8:
        return x.float() / 255.0
    x = x.float()
    if x.numel() and x.min() < 0:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def _as_image_storage(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.dtype == torch.uint8:
        return x.contiguous()
    return _as_float01(x).contiguous()


def _as_mask_storage(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.dtype in {torch.bool, torch.uint8}:
        return x.to(torch.uint8).contiguous()
    return x.float().contiguous()


def _as_action(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x).long()
    if x.ndim == 1:
        x = x[:, None]
    return x.int()


def _as_vector_obs(x: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x).float()


def _normalize_episode(data: dict) -> dict:
    obs = _first_present(data, ("obs", "image", "frames", "observation"))
    action = _first_present(data, ("action", "act", "actions"))
    reward = _first_present(data, ("reward", "rew", "rewards"))
    termination = _first_present(data, ("termination", "terminal", "done", "is_terminal", "dones"))
    if obs is None or action is None or reward is None or termination is None:
        raise ValueError("Offline episodes must contain obs/image, action, reward, and termination keys.")

    episode = {
        "obs": _as_image_storage(obs),
        "action": _as_action(action),
        "reward": torch.as_tensor(reward).float(),
        "termination": torch.as_tensor(termination).float(),
    }

    state = _first_present(data, ("state", "object_state", "vector", "vectors"))
    if state is not None:
        episode["state"] = _as_vector_obs(state)

    info = data.get("info", {})
    if info is not None and "ram" in info:
        episode["info"] = {"ram": torch.as_tensor(info["ram"]).to(torch.uint8)}

    masks = _first_present(data, ("masks", "mask"))
    if masks is not None:
        episode["masks"] = _as_mask_storage(masks)
    else:
        mask_list = []
        for key in ("mask1", "mask2", "mask3"):
            value = _first_present(data, (key,))
            if value is not None:
                mask_list.append(_as_mask_storage(value))
        if mask_list:
            mask_list = [m[:, None] if m.ndim == 3 else m for m in mask_list]
            episode["masks"] = torch.cat(mask_list, dim=1)

    lengths = [v.shape[0] for v in episode.values() if torch.is_tensor(v)]
    if "info" in episode and "ram" in episode["info"]:
        lengths.append(episode["info"]["ram"].shape[0])
    length = min(lengths)
    if "info" in episode and "ram" in episode["info"]:
        episode["info"]["ram"] = episode["info"]["ram"][:length]
    return {k: v[:length] if torch.is_tensor(v) else v for k, v in episode.items()}


class StaticSequenceDataset:
    def __init__(self, root: str, batch_length: int, device: str = "cuda"):
        root_path = Path(root).expanduser()
        if root_path.is_file():
            files = [root_path]
        else:
            files = sorted(
                p
                for suffix in ("*.pt", "*.pth", "*.npz")
                for p in root_path.rglob(suffix)
                if p.name != "info.pt"
            )
        if not files:
            raise FileNotFoundError(f"No .pt/.pth/.npz offline episodes found under {root_path}")

        self.episodes = []
        for path in files:
            episode = _normalize_episode(_load_file(path))
            if episode["obs"].shape[0] >= batch_length:
                self.episodes.append(episode)
        if not self.episodes:
            raise ValueError(f"No episode under {root_path} is long enough for batch_length={batch_length}")
        self.batch_length = batch_length
        self.keys = sorted(set().union(*(episode.keys() for episode in self.episodes)) - {"info"})
        if any("info" in episode and "ram" in episode["info"] for episode in self.episodes):
            self.keys.append("info/ram")
        self.device = self._resolve_device(device)
        self._move_to_device()
        self.offsets = torch.arange(self.batch_length, device=self.device)
        total_frames = sum(episode["obs"].shape[0] for episode in self.episodes)
        total_bytes = sum(self._episode_nbytes(episode) for episode in self.episodes)
        print(
            colorama.Fore.GREEN
            + f"Loaded {len(self.episodes)} offline episodes from {root_path} "
            + f"({total_frames} frames, {total_bytes / 1024**3:.2f} GiB tensors) on {self.device}"
            + colorama.Style.RESET_ALL
        )

    @staticmethod
    def _episode_nbytes(episode: dict) -> int:
        total = sum(v.numel() * v.element_size() for v in episode.values() if torch.is_tensor(v))
        if "info" in episode and "ram" in episode["info"]:
            total += episode["info"]["ram"].numel() * episode["info"]["ram"].element_size()
        return total

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    def _move_to_device(self):
        if self.device.type == "cpu":
            return
        try:
            for episode in self.episodes:
                for key, value in list(episode.items()):
                    if torch.is_tensor(value):
                        episode[key] = value.to(self.device, non_blocking=True)
                if "info" in episode and "ram" in episode["info"]:
                    episode["info"]["ram"] = episode["info"]["ram"].to(self.device, non_blocking=True)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            warnings.warn("CUDA OOM while preloading offline dataset; falling back to CPU-resident sampling.")
            self.device = torch.device("cpu")
            torch.cuda.empty_cache()
            for episode in self.episodes:
                for key, value in list(episode.items()):
                    if torch.is_tensor(value):
                        episode[key] = value.cpu()
                if "info" in episode and "ram" in episode["info"]:
                    episode["info"]["ram"] = episode["info"]["ram"].cpu()

    def _episode_value(self, episode: dict, key: str):
        if key == "info/ram":
            if "info" in episode and "ram" in episode["info"]:
                return episode["info"]["ram"]
            return None
        return episode.get(key)

    @staticmethod
    def _prepare_batch_value(key: str, value: torch.Tensor) -> torch.Tensor:
        if value.device.type == "cpu":
            value = value.cuda(non_blocking=True)
        if key == "obs" and value.dtype == torch.uint8:
            return value.float().div_(255.0)
        if key == "masks" and not value.dtype.is_floating_point:
            return value.float()
        return value

    def sample(self, batch_size: int) -> dict[str, torch.Tensor | None]:
        episode_indices = torch.randint(len(self.episodes), (batch_size,), device=self.device)
        groups = []
        for episode_idx in episode_indices.unique(sorted=False).tolist():
            selected = episode_indices == episode_idx
            count = int(selected.sum().item())
            episode = self.episodes[episode_idx]
            starts = torch.randint(
                episode["obs"].shape[0] - self.batch_length + 1,
                (count,),
                device=episode["obs"].device,
            )
            groups.append((episode_idx, starts))

        batch: dict[str, torch.Tensor | None | dict] = {}
        for key in self.keys:
            pieces = []
            piece_count = 0
            for episode_idx, starts in groups:
                episode = self.episodes[episode_idx]
                value = self._episode_value(episode, key)
                if value is None:
                    continue
                indices = starts[:, None] + self.offsets.to(value.device)[None, :]
                pieces.append(value[indices])
                piece_count += starts.shape[0]
            if piece_count != batch_size:
                if key == "info/ram":
                    batch.setdefault("info", {})["ram"] = None
                else:
                    batch[key] = None
                continue
            value = self._prepare_batch_value(key, torch.cat(pieces, dim=0))
            if key == "info/ram":
                batch.setdefault("info", {})["ram"] = value
            else:
                batch[key] = value
        return batch


def run_name_from_args(args) -> str:
    if args.run_name:
        return args.run_name
    size_label = "base" if args.size_config in {"", "base", "default"} else args.size_config
    return (
        f"{args.run_prefix}_{size_label}_{args.mask_preset}_spatial_"
        f"{weight_slug(args.spatial_regu_weight)}_temporal_{weight_slug(args.temporal_regu_weight)}"
    )


def latest_step_checkpoint(ckpt_dir: Path) -> tuple[Path | None, int]:
    best_path = None
    best_step = 0
    for path in ckpt_dir.glob("agent_step_*.pth"):
        try:
            step = int(path.stem.removeprefix("agent_step_"))
        except ValueError:
            continue
        if step > best_step:
            best_path = path
            best_step = step
    return best_path, best_step


def _maybe_attach_masks_for_visual_agent(agent, batch: dict[str, torch.Tensor | None], *, legacy_mask_input: bool = False):
    if not legacy_mask_input:
        return
    if batch.get("masks") is None or "state" in batch:
        return
    world_model = agent.world_model
    if not hasattr(world_model, "encoder"):
        return
    expected_channels = world_model.encoder.backbone[0].in_channels
    obs = batch["obs"]
    if obs is not None and obs.shape[2] < expected_channels:
        needed = expected_channels - obs.shape[2]
        batch["obs"] = torch.cat([obs, batch["masks"][:, :, :needed]], dim=2)


@torch.no_grad()
def log_offline_world_model_metrics(
    agent,
    dataset,
    batch_size: int,
    logger,
    prefix: str,
    *,
    legacy_mask_input: bool = False,
):
    batch = dataset.sample(batch_size)
    _maybe_attach_masks_for_visual_agent(agent, batch, legacy_mask_input=legacy_mask_input)
    if "state" in batch and batch["state"] is not None:
        agent.world_model.log_eval_metrics(
            batch["state"],
            batch["obs"],
            batch["action"],
            batch["reward"],
            batch["termination"],
            logger,
            prefix=prefix,
            masks=batch.get("masks"),
        )
    else:
        agent.world_model.log_eval_metrics(
            batch["obs"],
            batch["action"],
            batch["reward"],
            batch["termination"],
            logger,
            prefix=prefix,
            masks=batch.get("masks"),
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--run_prefix", type=str, default="pong_wm_reg_image_sweep/logdir/pong_wm_reg")
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--size-config", "--size_config", dest="size_config", type=str, default="base")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--train_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_length", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default="",
        help="Static held-out offline split for world-model evaluation.",
    )
    parser.add_argument("--eval_every", type=int, default=0, help="Evaluate every N train steps. Defaults to save_every.")
    parser.add_argument("--eval_batches", type=int, default=0, help="Number of held-out batches to log per eval.")
    parser.add_argument("--eval_batch_size", type=int, default=0, help="Defaults to batch_size.")
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Log detailed train metrics every N train steps. Use 1 for per-step logging.",
    )
    parser.add_argument("--spatial_regu_weight", type=float, default=0.0)
    parser.add_argument("--temporal_regu_weight", type=float, default=1.0)
    parser.add_argument("--mask_preset", type=str, default="mask1")
    parser.add_argument(
        "--legacy_mask_input",
        action="store_true",
        help="Legacy compatibility: concatenate dataset masks into visual observations for old RGB+mask configs.",
    )
    parser.add_argument("--no_resume", action="store_true", help="Start from scratch even if run_name checkpoints exist.")
    args = parser.parse_args()
    args.run_name = run_name_from_args(args)
    if args.eval_every <= 0:
        args.eval_every = args.save_every
    if args.eval_batch_size <= 0:
        args.eval_batch_size = args.batch_size
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    tools.seed_np_torch(seed=args.seed)
    logger = tools.Logger(
        run_name=args.run_name,
        config={
            "entrypoint": "train_offline.py",
            "env_name": args.env_name,
            "seed": args.seed,
            "config_name": args.config_name,
            "size_config": args.size_config,
            "dataset_path": args.dataset_path,
            "eval_dataset_path": args.eval_dataset_path,
            "eval_every": args.eval_every,
            "eval_batches": args.eval_batches,
            "eval_batch_size": args.eval_batch_size,
            "log_every": args.log_every,
            "spatial_regu_weight": args.spatial_regu_weight,
            "temporal_regu_weight": args.temporal_regu_weight,
            "mask_preset": args.mask_preset,
            "legacy_mask_input": args.legacy_mask_input,
        },
    )
    run_dir = Path("runs") / args.run_name
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    os.environ["STORM_SIZE_CONFIG"] = args.size_config

    config_module = importlib.import_module(f"configs.{args.config_name}")
    build_offline = getattr(config_module, "build_offline", None)
    build = getattr(config_module, "build")
    print(colorama.Fore.RED + f"Using {args.config_name}.py" + colorama.Style.RESET_ALL)
    shutil.copy(f"configs/{args.config_name}.py", f"runs/{args.run_name}/config.py")

    if build_offline is not None:
        params, agent = build_offline(args.env_name, args.seed)
    else:
        params, env, action_space, feature_extractor, replay_buffer, agent = build(args.env_name, args.seed)
        if hasattr(env, "close"):
            env.close()
        if hasattr(feature_extractor, "reset"):
            feature_extractor.reset()
    if not args.legacy_mask_input:
        world_model = agent.world_model
        if hasattr(world_model, "encoder"):
            expected_channels = world_model.encoder.backbone[0].in_channels
            if expected_channels > params.spatial_regu.image_channels:
                raise ValueError(
                    f"{args.config_name} expects {expected_channels} input channels. "
                    "Spatial-reg STORM follows the Diamond/Dreamer convention: "
                    "the world model input must be RGB-only and masks are passed only to the loss. "
                    "Use --config_name atari_visual for new runs, or pass --legacy_mask_input only "
                    "to reproduce old RGB+mask-input checkpoints."
                )
    params.spatial_regu.enabled = True
    params.spatial_regu.weight = args.spatial_regu_weight
    params.spatial_regu.mask_weights = mask_weights_from_preset(args.mask_preset)
    logger.log("config/size_config_index", {"base": 0, "size200m": 1, "size400m": 2, "size800m": 3}.get(args.size_config, -1))

    start_step = 0
    resume_path, resume_step = latest_step_checkpoint(ckpt_dir)
    if not args.no_resume and resume_path is not None:
        agent.load_state_dict(torch.load(resume_path, map_location="cpu"))
        start_step = resume_step
        print(colorama.Fore.GREEN + f"Resumed from {resume_path} at step {start_step}" + colorama.Style.RESET_ALL)
    elif args.no_resume:
        print(colorama.Fore.YELLOW + "Resume disabled; starting from scratch." + colorama.Style.RESET_ALL)

    dataset = StaticSequenceDataset(args.dataset_path, args.batch_length)
    eval_dataset = None
    eval_dataset_path = Path(args.eval_dataset_path).expanduser() if args.eval_dataset_path else None
    if args.eval_batches > 0 and eval_dataset_path is None:
        raise ValueError("--eval_batches > 0 requires --eval_dataset_path with a static held-out split.")
    if args.eval_batches > 0:
        print(colorama.Fore.GREEN + f"Using offline eval dataset: {eval_dataset_path}" + colorama.Style.RESET_ALL)
        eval_dataset = StaticSequenceDataset(str(eval_dataset_path), args.batch_length)

    for step in tqdm(range(start_step + 1, args.train_steps + 1)):
        batch = dataset.sample(args.batch_size)
        _maybe_attach_masks_for_visual_agent(agent, batch, legacy_mask_input=args.legacy_mask_input)
        should_log_train = args.log_every > 0 and (
            step == start_step + 1 or step % args.log_every == 0 or step == args.train_steps
        )
        update_logger = PrefixedLogger(logger, "train") if should_log_train else None
        if "state" in batch and batch["state"] is not None:
            agent.world_model.update(
                batch["state"],
                batch["obs"],
                batch["action"],
                batch["reward"],
                batch["termination"],
                logger=update_logger,
                masks=batch.get("masks"),
            )
        else:
            agent.world_model.update(
                batch["obs"],
                batch["action"],
                batch["reward"],
                batch["termination"],
                logger=update_logger,
                masks=batch.get("masks"),
            )

        if should_log_train:
            logger.log("offline/train_step", step)
            logger.log("train/step", step)
        if step % args.save_every == 0:
            torch.save(agent.state_dict(), f"runs/{args.run_name}/ckpt/agent_step_{step}.pth")
            torch.save(agent.state_dict(), f"runs/{args.run_name}/ckpt/latest_agent.pth")
        if eval_dataset is not None and step % args.eval_every == 0:
            for _ in range(args.eval_batches):
                log_offline_world_model_metrics(
                    agent,
                    eval_dataset,
                    args.eval_batch_size,
                    logger,
                    prefix="eval",
                    legacy_mask_input=args.legacy_mask_input,
                )
            logger.log("offline/eval_step", step)
            logger.log("eval/step", step)

    if eval_dataset is not None and args.train_steps % args.eval_every != 0:
        for _ in range(args.eval_batches):
            log_offline_world_model_metrics(
                agent,
                eval_dataset,
                args.eval_batch_size,
                logger,
                prefix="eval",
                legacy_mask_input=args.legacy_mask_input,
            )
        logger.log("offline/eval_step", args.train_steps)
        logger.log("eval/step", args.train_steps)

    torch.save(agent.state_dict(), f"runs/{args.run_name}/ckpt/latest_agent.pth")
    logger.close()
