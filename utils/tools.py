import torch
import os
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from einops import repeat
from contextlib import contextmanager
import time
import matplotlib.pyplot as plt
import atexit

from collections import defaultdict

try:
    import wandb
except ImportError:
    wandb = None


def seed_np_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, run_name, config=None) -> None:
        self.log_dir = self._build_log_dir(run_name)
        self.wandb_dir = self._build_wandb_dir()
        self.wandb_run = None
        self.use_wandb = os.environ.get("WANDB_ENABLED", "0").lower() in {"1", "true", "yes"}
        self._wandb_defined_metrics = set()

        if self.use_wandb:
            if wandb is None:
                raise ImportError("WANDB_ENABLED is set, but wandb is not installed. Please `pip install wandb`.")

            init_config = {
                "run_name": run_name,
                "log_dir": self.log_dir,
            }
            if config is not None:
                init_config.update(config)

            self.wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "oc-storm"),
                entity=os.environ.get("WANDB_ENTITY", "ssl-lab"),
                name=run_name,
                config=init_config,
                dir=self.wandb_dir,
                mode=os.environ.get("WANDB_MODE", "online"),
                sync_tensorboard=False,
                reinit="finish_previous",
            )
            atexit.register(self.close)

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=1)  # tensorboard writer
        self.tag_step = defaultdict(int)

    @staticmethod
    def _build_log_dir(run_name: str) -> str:
        base_dir = "runs"
        os.makedirs(base_dir, exist_ok=True)

        timestamp = time.strftime("%y%m%d-%H%M%S", time.localtime())
        index = 0
        while True:
            candidate = os.path.join(base_dir, f"{run_name}-{timestamp}-{index}")
            if not os.path.exists(candidate):
                return candidate
            index += 1

    @staticmethod
    def _build_wandb_dir() -> str:
        base_dir = "wandb"
        os.makedirs(base_dir, exist_ok=True)
        return os.path.abspath(base_dir)

    def log(self, tag: str, value, step: int | None = None):
        if step is None:
            self.tag_step[tag] += 1
            log_step = self.tag_step[tag]
        else:
            log_step = int(step)
            self.tag_step[tag] = max(self.tag_step[tag], log_step)
        if value is None:  # None refers to skip logging illeagl values, but still increase the step count
            return
        if "video" in tag:
            self.writer.add_video(tag, value, log_step, fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, log_step)
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, log_step)
        else:
            self.writer.add_scalar(tag, value, log_step)
            self._log_scalar_to_wandb(tag, value, log_step)

    def _log_scalar_to_wandb(self, tag: str, value, step: int):
        if self.wandb_run is None:
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return
            value = value.detach().item()
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return

        step_tag = f"{tag}/step"
        if tag not in self._wandb_defined_metrics:
            wandb.define_metric(step_tag, hidden=True)
            wandb.define_metric(tag, step_metric=step_tag)
            self._wandb_defined_metrics.add(tag)
        self.wandb_run.log({step_tag: int(step), tag: value})

    def close(self):
        self.writer.flush()
        self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
