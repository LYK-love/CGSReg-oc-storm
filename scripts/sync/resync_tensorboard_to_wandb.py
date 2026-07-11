#!/usr/bin/env python3
import argparse
import hashlib
import re
from pathlib import Path

import wandb
from tensorboard.backend.event_processing import event_accumulator


TIMESTAMP_SUFFIX = re.compile(r"-(\d{6})-(\d{6})-(\d+)$")


def normalized_run_name(event_file: Path, root: Path) -> str:
    event_dir = event_file.parent
    rel = event_dir.relative_to(root)
    parts = list(rel.parts)
    parts[-1] = TIMESTAMP_SUFFIX.sub("", parts[-1])
    return "/".join(parts)


def run_id_for(name: str, suffix: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"{suffix}-{digest}"


def infer_config(run_name: str) -> dict:
    config = {"source_run_name": run_name}
    size_match = re.search(r"pong_wm_reg_(base|size\d+m)_", run_name)
    if size_match:
        config["size_config"] = size_match.group(1)
    mask_match = re.search(r"_(mask[^_]+)_spatial_", run_name)
    if mask_match:
        config["mask_preset"] = mask_match.group(1)
    spatial_match = re.search(r"_spatial_([^_]+)_temporal_", run_name)
    if spatial_match:
        config["spatial_weight_label"] = spatial_match.group(1)
    temporal_match = re.search(r"_temporal_([^_/]+)$", run_name)
    if temporal_match:
        config["temporal_weight_label"] = temporal_match.group(1)
    return config


def load_scalars(event_file: Path) -> dict[str, list]:
    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    return {tag: accumulator.Scalars(tag) for tag in accumulator.Tags().get("scalars", [])}


def sync_event_file(args, event_file: Path) -> tuple[str, int, int]:
    run_name = normalized_run_name(event_file, args.root)
    scalars = load_scalars(event_file)
    point_count = sum(len(points) for points in scalars.values())

    if args.dry_run:
        return run_name, len(scalars), point_count

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.name_prefix + run_name,
        id=run_id_for(run_name, args.id_prefix),
        group=args.group,
        job_type="tensorboard-resync",
        tags=args.tags,
        config={
            **infer_config(run_name),
            "source_event_file": str(event_file),
            "metric_namespace_fix": "removed TensorBoard logdir prefix from W&B metric names",
        },
        dir=args.wandb_dir,
        resume=args.resume,
        reinit="finish_previous",
    )

    for tag in sorted(scalars):
        step_tag = f"{tag}/step"
        wandb.define_metric(step_tag, hidden=True)
        wandb.define_metric(tag, step_metric=step_tag)
        for point in scalars[tag]:
            run.log({step_tag: point.step, tag: point.value})

    run.finish()
    return run_name, len(scalars), point_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-upload TensorBoard scalar events to W&B with normalized metric names."
    )
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--entity", default="ssl-lab")
    parser.add_argument("--project", default="oc-storm-fixed")
    parser.add_argument("--group", default="pong_wm_reg_image_sweep_metric_fix")
    parser.add_argument("--wandb-dir", default="wandb")
    parser.add_argument("--name-prefix", default="")
    parser.add_argument("--id-prefix", default="tbfix")
    parser.add_argument("--resume", choices=["allow", "never", "must"], default="never")
    parser.add_argument("--tags", nargs="*", default=["metric-name-fix", "tensorboard-resync"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.root = args.root.resolve()
    event_files = sorted(args.root.rglob("events.out.tfevents*"))
    if args.limit:
        event_files = event_files[: args.limit]

    if not event_files:
        raise SystemExit(f"No TensorBoard event files found under {args.root}")

    for event_file in event_files:
        run_name, tag_count, point_count = sync_event_file(args, event_file)
        print(f"{run_name}\ttags={tag_count}\tpoints={point_count}")


if __name__ == "__main__":
    main()
