
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pathlib
import sys
import warnings

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder))

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
warnings.filterwarnings(
    'ignore',
    message='pkg_resources is deprecated as an API.*',
    category=UserWarning,
)

from interactive.oc_storm_adapter import build_oc_storm_session, resolve_policy_checkpoint_path
from wm_play.cli import (
  add_atari_env_args,
  add_pixel_policy_args,
  add_remote_server_args,
  add_wm_bootstrap_dataset_arg,
  add_wm_terminal_args,
  add_world_model_checkpoint_args,
  add_world_model_initial_source_arg,
  validate_remote_server_args,
)
try:
  from wm_play.cli import add_pixel_policy_args
except ImportError:
  def add_pixel_policy_args(parser):
    parser.add_argument('--policy-checkpoint', action='append', default=[],
                        help='Repeatable controller policy checkpoint path.')
    parser.add_argument('--policy-name', action='append', default=[],
                        help='Optional repeatable controller policy display name.')
    return parser
from wm_play.server_summary import CheckpointEntry, print_remote_server_summary
from wm_play.web_server import run_web_server



def _ram_panel_enabled(args, session) -> bool:
  return (
      bool(getattr(args, 'ram', False)) and
      not bool(getattr(session, 'wm_slots', None)) and
      callable(getattr(session, 'get_web_state', None)) and
      callable(getattr(session, '_read_rgb_frame', None)))


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description='oc-storm remote play server', allow_abbrev=False)
  parser.add_argument('--config', required=True, type=str,
                      help='Path to the experiment config.py file.')
  add_atari_env_args(parser, required=True)
  parser.add_argument('--seed', type=int, default=0,
                      help='Environment and extractor seed used by the config build function.')
  add_world_model_checkpoint_args(parser)
  parser.add_argument('--checkpoint', dest='wm_checkpoint', action='append',
                      help='Deprecated alias for --wm-checkpoint.')
  add_wm_bootstrap_dataset_arg(parser)
  parser.add_argument('--bootstrap-dataset', default='',
                      help='Deprecated alias for --wm-bootstrap-dataset.')
  parser.add_argument('--wm-sample-mode', choices=['probs', 'mode', 'random_sample'], default='probs',
                      help='Latent categorical prediction used while playing WMs. probs is the deterministic expectation.')
  add_world_model_initial_source_arg(parser)
  add_wm_terminal_args(parser, default=True)
  parser.add_argument('--wm-disable-kv-cache', action='store_true',
                      help='Disable transformer KV cache for WM play. Each step recomputes from the recent latent/action context.')
  parser.add_argument('--wm-kv-cache-dtype', choices=['fp32', 'amp'], default='fp32',
                      help='Precision used by the WM KV-cache path. fp32 is the stable interactive default; amp preserves the old BF16/AMP behavior.')
  add_pixel_policy_args(parser)
  add_remote_server_args(parser)
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  validate_remote_server_args(args)
  wm_checkpoints = args.wm_checkpoint or []
  if len(args.wm_name) not in (0, len(wm_checkpoints)):
    raise SystemExit('Provide either zero --wm-name values or exactly one name per checkpoint.')
  effective_policy_checkpoints = args.policy_checkpoint or []
  resolved_policy_checkpoints = [
      str(resolve_policy_checkpoint_path(x)) for x in effective_policy_checkpoints
  ]
  if len(args.policy_name) != 0:
    policy_count = len(effective_policy_checkpoints)
    if len(args.policy_name) != policy_count:
      raise SystemExit('Provide either zero --policy-name values or exactly one name per policy.')

  session = build_oc_storm_session(
      config_path=Path(args.config).expanduser().resolve(),
      env_name=args.env_name,
      seed=args.seed,
      checkpoint_args=[str(Path(x).expanduser().resolve()) for x in wm_checkpoints],
      wm_name_args=list(args.wm_name),
      policy_checkpoint_args=resolved_policy_checkpoints,
      policy_name_args=list(args.policy_name),
      controller=args.controller,
      bootstrap_dataset=args.bootstrap_dataset or None,
      wm_sample_mode=args.wm_sample_mode,
      wm_respect_terminal=args.wm_respect_terminal,
      wm_disable_kv_cache=args.wm_disable_kv_cache,
      wm_kv_cache_dtype=args.wm_kv_cache_dtype,
      wm_initial_source=args.wm_initial_source,
  )

  print_remote_server_summary(
      project='oc-storm/STORM',
      controller=session.controller,
      tcp_host=args.web_host,
      tcp_port=args.web_port,
      client_command=f'open http://<server-ip>:{args.web_port}',
      real_env=True,
      wm_checkpoints=[
          CheckpointEntry(name=slot.name, path=str(slot.checkpoint))
          for slot in session.wm_slots
      ],
      policy_checkpoints=[
          CheckpointEntry(name=slot.name, path=str(slot.checkpoint))
          for slot in session.policy_slots
      ],
      extras=[
          ('config', str(Path(args.config).expanduser())),
          ('env', args.env_name),
          ('seed', args.seed),
          ('device', getattr(session, 'device', None)),
          ('cuda visible', os.environ.get('CUDA_VISIBLE_DEVICES')),
          ('features', getattr(session, 'feature_extractor_source', None)),
          ('wm sample', getattr(session, 'wm_sample_mode', None)),
          ('wm terminal', 'respect' if getattr(session, 'wm_respect_terminal', False) else 'ignore'),
          ('wm cache', 'off' if getattr(session, 'wm_disable_kv_cache', False) else f'kv-{getattr(session, "wm_kv_cache_dtype", "fp32")}'),
          ('wm init', 'zero-context prior' if args.wm_initial_source == 'prior' else ('offline dataset context' if args.wm_initial_source == 'dataset' else f'real-env context ({int(getattr(getattr(session, "params", None), "imagine_context_length", 4))} frames)')),
          ('bootstrap', str(Path(args.bootstrap_dataset).expanduser()) if args.wm_initial_source == 'dataset' and args.bootstrap_dataset else ('ignored; zero-context prior is used' if args.bootstrap_dataset and args.wm_initial_source == 'prior' else ('ignored; real-env context is used' if args.bootstrap_dataset else None))),
      ],
      fps=args.fps,
      size=args.size,
      jpeg_quality=args.jpeg_quality,
      ram_panel=_ram_panel_enabled(args, session))

  try:
    run_web_server(args, session)
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  main()
