#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINK_NAMES=(datasets runs logs wandb debug_outputs results eval_results archive_runs)

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    realpath -m "$path"
  else
    realpath -m "$PWD/$path"
  fi
}

expected_root="${1:-}"
if [[ -n "$expected_root" ]]; then
  expected_root="$(resolve_path "$expected_root")"
fi

status=0
echo "Repository root: $ROOT_DIR"
if [[ -n "$expected_root" ]]; then
  echo "Expected data root: $expected_root"
fi

for name in "${LINK_NAMES[@]}"; do
  link_path="$ROOT_DIR/$name"
  expected_target=""
  if [[ -n "$expected_root" ]]; then
    expected_target="$expected_root/$name"
  fi

  if [[ ! -e "$link_path" && ! -L "$link_path" ]]; then
    echo "MISSING: $name"
    status=1
    continue
  fi
  if [[ ! -L "$link_path" ]]; then
    echo "NOT_SYMLINK: $name ($link_path)"
    status=1
    continue
  fi

  raw_target="$(readlink "$link_path")"
  target="$(resolve_path "$raw_target")"
  if [[ ! -d "$target" ]]; then
    echo "BROKEN: $name -> $raw_target (resolved: $target)"
    status=1
    continue
  fi
  if [[ -n "$expected_target" && "$target" != "$expected_target" ]]; then
    echo "WRONG_TARGET: $name -> $target (expected: $expected_target)"
    status=1
    continue
  fi
  if [[ ! -w "$target" ]]; then
    echo "NOT_WRITABLE: $name -> $target"
    status=1
    continue
  fi
  echo "OK: $name -> $target"
done

exit "$status"
