#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINK_NAMES=(datasets runs logs wandb debug_outputs results eval_results archive_runs)

yes_no() {
  local prompt="$1" answer
  while true; do
    read -r -p "$prompt [y/N] " answer
    case "$answer" in
      y|Y|yes|YES) return 0 ;;
      ""|n|N|no|NO) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    realpath -m "$path"
  else
    realpath -m "$PWD/$path"
  fi
}

target_root="${1:-}"
if [[ -z "$target_root" ]]; then
  echo "Enter the directory where OC-STORM runtime data should live."
  echo "The script will create these subdirectories under it:"
  printf "  %s\n" "${LINK_NAMES[@]}"
  read -r -p "Data root: " target_root
fi

if [[ -z "$target_root" ]]; then
  echo "ERROR: data root cannot be empty." >&2
  exit 1
fi

TARGET_ROOT="$(resolve_path "$target_root")"
if [[ -e "$TARGET_ROOT" && ! -d "$TARGET_ROOT" ]]; then
  echo "ERROR: target exists but is not a directory: $TARGET_ROOT" >&2
  exit 1
fi
if [[ ! -d "$TARGET_ROOT" ]]; then
  if yes_no "Create target directory $TARGET_ROOT?"; then
    mkdir -p "$TARGET_ROOT"
  else
    echo "ERROR: target directory does not exist: $TARGET_ROOT" >&2
    exit 1
  fi
fi
if [[ ! -w "$TARGET_ROOT" ]]; then
  echo "ERROR: target directory is not writable: $TARGET_ROOT" >&2
  exit 1
fi

echo "Repository root: $ROOT_DIR"
echo "Data root:       $TARGET_ROOT"

for name in "${LINK_NAMES[@]}"; do
  link_path="$ROOT_DIR/$name"
  target_path="$TARGET_ROOT/$name"
  mkdir -p "$target_path"

  if [[ -L "$link_path" ]]; then
    current_target="$(resolve_path "$(readlink "$link_path")")"
    if [[ "$current_target" == "$target_path" ]]; then
      echo "OK: $name -> $target_path"
      continue
    fi
    if yes_no "Replace existing symlink $name -> $current_target with $target_path?"; then
      rm "$link_path"
      ln -s "$target_path" "$link_path"
      echo "LINKED: $name -> $target_path"
    else
      echo "SKIP: $name"
    fi
    continue
  fi

  if [[ -e "$link_path" ]]; then
    if [[ ! -d "$link_path" ]]; then
      echo "ERROR: $link_path exists and is not a directory or symlink." >&2
      exit 1
    fi
    if rmdir "$link_path" 2>/dev/null; then
      ln -s "$target_path" "$link_path"
      echo "LINKED: $name -> $target_path"
      continue
    fi
    echo "FOUND: non-empty directory $link_path"
    if yes_no "Move its contents into $target_path and replace it with a symlink?"; then
      shopt -s dotglob nullglob
      items=("$link_path"/*)
      if ((${#items[@]} > 0)); then
        mv -n "${items[@]}" "$target_path"/
      fi
      shopt -u dotglob nullglob
      rmdir "$link_path"
      ln -s "$target_path" "$link_path"
      echo "LINKED: $name -> $target_path"
    else
      echo "SKIP: $name"
    fi
    continue
  fi

  ln -s "$target_path" "$link_path"
  echo "LINKED: $name -> $target_path"
done

echo
echo "Done. Run scripts/check_paths.sh to verify the links."
