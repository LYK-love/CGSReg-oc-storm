#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"

WATCH_DIRS=(${WATCH_DIRS:-runs results debug_outputs})
read -r -a EXCLUDE_GLOBS_ARRAY <<< "${EXCLUDE_GLOBS:-agent_step_*.pth *.mp4 *.avi *.mov *.mkv *.webm *.gif *.png *.jpg *.jpeg *.npy *.npz}"
STATE_DIR="${STATE_DIR:-.sync_state}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/SSD_RAID0/lyk/oc-storm/archives}"
QUIET_MINUTES="${QUIET_MINUTES:-30}"
LOCK_FILE="${STATE_DIR}/upload_experiments.lock"
STAMP_FILE="${STATE_DIR}/last_uploaded.stamp"
LOG_FILE="${STATE_DIR}/upload_experiments.log"

TSINGHUA_REMOTE_DIR="${TSINGHUA_REMOTE_DIR:-/luyukuan}"
UPLOAD_URL="${TSINGHUA_UPLOAD_URL:-}"

mkdir -p "${STATE_DIR}" "${ARCHIVE_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${LOG_FILE}" >&2
}

resolve_upload_url() {
  local url="$1" token endpoint
  if [[ "${url}" == */u/d/* ]]; then
    token="$(printf '%s\n' "${url}" | sed -E 's#.*/u/d/([A-Za-z0-9]+)/?.*#\1#')"
    endpoint="$(
      curl -fsS "https://cloud.tsinghua.edu.cn/api/v2.1/upload-links/${token}/upload/" \
        | python3 -c 'import sys, json; print(json.load(sys.stdin).get("upload_link", ""))' 2>/dev/null \
        || true
    )"
    if [ -z "${endpoint}" ]; then
      endpoint="$(curl -fsS "https://cloud.tsinghua.edu.cn/api2/upload-link/?t=${token}" | tr -d '"' || true)"
    fi
    printf '%s\n' "${endpoint}"
  else
    printf '%s\n' "${url}"
  fi
}

has_files() {
  local dir
  for dir in "${WATCH_DIRS[@]}"; do
    if [ -d "${dir}" ] && find -L "${dir}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
      return 0
    fi
  done
  return 1
}

has_active_jobs() {
  pgrep -af "tiny-exp-scheduler|train_offline.py|train_async.py|train.py|eval.py" \
    | grep -F "${PROJECT_ROOT}" \
    | grep -v "upload_experiments_when_done.sh" >/dev/null 2>&1
}

newest_artifact_epoch() {
  local dir newest=0 current
  for dir in "${WATCH_DIRS[@]}"; do
    if [ -e "${dir}" ]; then
      current="$(find -L "${dir}" -mindepth 1 -printf '%T@\n' 2>/dev/null | sort -nr | head -1 || true)"
      current="${current%%.*}"
      if [ -n "${current}" ] && [ "${current}" -gt "${newest}" ]; then
        newest="${current}"
      fi
    fi
  done
  printf '%s\n' "${newest}"
}

path_is_excluded() {
  local path="$1" pattern
  for pattern in "${EXCLUDE_GLOBS_ARRAY[@]}"; do
    case "${path}" in
      ${pattern}|*/${pattern})
        return 0
        ;;
    esac
  done
  return 1
}

artifact_stamp() {
  local dir rel_path size mtime
  for dir in "${WATCH_DIRS[@]}"; do
    if [ -e "${dir}" ]; then
      while IFS=$'\t' read -r rel_path size mtime; do
        if ! path_is_excluded "${dir}/${rel_path}"; then
          printf '%s\t%s\t%s\n' "${dir}/${rel_path}" "${size}" "${mtime}"
        fi
      done < <(find -L "${dir}" -mindepth 1 -printf '%P\t%s\t%T@\n' 2>/dev/null)
    fi
  done | sort | sha256sum | awk '{print $1}'
}

archive_artifacts() {
  local timestamp archive
  local tar_excludes=() pattern
  timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
  archive="${ARCHIVE_DIR}/oc_storm_experiments_${timestamp}.tar.zst"

  for pattern in "${EXCLUDE_GLOBS_ARRAY[@]}"; do
    tar_excludes+=(--exclude="${pattern}")
  done

  log "Creating archive: ${archive}"
  tar --zstd -cf "${archive}" "${tar_excludes[@]}" --dereference --ignore-failed-read "${WATCH_DIRS[@]}"
  log "Archive size: $(stat -c '%s' "${archive}") bytes"
  printf '%s\n' "${archive}"
}

upload_archive() {
  local archive="$1" upload_endpoint
  if [ -z "${UPLOAD_URL}" ]; then
    log "TSINGHUA_UPLOAD_URL is not set; archive kept locally: ${archive}"
    return 2
  fi
  upload_endpoint="$(resolve_upload_url "${UPLOAD_URL}")"
  if [ -z "${upload_endpoint}" ]; then
    log "Could not resolve Tsinghua upload endpoint."
    return 1
  fi

  log "Uploading $(basename "${archive}") to ${TSINGHUA_REMOTE_DIR}"
  if ! curl --fail --show-error --location \
    -F "parent_dir=${TSINGHUA_REMOTE_DIR}" \
    -F "file=@${archive}" \
    "${upload_endpoint}?ret-json=1"; then
    log "Upload failed: $(basename "${archive}")"
    return 1
  fi
  log "Upload finished: $(basename "${archive}")"
}

main() {
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    log "Another upload check is already running."
    exit 0
  fi

  if ! has_files; then
    log "No experiment artifacts found; nothing to upload."
    exit 0
  fi

  if has_active_jobs; then
    log "Experiment process still running; skip."
    exit 0
  fi

  newest="$(newest_artifact_epoch)"
  now="$(date +%s)"
  quiet_seconds=$((QUIET_MINUTES * 60))
  if [ "${newest}" -gt 0 ] && [ $((now - newest)) -lt "${quiet_seconds}" ]; then
    log "Artifacts changed recently; waiting for ${QUIET_MINUTES} quiet minutes."
    exit 0
  fi

  stamp="$(artifact_stamp)"
  if [ -f "${STAMP_FILE}" ] && [ "$(cat "${STAMP_FILE}")" = "${stamp}" ]; then
    log "Current artifact snapshot was already uploaded."
    exit 0
  fi

  archive="$(archive_artifacts)"
  if upload_archive "${archive}"; then
    printf '%s\n' "${stamp}" > "${STAMP_FILE}"
  else
    log "Upload was not completed; stamp not advanced."
    exit 1
  fi
}

main "$@"
