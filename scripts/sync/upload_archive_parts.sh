#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  printf 'Usage: %s /path/to/archive.tar.zst\n' "$0" >&2
  exit 2
fi

ARCHIVE="$1"
if [ ! -f "${ARCHIVE}" ]; then
  printf 'Archive not found: %s\n' "${ARCHIVE}" >&2
  exit 2
fi

if [ -f "${HOME}/.config/oc-storm-upload.env" ]; then
  # shellcheck disable=SC1091
  . "${HOME}/.config/oc-storm-upload.env"
fi

UPLOAD_URL="${TSINGHUA_UPLOAD_URL:-}"
REMOTE_DIR="${TSINGHUA_REMOTE_DIR:-/luyukuan}"
PART_SIZE="${PART_SIZE:-512M}"
PART_DIR="${PART_DIR:-${ARCHIVE}.parts}"
STAMP_DIR="${PART_DIR}/.uploaded"

if [ -z "${UPLOAD_URL}" ]; then
  printf 'TSINGHUA_UPLOAD_URL is not set.\n' >&2
  exit 2
fi

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

UPLOAD_ENDPOINT="$(resolve_upload_url "${UPLOAD_URL}")"
if [ -z "${UPLOAD_ENDPOINT}" ]; then
  printf 'Could not resolve Tsinghua upload endpoint.\n' >&2
  exit 1
fi

mkdir -p "${PART_DIR}" "${STAMP_DIR}"

base_name="$(basename "${ARCHIVE}")"
prefix="${PART_DIR}/${base_name}.part-"

if ! compgen -G "${prefix}*" >/dev/null; then
  printf '[%s] Splitting %s into %s parts under %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${ARCHIVE}" "${PART_SIZE}" "${PART_DIR}"
  split -b "${PART_SIZE}" -d -a 3 "${ARCHIVE}" "${prefix}"
fi

(
  cd "${PART_DIR}"
  sha256sum "${base_name}".part-* > SHA256SUMS
  sha256sum "$(realpath --relative-to="${PART_DIR}" "${ARCHIVE}")" > ARCHIVE_SHA256SUM || sha256sum "${ARCHIVE}" > ARCHIVE_SHA256SUM
  {
    printf 'Original archive: %s\n' "${base_name}"
    printf 'Rebuild command: cat %s.part-* > %s\n' "${base_name}" "${base_name}"
    printf 'Verify command: sha256sum -c SHA256SUMS && sha256sum -c ARCHIVE_SHA256SUM\n'
  } > README_REBUILD.txt
)

upload_one() {
  local file="$1" stamp
  stamp="${STAMP_DIR}/$(basename "${file}").ok"
  if [ -f "${stamp}" ]; then
    printf '[%s] Skip already uploaded: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$(basename "${file}")"
    return 0
  fi

  printf '[%s] Uploading %s to %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$(basename "${file}")" "${REMOTE_DIR}"
  curl --fail --show-error --location \
    --retry 8 --retry-delay 20 --retry-all-errors \
    -F "parent_dir=${REMOTE_DIR}" \
    -F "file=@${file}" \
    "${UPLOAD_ENDPOINT}?ret-json=1"
  touch "${stamp}"
}

for file in "${prefix}"* "${PART_DIR}/SHA256SUMS" "${PART_DIR}/ARCHIVE_SHA256SUM" "${PART_DIR}/README_REBUILD.txt"; do
  upload_one "${file}"
done

printf '[%s] All parts uploaded.\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
