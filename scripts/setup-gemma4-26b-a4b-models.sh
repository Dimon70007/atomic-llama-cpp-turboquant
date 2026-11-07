#!/usr/bin/env bash
# Populate .scratch/gemma-4-26b-a4b/ with local GGUF symlinks (not committed).
# Same layout as scripts/run-gemma4-mtp-server.sh / run-gemma4-server-turbo.sh.
#
# Usage:
#   ./scripts/setup-gemma4-26b-a4b-models.sh
#   MODELS_DIR=$HOME/models ./scripts/setup-gemma4-26b-a4b-models.sh
#
# Env:
#   MODELS_DIR  - search dir for GGUFs (default: $HOME/models)
#   FORCE=1      - replace existing links/files in .scratch

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${ROOT}/.scratch/gemma-4-26b-a4b"
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
FORCE="${FORCE:-0}"

mkdir -p "${DIR}"

# name -> preferred source basenames (first hit wins)
link_one() {
  local dest_name="$1"
  shift
  local dest="${DIR}/${dest_name}"
  local src="" cand
  for cand in "$@"; do
    if [[ -f "${MODELS_DIR}/${cand}" ]]; then
      src="${MODELS_DIR}/${cand}"
      break
    fi
    if [[ -f "${cand}" ]]; then
      src="${cand}"
      break
    fi
  done
  if [[ -z "${src}" ]]; then
    echo "skip: ${dest_name} (not found under ${MODELS_DIR})" >&2
    return 0
  fi
  if [[ -e "${dest}" || -L "${dest}" ]]; then
    if [[ "${FORCE}" != "1" ]]; then
      echo "keep: ${dest} -> $(readlink "${dest}" 2>/dev/null || echo existing)"
      return 0
    fi
    rm -f "${dest}"
  fi
  ln -sfn "${src}" "${dest}"
  echo "link: ${dest_name} -> ${src}"
}

# MTP / turbo default (run-gemma4-mtp-server.sh)
link_one "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf" \
  "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

# TASK-066 nomtp ladder / Metal batch-invariance fixture
link_one "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf" \
  "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"

# Optional qat sibling if present
link_one "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf" \
  "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"

# TASK-066 P3 MTP draft head (also used by run-gemma4-mtp-server via DRAFT_GGUF)
link_one "mtp-gemma-4-26B-A4B-it-Q8_0.gguf" \
  "mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
link_one "mtp-gemma-4-26B-A4B-it-F16.gguf" \
  "mtp-gemma-4-26B-A4B-it-F16.gguf"
# Alias expected by run-gemma4-mtp-server.sh default DRAFT path (repo .scratch root)
link_one_root() {
  local dest_name="$1"
  shift
  local dest="${ROOT}/.scratch/${dest_name}"
  local src="" cand
  for cand in "$@"; do
    if [[ -f "${MODELS_DIR}/${cand}" ]]; then src="${MODELS_DIR}/${cand}"; break; fi
    if [[ -f "${DIR}/${cand}" ]]; then src="${DIR}/${cand}"; break; fi
    if [[ -f "${cand}" ]]; then src="${cand}"; break; fi
  done
  if [[ -z "${src}" ]]; then
    echo "skip: .scratch/${dest_name}" >&2
    return 0
  fi
  mkdir -p "${ROOT}/.scratch"
  if [[ -e "${dest}" || -L "${dest}" ]]; then
    if [[ "${FORCE}" != "1" ]]; then
      echo "keep: ${dest}"
      return 0
    fi
    rm -f "${dest}"
  fi
  ln -sfn "${src}" "${dest}"
  echo "link: .scratch/${dest_name} -> ${src}"
}
link_one_root "gemma-assistant-mtp.gguf" \
  "mtp-gemma-4-26B-A4B-it-Q8_0.gguf" \
  "gemma-assistant-mtp.gguf" \
  "mtp-gemma-4-26B-A4B-it-F16.gguf"

echo "ok: ${DIR}"
ls -la "${DIR}"
ls -la "${ROOT}/.scratch/gemma-assistant-mtp.gguf" 2>/dev/null || true
