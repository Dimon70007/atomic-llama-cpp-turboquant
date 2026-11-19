#!/usr/bin/env bash
# Shared llama-server resolution and busy-host guard for the TASK-073 test runners.
#
# Why this file exists: four runners each spelled out their own server path, and
# three of them hardcoded <repo>/build-dev, a Darwin dev-tree name. On any other
# build directory they aborted before measuring anything (TASK-073.1 F17). One
# was fixed in TASK-077 ST-6.4, the rest were found later - the duplication is
# what let that happen, so the resolution lives in one place now.
#
# Both functions below fail closed with the same message on every platform. A
# measurement host has one correct state: the right binary, nothing else holding
# the GPU. Guessing around a bad path or a busy device produces numbers that look
# fine and are not, which is the failure mode this task exists to remove.
#
# Source it, do not execute it:
#   # shellcheck source=tests/lib/llama_server_bin.sh
#   source "${REPO_ROOT}/tests/lib/llama_server_bin.sh"

# Print the first executable llama-server, in priority order.
#
# Args:
#   $1  repo root (the fork tree)
#   $2  explicit override from the caller's own env var, may be empty
#
# Priority: explicit > $CTEST_BINARY_DIRECTORY > $BUILD_DIR > build/ > build-dev/
# Returns 1 with an actionable message when nothing is executable.
llama_server_resolve() {
  local repo_root="${1:?repo_root required}"
  local explicit="${2:-}"
  local -a candidates=()

  if [[ -n "${explicit}" ]]; then
    candidates+=("${explicit}")
  fi
  if [[ -n "${CTEST_BINARY_DIRECTORY:-}" ]]; then
    candidates+=("${CTEST_BINARY_DIRECTORY}/bin/llama-server")
  fi
  if [[ -n "${BUILD_DIR:-}" ]]; then
    candidates+=("${BUILD_DIR}/bin/llama-server")
  fi
  candidates+=(
    "${repo_root}/build/bin/llama-server"
    "${repo_root}/build-dev/bin/llama-server"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "FAIL: llama-server not found." >&2
  echo >&2
  echo "Tried:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  echo >&2
  echo "Fix: set the path explicitly." >&2
  echo "  BUILD_DIR=/abs/path/to/build-dir <runner>     # preferred: build tree under test" >&2
  echo "  <RUNNER>_SERVER=/abs/path/to/llama-server <runner>   # exact binary, wins over everything" >&2
  echo >&2
  echo "Or build it there:" >&2
  echo "  cmake --build /abs/path/to/build-dir --target llama-server" >&2
  return 1
}

# PIDs whose executable IS llama-server, one per line.
#
# -x, not -f: -f matches any command line that merely mentions the string. The
# agent runtime on this host carries a path with "llama-server" in its argv, so
# -f reported a busy GPU that was not busy and aborted the sweep in 0.13 s
# (TASK-073.1.3; the same false positive TASK-077 F32 fixed in Python).
llama_server_busy_pids() {
  pgrep -x llama-server 2>/dev/null || true
}

# Stop when a server already holds the GPU, identically on every platform.
#
# One behaviour by operator decision (2026-11-19): stop with a clear error rather
# than warn. A leftover server competes for VRAM and perturbs the very numbers
# these gates measure, so "it ran anyway" is worse than "it refused". The caller
# turns a 1 return into its own exit code, since this is a sourced file.
llama_server_busy_guard() {
  local pids pid cmd
  pids="$(llama_server_busy_pids)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  echo "FAIL: a llama-server is already running. Stop it before measuring:" >&2
  echo >&2
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    cmd="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || ps -o args= -p "${pid}" 2>/dev/null || true)"
    if [[ -n "${cmd}" ]]; then
      echo "  pid ${pid}: ${cmd}" >&2
    else
      echo "  pid ${pid}" >&2
    fi
  done <<<"${pids}"
  echo >&2
  echo "Fix: stop it, then re-run." >&2
  echo "  make minillm-stop                       # product service" >&2
  echo "  kill <pid>                              # a leftover run of your own" >&2
  return 1
}
