#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# Export only for the documented Bash installers; other Git commands pass through.
function git() (
  local original=("$@") args=("$@") options=() repository destination staging=""
  local attempt status=0
  local attempts="${DOCTEST_GIT_ATTEMPTS:-3}" limit="${DOCTEST_GIT_TIMEOUT:-1200}"
  local delay="${DOCTEST_GIT_RETRY_DELAY:-30}"
  case "${1:-}" in
    clone)
      shift
      # The documented installers use these clone options. Leave other forms to Git.
      while [[ $# -gt 0 ]]; do
        case "$1" in
          -b|--branch|--depth)
            [[ $# -ge 2 ]] || { command git "${original[@]}"; return $?; }
            options+=("$1" "$2"); shift 2 ;;
          --recursive|--recurse-submodules|--single-branch|--depth=*|--branch=*)
            options+=("$1"); shift ;;
          --) shift; break ;;
          -*) command git "${original[@]}"; return $? ;;
          *) break ;;
        esac
      done
      [[ $# -ge 1 && $# -le 2 ]] || { command git "${original[@]}"; return $?; }
      repository="$1"
      destination="${repository%/}"
      destination="${destination##*/}"
      destination="${2:-${destination%.git}}"
      if [[ -z "${destination}" || -e "${destination}" || -L "${destination}" ]]; then
        printf 'Error: refusing to overwrite clone destination: %s\n' "${destination}" >&2
        return 128
      fi
      staging="$(mktemp -d "$(dirname -- "${destination}")/.doctest-clone.XXXXXX")" || return $?
      # Only this invocation's temporary directory is ever removed.
      trap 'rm -rf -- "${staging}"' EXIT
      args=(clone "${options[@]}" "${repository}" "${staging}/checkout")
      if [[ "${repository}" == https://github.com/verl-project/verl.git && -n "${VERL_REPOSITORY:-}" ]]; then
        printf 'Info: Redirecting the documented verl clone to the test repository.\n'
        args=(-c "url.${VERL_REPOSITORY}.insteadOf=${repository}" "${args[@]}")
      fi
      ;;
    submodule)
      [[ "${2:-}" == update ]] || { command git "$@"; return $?; }
      ;;
    *) command git "$@"; return $? ;;
  esac
  [[ "${attempts}" =~ ^[1-9][0-9]*$ && "${limit}" =~ ^[1-9][0-9]*$ && "${delay}" =~ ^(0|[1-9][0-9]*)$ ]] || {
    printf 'Error: Git attempts/timeout must be positive seconds/count; retry delay must be nonnegative.\n' >&2
    return 2
  }
  export GIT_HTTP_LOW_SPEED_LIMIT="${GIT_HTTP_LOW_SPEED_LIMIT:-10240}"
  export GIT_HTTP_LOW_SPEED_TIME="${GIT_HTTP_LOW_SPEED_TIME:-120}"
  export GIT_TERMINAL_PROMPT=0
  trap 'exit 130' INT
  trap 'exit 143' TERM
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    printf 'Info: git %s attempt %s/%s (timeout %ss, HTTP low speed %s B/s for %ss).\n' \
      "${original[0]}" "${attempt}" "${attempts}" "${limit}" \
      "${GIT_HTTP_LOW_SPEED_LIMIT}" "${GIT_HTTP_LOW_SPEED_TIME}" >&2
    if timeout --kill-after=30s "${limit}s" git "${args[@]}"; then
      if [[ -n "${staging}" ]]; then
        mv -T --no-clobber -- "${staging}/checkout" "${destination}" || return $?
        [[ ! -e "${staging}/checkout" ]] || return 128
        if [[ "${repository}" == https://github.com/verl-project/verl.git && -n "${VERL_REVISION:-}" ]]; then
          command git -C "${destination}" checkout --detach "${VERL_REVISION}" || return $?
          (cd "${destination}" && git submodule update --init --recursive) || return $?
          command git -C "${destination}" rev-parse HEAD || return $?
        fi
      fi
      return 0
    else
      status=$?
    fi
    printf 'Warning: git %s attempt %s failed (exit %s).\n' "${original[0]}" "${attempt}" "${status}" >&2
    # Cancellation is not a transient download failure.
    [[ "${status}" != 130 && "${status}" != 143 ]] || return "${status}"
    if [[ -n "${staging}" ]]; then
      rm -rf -- "${staging}/checkout" || return $?
    fi
    if ((attempt < attempts)); then
      sleep "$((delay * attempt))" || return $?
    fi
  done
  return "${status}"
)
export -f git
