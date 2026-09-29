#!/bin/bash
log_info() { printf '\033[96mInfo: %s\033[0m\n' "$*"; }
die() {
  printf '\033[31mError: %s\033[0m\n' "$*" >&2
  exit 1
}
function require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}
DOCTEST_PYTHON="${DOCTEST_PYTHON:-python3}"
function run_shell_block() {
  local marker="$1"
  local block
  block="$("${DOCTEST_PYTHON}" "${DOCTEST_HELPER_PATH}" extract "${marker}")" || return $?
  set +u
  source /dev/stdin <<<"${block}"
  set -u
}
function check_shell_block() {
  local block
  block="$("${DOCTEST_PYTHON}" "${DOCTEST_HELPER_PATH}" extract "$1")" || return $?
  bash -n <<<"${block}"
}
