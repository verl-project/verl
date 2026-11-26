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
VERIFY_RUNTIME_PATH="${DOCTEST_DIR}/scripts/verify_runtime.py"
function doctest_helper() {
  "${DOCTEST_PYTHON}" "${DOCTEST_DIR}/scripts/doctest_helper.py" "$@"
}
function run_shell_block() {
  local block
  block="$(doctest_helper extract --runtime "$1")" || return $?
  set +u
  source /dev/stdin <<<"${block}"
  set -u
}
function check_shell_block() {
  local block
  block="$(doctest_helper extract "$1")" || return $?
  bash -n <<<"${block}"
}
