#!/usr/bin/env bash
set -e

if [ -f /root/.bashrc ]; then
    # shellcheck disable=SC1091
    source /root/.bashrc >/dev/null 2>&1 || true
fi

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

if [ -f /opt/intel/oneapi/ccl/latest/env/vars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/ccl/latest/env/vars.sh --force >/dev/null 2>&1 || true
fi

if [ -f "${VERL_PRIMARY_ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${VERL_PRIMARY_ENV_FILE}"
    set +a
elif [ -f "${VERL_FALLBACK_ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${VERL_FALLBACK_ENV_FILE}"
    set +a
fi

exec "$@"
