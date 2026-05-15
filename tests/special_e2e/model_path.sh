#!/usr/bin/env bash

DEFAULT_VERL_TEST_MODEL_ROOT=${HOME}/models
if [ -z "${VERL_TEST_MODEL_ROOT+x}" ]; then
    VERL_TEST_MODEL_ROOT=${DEFAULT_VERL_TEST_MODEL_ROOT}
fi

if [ "${VERL_TEST_MODEL_ROOT%/}" = "${DEFAULT_VERL_TEST_MODEL_ROOT%/}" ] \
    && [ ! -e "${VERL_TEST_MODEL_ROOT}" ] \
    && [ -d /root/.cache/models ]; then
    ln -sfn /root/.cache/models "${VERL_TEST_MODEL_ROOT}"
fi

verl_model_path() {
    local model_id="$1"
    printf '%s/%s\n' "${VERL_TEST_MODEL_ROOT%/}" "${model_id}"
}
