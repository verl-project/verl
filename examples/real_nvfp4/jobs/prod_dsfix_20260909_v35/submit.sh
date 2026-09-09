#!/usr/bin/env bash
set -euo pipefail

# Exact-source successor to v34 after isolating the online-quantization unit
# test's intentionally exported worker environment.
export RN4PT_VERSION_OVERRIDE=verl_real_nvfp4_prod_dsfix_20260909_v35
export RN4PT_RUNTIME_IMAGE_OVERRIDE=/lustre/fsw/general_sa/shuazhang/images/verl.vllm026.mcore018.te218e7.realnvfp4.fi0615p1.prod.20260909.v35.sqsh
export RN4PT_IMAGE_ENV_OVERRIDE=/opt/verl-rn4pt-20260909-v35
export RN4PT_EXP_TAG_OVERRIDE=20260909_v35
export RN4PT_JOB_LABEL_OVERRIDE=v35

exec "$(dirname "$0")/../prod_overlong_20260908_v31/submit.sh" "$@"
