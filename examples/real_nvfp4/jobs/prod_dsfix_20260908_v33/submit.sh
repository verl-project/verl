#!/usr/bin/env bash
set -euo pipefail

# Fresh namespaces and runtime image for the dynamic-sampling, strict-verifier,
# and trainer/rollout carve-out fixes. The audited v31 scheduler harness remains
# the implementation; its manifest exposes these version overrides explicitly.
export RN4PT_VERSION_OVERRIDE=verl_real_nvfp4_prod_dsfix_20260908_v33
export RN4PT_RUNTIME_IMAGE_OVERRIDE=/lustre/fsw/general_sa/shuazhang/images/verl.vllm026.mcore018.te218e7.realnvfp4.fi0615p1.prod.20260908.v33.sqsh
export RN4PT_IMAGE_ENV_OVERRIDE=/opt/verl-rn4pt-20260908-v33
export RN4PT_EXP_TAG_OVERRIDE=20260908_v33
export RN4PT_JOB_LABEL_OVERRIDE=v33

exec "$(dirname "$0")/../prod_overlong_20260908_v31/submit.sh" "$@"
