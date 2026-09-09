#!/usr/bin/env bash
set -euo pipefail

# Harness-only successor to v35. Runtime code and image are unchanged; the
# production-shape smoke now performs the minimum two updates needed to cover
# the first live refit and the following update.
export RN4PT_VERSION_OVERRIDE=verl_real_nvfp4_prod_dsfix_20260909_v36
export RN4PT_RUNTIME_IMAGE_OVERRIDE=/lustre/fsw/general_sa/shuazhang/images/verl.vllm026.mcore018.te218e7.realnvfp4.fi0615p1.prod.20260909.v35.sqsh
export RN4PT_IMAGE_ENV_OVERRIDE=/opt/verl-rn4pt-20260909-v35
export RN4PT_EXP_TAG_OVERRIDE=20260909_v36
export RN4PT_JOB_LABEL_OVERRIDE=v36
export RN4PT_SMOKE8_TARGET_STEP_OVERRIDE=2

# Reuse v35's already-passed probe/build/preflight evidence. The shared
# manifest independently verifies the image checksum, lock hash, and that all
# commits since the image build changed only job harness files.
readonly SOURCE_STATE=/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/run_state/verl_real_nvfp4_prod_dsfix_20260909_v35
readonly TARGET_STATE=/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/run_state/verl_real_nvfp4_prod_dsfix_20260909_v36
mkdir -p "$TARGET_STATE"
for artifact in probe.pass build.pass preflight.pass; do
  [[ -s "$SOURCE_STATE/$artifact" ]] || {
    echo "REAL_NVFP4_PERTOKEN_REFUSED: v35 prerequisite is missing: $SOURCE_STATE/$artifact" >&2
    exit 2
  }
  cp --reflink=auto "$SOURCE_STATE/$artifact" "$TARGET_STATE/$artifact"
done

exec "$(dirname "$0")/../prod_overlong_20260908_v31/submit.sh" "$@"
