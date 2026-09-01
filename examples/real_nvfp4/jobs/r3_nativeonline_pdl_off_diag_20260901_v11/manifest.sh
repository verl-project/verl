#!/usr/bin/env bash
# shellcheck disable=SC2034

readonly RN4PT_VERSION_OVERRIDE=verl_real_nvfp4_r3_nativeonline_pdl_off_diag_20260901_v11
readonly RN4PT_BUNDLE_OVERRIDE=/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/verl_nvfp4_e2e_r3_20260831_v3/examples/real_nvfp4/jobs/r3_nativeonline_pdl_off_diag_20260901_v11
readonly RN4PT_BASE_IMAGE_OVERRIDE=/lustre/fsw/general_sa/shuazhang/images/verl.vllm026.mcore018.te218e7.realnvfp4.nativeonline.post026.20260901.v10.sqsh
readonly RN4PT_RUNTIME_IMAGE_OVERRIDE=/lustre/fsw/general_sa/shuazhang/images/verl.vllm026.mcore018.te218e7.realnvfp4.nativeonline.post026.pdl-off-diag.20260901.v11.sqsh
readonly RN4PT_SMOKE_EXP_OVERRIDE=verl_30b_realw4a4_r3_nativeonline_pdl_off_diag_smoke_20260901_v11
readonly RN4PT_SHORT_EXP_OVERRIDE=verl_30b_realw4a4_r3_nativeonline_pdl_off_diag_8n_20260901_v11
readonly RN4PT_VLLM_DISABLE_TRTLLM_MOE_PDL=1

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/../r3_nativeonline_post026_20260901_v10/manifest.sh"
