#!/usr/bin/env bash
set -euo pipefail

BUNDLE=$(cd "$(dirname "$0")" && pwd)
readonly BUNDLE
export RN4PT_MANIFEST_OVERRIDE=$BUNDLE/manifest.sh
exec bash "$BUNDLE/../r3_nativeonline_post026_20260901_v10/submit.sh" "$@"
