#!/usr/bin/env bash
set -euo pipefail

readonly BUNDLE
BUNDLE=$(cd "$(dirname "$0")" && pwd)
export RN4PT_MANIFEST_OVERRIDE=$BUNDLE/manifest.sh
exec "$BUNDLE/../r3_nativeonline_post026_20260901_v10/submit.sh" "$@"
