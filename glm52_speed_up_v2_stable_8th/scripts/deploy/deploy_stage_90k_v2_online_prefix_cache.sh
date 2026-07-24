#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export V2_ENABLE_PREFIX_CACHING=1
export V2_ONLINE_FORMAL_DEPLOY=1

exec "${SCRIPT_DIR}/deploy_stage_90k_v2.sh"
