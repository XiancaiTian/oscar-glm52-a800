#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy_common_env.sh"

"${TASK_ROOT}/scripts/deploy/lib/manage_stack_2p1d.sh" down "${SCHEME}" || true
"${TASK_ROOT}/scripts/deploy/lib/manage_stack_1p1d.sh" down "${SCHEME}" || true
record_acceptance "stop_all" "PASS" "task service processes stopped"
