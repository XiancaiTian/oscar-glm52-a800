#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy_common_env.sh"
write_environment_md

mkdir -p "${TASK_ROOT}/logs/deploy" "${TASK_ROOT}/state"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG="${TASK_ROOT}/logs/deploy/${TS}_deploy_2p1d_from_image.log"
DEPLOY_ACTION="${DEPLOY_ACTION:-restart}"

"${TASK_ROOT}/scripts/deploy/lib/manage_stack_2p1d.sh" "${DEPLOY_ACTION}" "${SCHEME}" 2>&1 | tee "${LOG}"
BASE_URL="http://${DECODE_HEAD}:${PROXY_PORT}/v1"
DEPLOY_MANIFEST="$(latest_deploy_manifest)"
cat > "${TASK_ROOT}/state/2p1d_endpoint.env" <<EOF
BASE_URL=${BASE_URL}
TOPOLOGY=2P1D
DEPLOY_LOG=${LOG}
DEPLOY_MANIFEST=${DEPLOY_MANIFEST}
SCHEME=${SCHEME}
DEPLOY_ACTION=${DEPLOY_ACTION}
IMAGE_TAG=${IMAGE_TAG}
EOF
record_acceptance "2p1d_deploy" "PASS" "endpoint=${BASE_URL} manifest=${DEPLOY_MANIFEST}"
echo "base_url=${BASE_URL}"
echo "deploy_manifest=${DEPLOY_MANIFEST}"
