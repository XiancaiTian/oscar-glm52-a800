#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy_common_env.sh"

mkdir -p "${TASK_ROOT}/logs/deploy"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG="${TASK_ROOT}/logs/deploy/${TS}_load_image_all.log"

if [[ ! -s "${IMAGE_TAR}" ]]; then
  echo "missing image tar: ${IMAGE_TAR}" >&2
  exit 2
fi

for host in ${TARGET_HOSTS}; do
  echo "[$(date -u +%FT%TZ)] load image on ${host}" | tee -a "${LOG}"
  remote "${host}" "if docker image inspect ${IMAGE_TAG} >/dev/null 2>&1; then echo image_present=${IMAGE_TAG}; else docker load -i ${IMAGE_TAR}; fi; docker image inspect ${IMAGE_TAG} >/dev/null" 2>&1 | tee -a "${LOG}"
done

record_acceptance "image_load" "PASS" "hosts=${TARGET_HOSTS} image=${IMAGE_TAG} tar=${IMAGE_TAR}"
echo "image_load_log=${LOG}"
