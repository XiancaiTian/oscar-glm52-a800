#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/load_deploy_config.sh"
load_deploy_config

MODE="${1:-run}"
RUN_ID="${RUN_ID:-deploy_${DEPLOY_LABEL}_current_verified_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${TASK_ROOT}/logs/deploy"
RAW_DIR="${TASK_ROOT}/reports/raw"
ACCEPT_DIR="${TASK_ROOT}/reports/acceptance"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
DEPLOY_EXECUTOR="${SCRIPT_DIR}/deploy_stage_90k_v2.sh"

usage() {
  cat <<'EOF'
Usage:
  deploy_8th_current_verified.sh [run|print-config|--print-config|--dry-run]

This is the configurable 8th deployment entrypoint. It validates:
  - role hosts, TARGET_HOSTS, ALLOWED_HOSTS, and EXPECTED_HOSTS_CSV are consistent
  - prefix-cache: enabled
  - prefill shape bucket: 90k padding, max 202752
  - shape warmup list: 92160/184320/200704
  - proxy context-length guard: enabled, max_model_len 202752
  - proxy ClaudeCLI-2.1.204 system-role compatibility: enabled
  - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: 0

Environment overrides:
  DEPLOY_CONFIG_FILE  Override config file path.
  RUN_ID       Override wrapper log/report id.
  SSH_USER     Override remote SSH user if needed.
EOF
}

expected_hosts_csv() {
  printf '%s' "${EXPECTED_HOSTS_CSV}" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -
}

actual_hosts_csv() {
  printf '%s\n' \
    ${TARGET_HOSTS} \
    "${BUILD_HOST}" \
    "${PREFILL_HEAD}" \
    "${PREFILL_WORKER}" \
    "${PREFILL0_HEAD}" \
    "${PREFILL0_WORKER}" \
    "${PREFILL1_HEAD}" \
    "${PREFILL1_WORKER}" \
    "${DECODE_HEAD}" \
    "${DECODE_WORKER}" | sed '/^$/d' | sort -u | paste -sd, -
}

allowed_env_csv() {
  printf '%s' "${ALLOWED_HOSTS}" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -
}

shape_values_csv() {
  paste -sd, "${SHAPE_LIST_FILE}"
}

assert_allowed_hosts() {
  local expected actual allowed
  expected="$(expected_hosts_csv)"
  actual="$(actual_hosts_csv)"
  allowed="$(allowed_env_csv)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR: deploy target hosts must be exactly ${expected}; actual=${actual}" >&2
    exit 2
  fi
  if [[ "${allowed}" != "${expected}" ]]; then
    echo "ERROR: ALLOWED_HOSTS must be exactly ${expected}; actual=${allowed}" >&2
    exit 2
  fi
  [[ -z "${FORBIDDEN_HOSTS_CSV}" ]] && return 0
  IFS=',' read -r -a forbidden_hosts <<< "${FORBIDDEN_HOSTS_CSV}"
  for host in "${forbidden_hosts[@]}"; do
    [[ -n "${host}" ]] || continue
    if grep -Fq "${host}" <<< "${actual},${ALLOWED_HOSTS},http://${DECODE_HEAD}:${PROXY_PORT}/v1"; then
      echo "ERROR: forbidden host appears in active ${DEPLOY_LABEL} config: ${host}" >&2
      exit 2
    fi
  done
}

assert_shape_list() {
  if [[ ! -f "${SHAPE_LIST_FILE}" ]]; then
    echo "ERROR: missing shape list file: ${SHAPE_LIST_FILE}" >&2
    exit 3
  fi

  local actual_sha actual_values
  actual_sha="$(sha256sum "${SHAPE_LIST_FILE}" | awk '{print $1}')"
  actual_values="$(shape_values_csv)"
  if [[ "${actual_values}" != "${EXPECTED_STAGE_90K_SHAPES}" ]]; then
    echo "ERROR: shape list mismatch: expected=${EXPECTED_STAGE_90K_SHAPES} actual=${actual_values}" >&2
    exit 3
  fi
  if [[ "${actual_sha}" != "${EXPECTED_STAGE_90K_SHAPES_SHA256}" ]]; then
    echo "ERROR: shape list checksum mismatch: expected=${EXPECTED_STAGE_90K_SHAPES_SHA256} actual=${actual_sha}" >&2
    exit 3
  fi
}

assert_frozen_config() {
  local errors=0
  [[ "${V2_ENABLE_PREFIX_CACHING}" == "1" ]] || { echo "ERROR: V2_ENABLE_PREFIX_CACHING must be 1" >&2; errors=1; }
  [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "1" ]] || { echo "ERROR: VLLM_ENABLE_PREFIX_CACHING must be 1" >&2; errors=1; }
  [[ "${VLLM_PREFILL_SHAPE_BUCKET_MULTIPLE}" == "92160" ]] || { echo "ERROR: shape bucket multiple must be 92160" >&2; errors=1; }
  [[ "${VLLM_PREFILL_SHAPE_BUCKET_MAX}" == "202752" ]] || { echo "ERROR: shape bucket max must be 202752" >&2; errors=1; }
  [[ "${PREFILL_PROXY_CONTEXT_LENGTH_GUARD}" == "1" ]] || { echo "ERROR: proxy context-length guard must be enabled" >&2; errors=1; }
  [[ "${PREFILL_PROXY_MAX_MODEL_LEN}" == "202752" ]] || { echo "ERROR: proxy max model len must be 202752" >&2; errors=1; }
  [[ "${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS}" == "0" ]] || { echo "ERROR: cudagraph memory profiler estimate must be 0" >&2; errors=1; }
  [[ "${SAFETENSORS_LOAD_STRATEGY}" == "lazy" ]] || { echo "ERROR: SAFETENSORS_LOAD_STRATEGY must be lazy" >&2; errors=1; }
  [[ "${PREFILL_SAFETENSORS_LOAD_STRATEGY}" == "lazy" ]] || { echo "ERROR: PREFILL_SAFETENSORS_LOAD_STRATEGY must be lazy" >&2; errors=1; }
  [[ "${DECODE_SAFETENSORS_LOAD_STRATEGY}" == "lazy" ]] || { echo "ERROR: DECODE_SAFETENSORS_LOAD_STRATEGY must be lazy" >&2; errors=1; }
  if [[ "${errors}" != "0" ]]; then
    exit 4
  fi
}

assert_proxy_claude204_patch() {
  if [[ ! -f "${PROXY_SOURCE}" ]]; then
    echo "ERROR: missing proxy source: ${PROXY_SOURCE}" >&2
    exit 4
  fi
  grep -Fq "def _normalize_anthropic_system_messages" "${PROXY_SOURCE}"
  grep -Fq "PROXY_ANTHROPIC_SYSTEM_NORMALIZED" "${PROXY_SOURCE}"
  grep -Fq "PROXY_PREFILL_INTERNAL_WARMUP_ALLOWED" "${PROXY_SOURCE}"
  grep -Fq 'endpoint == "/messages/count_tokens"' "${PROXY_SOURCE}"
  grep -Fq 'api == "/messages"' "${PROXY_SOURCE}"
  python3 -m py_compile "${PROXY_SOURCE}"
}

print_config() {
  cat <<EOF
run_id=${RUN_ID}
task_root=${TASK_ROOT}
deploy_config_file=${DEPLOY_CONFIG_FILE}
deploy_label=${DEPLOY_LABEL}
log_file=${LOG_FILE}
deploy_executor=${DEPLOY_EXECUTOR}
target_hosts=$(expected_hosts_csv)
allowed_hosts=${ALLOWED_HOSTS}
model_id=${MODEL_ID}
model_path=${MODEL_PATH}
image_tag=${IMAGE_TAG}
image_tar=${IMAGE_TAR}
build_host=${BUILD_HOST}
prefill0=${PREFILL0_HEAD},${PREFILL0_WORKER}
prefill1=${PREFILL1_HEAD},${PREFILL1_WORKER}
decode=${DECODE_HEAD},${DECODE_WORKER}
iface=${IFACE}
host_iface_overrides=${HOST_IFACE_OVERRIDES}
nccl_ib_hca=${NCCL_IB_HCA}
ucx_tls=${UCX_TLS}
ucx_devices=${UCX_DEVICES}
ucx_net_devices=${UCX_NET_DEVICES}
containers=${PREFILL_HEAD_CONTAINER},${PREFILL_WORKER_CONTAINER},${PREFILL1_HEAD_CONTAINER},${PREFILL1_WORKER_CONTAINER},${DECODE_HEAD_CONTAINER},${DECODE_WORKER_CONTAINER}
proxy_endpoint=http://${DECODE_HEAD}:${PROXY_PORT}/v1
prefix_cache=${V2_ENABLE_PREFIX_CACHING}
online_formal_deploy=${V2_ONLINE_FORMAL_DEPLOY}
patch_cache_suffix=${V2_PATCH_CACHE_SUFFIX}
shape_padding_multiple=${VLLM_PREFILL_SHAPE_BUCKET_MULTIPLE}
shape_bucket_max=${VLLM_PREFILL_SHAPE_BUCKET_MAX}
shape_warmup_values=$(shape_values_csv)
shape_list=${SHAPE_LIST_FILE}
shape_list_sha256=${EXPECTED_STAGE_90K_SHAPES_SHA256}
context_length_guard=${PREFILL_PROXY_CONTEXT_LENGTH_GUARD}
proxy_max_model_len=${PREFILL_PROXY_MAX_MODEL_LEN}
claude204_proxy_compat=enabled
claude204_proxy_marker=PROXY_ANTHROPIC_SYSTEM_NORMALIZED
prefill_internal_warmup=enabled
proxy_source=${PROXY_SOURCE}
memory_profiler_estimate_cudagraphs=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS}
safetensors_load_strategy=${SAFETENSORS_LOAD_STRATEGY}
stop_non_task_containers=${STOP_NON_TASK_CONTAINERS}
responses_api_verification=skipped_by_instruction
EOF
}

write_wrapper_acceptance() {
  local wrapper_acceptance latest_deploy_acceptance latest_manifest
  wrapper_acceptance="${ACCEPT_DIR}/${RUN_ID}_acceptance.json"
  latest_deploy_acceptance="$(ls -1t "${ACCEPT_DIR}"/*_stage_90k_deploy_script_acceptance.json 2>/dev/null | head -n 1 || true)"
  latest_manifest="$(ls -1t "${RAW_DIR}"/*_deploy_manifest.json 2>/dev/null | head -n 1 || true)"
  python3 - "${wrapper_acceptance}" "${RUN_ID}" "${LOG_FILE}" "${latest_deploy_acceptance}" "${latest_manifest}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

out_path, run_id, log_file, deploy_acceptance, manifest = sys.argv[1:6]
payload = {
    "run_id": run_id,
    "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "gate": "deploy_8th_current_verified_wrapper",
    "pass": True,
    "log_file": log_file,
    "deploy_acceptance": deploy_acceptance or None,
    "deploy_manifest": manifest or None,
    "deploy_config_file": os.environ.get("DEPLOY_CONFIG_FILE"),
    "deploy_label": os.environ.get("DEPLOY_LABEL"),
    "target_hosts": [
        host
        for host in os.environ.get("EXPECTED_HOSTS_CSV", "").split(",")
        if host
    ],
    "model_id": os.environ.get("MODEL_ID"),
    "model_path": os.environ.get("MODEL_PATH"),
    "image_tag": os.environ.get("IMAGE_TAG"),
    "containers": [
        os.environ.get("PREFILL_HEAD_CONTAINER"),
        os.environ.get("PREFILL_WORKER_CONTAINER"),
        os.environ.get("PREFILL1_HEAD_CONTAINER"),
        os.environ.get("PREFILL1_WORKER_CONTAINER"),
        os.environ.get("DECODE_HEAD_CONTAINER"),
        os.environ.get("DECODE_WORKER_CONTAINER"),
    ],
    "prefix_cache": "enabled",
    "shape_warmup_values": [92160, 184320, 200704],
    "proxy_context_length_guard": True,
    "proxy_max_model_len": 202752,
    "claude204_proxy_compat": True,
    "claude204_proxy_marker": "PROXY_ANTHROPIC_SYSTEM_NORMALIZED",
    "prefill_internal_warmup": True,
    "responses_api_verification": "skipped_by_instruction",
    "memory_profiler_estimate_cudagraphs": 0,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

case "${MODE}" in
  run)
    ;;
  print-config|--print-config|--dry-run)
    assert_allowed_hosts
    assert_shape_list
    assert_frozen_config
    assert_proxy_claude204_patch
    print_config
    exit 0
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

assert_allowed_hosts
assert_shape_list
assert_frozen_config
assert_proxy_claude204_patch
mkdir -p "${LOG_DIR}" "${RAW_DIR}" "${ACCEPT_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  print_config
  echo
  echo "[deploy] executing configurable ${DEPLOY_LABEL} stage_90k deployment"
  "${DEPLOY_EXECUTOR}"
  echo
  echo "[acceptance] recording configurable wrapper evidence"
  write_wrapper_acceptance
  echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${LOG_FILE}"
