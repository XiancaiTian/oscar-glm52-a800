#!/usr/bin/env bash

deploy_unique_space() {
  printf '%s\n' "$@" | tr ', ' '\n' | sed '/^$/d' | sort -u | paste -sd' ' -
}

deploy_space_to_csv() {
  printf '%s\n' "$@" | tr ', ' '\n' | sed '/^$/d' | sort -u | paste -sd, -
}

deploy_host_id() {
  printf '%s' "$1" | tr '.:' '__' | tr '-' '_'
}

load_deploy_config() {
  local loader_dir task_root config_file
  loader_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  task_root="$(cd "${loader_dir}/../.." && pwd)"
  export TASK_ROOT="${TASK_ROOT:-${task_root}}"
  export BASE="${BASE:-$(cd "${TASK_ROOT}/.." && pwd)}"

  config_file="${DEPLOY_CONFIG_FILE:-${TASK_ROOT}/configs/deploy_8th.env}"
  if [[ -f "${config_file}" ]]; then
    # Config files in this deployment package are trusted shell env files.
    # shellcheck disable=SC1090
    source "${config_file}"
  fi
  export DEPLOY_CONFIG_FILE="${config_file}"

  export DEPLOY_LABEL="${DEPLOY_LABEL:-8th}"
  export DEPLOY_GENERATION="${DEPLOY_GENERATION:-8}"
  export SCHEME="${SCHEME:-stage62_glm52_opt_indexshare_rdma_mtp_skipshare_gmem092_2p1d_v2_stable}"

  export MODEL_PATH="${MODEL_PATH:-/nfs/AIED/models/GLM-5.2-FP8}"
  export MODEL_ID="${MODEL_ID:-GLM-5.2-FP8}"
  export IMAGE_TAG="${IMAGE_TAG:-192.168.14.129:80/ae/vllm_openai_glm52:v0.19.0-v2-stable-2p1d-usagefix-20260625_114616}"
  export IMAGE_TAR="${IMAGE_TAR:-${TASK_ROOT}/artifacts/images/vllm_openai_glm52_v2_stable_2p1d_usagefix_20260625_114616.tar}"

  export BUILD_HOST="${BUILD_HOST:-192.168.16.77}"
  export PREFILL0_HEAD="${PREFILL0_HEAD:-192.168.16.63}"
  export PREFILL0_WORKER="${PREFILL0_WORKER:-192.168.16.65}"
  export PREFILL1_HEAD="${PREFILL1_HEAD:-192.168.16.68}"
  export PREFILL1_WORKER="${PREFILL1_WORKER:-192.168.16.69}"
  export DECODE_HEAD="${DECODE_HEAD:-192.168.16.76}"
  export DECODE_WORKER="${DECODE_WORKER:-192.168.16.77}"
  export PREFILL_HEAD="${PREFILL_HEAD:-${PREFILL0_HEAD}}"
  export PREFILL_WORKER="${PREFILL_WORKER:-${PREFILL0_WORKER}}"

  export PROXY_PORT="${PROXY_PORT:-19181}"
  export PREFILL_PORT="${PREFILL_PORT:-19182}"
  export DECODE_PORT="${DECODE_PORT:-19183}"
  export PREFILL_MASTER_PORT="${PREFILL_MASTER_PORT:-32466}"
  export DECODE_MASTER_PORT="${DECODE_MASTER_PORT:-32454}"
  export SIDE_PORT="${SIDE_PORT:-25700}"

  local role_hosts
  role_hosts="$(deploy_unique_space \
    "${BUILD_HOST}" \
    "${PREFILL0_HEAD}" "${PREFILL0_WORKER}" \
    "${PREFILL1_HEAD}" "${PREFILL1_WORKER}" \
    "${DECODE_HEAD}" "${DECODE_WORKER}")"
  export TARGET_HOSTS="${TARGET_HOSTS:-${role_hosts}}"
  export ALLOWED_HOSTS="${ALLOWED_HOSTS:-$(deploy_space_to_csv "${TARGET_HOSTS}")}"
  export EXPECTED_HOSTS_CSV="${EXPECTED_HOSTS_CSV:-$(deploy_space_to_csv "${TARGET_HOSTS}")}"
  export FORBIDDEN_HOSTS_CSV="${FORBIDDEN_HOSTS_CSV:-}"

  export PROXY_ROOT="${PROXY_ROOT:-http://${DECODE_HEAD}:${PROXY_PORT}}"
  export OPENAI_BASE="${OPENAI_BASE:-${PROXY_ROOT}/v1}"
  export MESSAGES_ENDPOINT="${MESSAGES_ENDPOINT:-${OPENAI_BASE}/messages}"
  export MESSAGES_COUNT_ENDPOINT="${MESSAGES_COUNT_ENDPOINT:-${OPENAI_BASE}/messages/count_tokens}"
  export P0_PREFILL="${P0_PREFILL:-${PREFILL0_HEAD}:${PREFILL_PORT}}"
  export P1_PREFILL="${P1_PREFILL:-${PREFILL1_HEAD}:${PREFILL_PORT}}"
  export PREFILLS="${PREFILLS:-${P0_PREFILL},${P1_PREFILL}}"
  export DECODE="${DECODE:-${DECODE_HEAD}:${DECODE_PORT}}"

  export PREFILL_HEAD_CONTAINER="${PREFILL_HEAD_CONTAINER:-glm52_v2_stable_${DEPLOY_LABEL}_p0_prefill_$(deploy_host_id "${PREFILL0_HEAD}")}"
  export PREFILL_WORKER_CONTAINER="${PREFILL_WORKER_CONTAINER:-glm52_v2_stable_${DEPLOY_LABEL}_p0_prefill_$(deploy_host_id "${PREFILL0_WORKER}")}"
  export PREFILL1_HEAD_CONTAINER="${PREFILL1_HEAD_CONTAINER:-glm52_v2_stable_${DEPLOY_LABEL}_p1_prefill_$(deploy_host_id "${PREFILL1_HEAD}")}"
  export PREFILL1_WORKER_CONTAINER="${PREFILL1_WORKER_CONTAINER:-glm52_v2_stable_${DEPLOY_LABEL}_p1_prefill_$(deploy_host_id "${PREFILL1_WORKER}")}"
  export DECODE_HEAD_CONTAINER="${DECODE_HEAD_CONTAINER:-glm52_v2_stable_${DEPLOY_LABEL}_decode_$(deploy_host_id "${DECODE_HEAD}")}"
  export DECODE_WORKER_CONTAINER="${DECODE_WORKER_CONTAINER:-glm52_v2_stable_${DEPLOY_LABEL}_decode_$(deploy_host_id "${DECODE_WORKER}")}"
  export TASK_ROOT_IN_CONTAINER="${TASK_ROOT_IN_CONTAINER:-/workspace/$(basename "${TASK_ROOT}")}"
  export CONTAINER_INSTALL_ROOT="${CONTAINER_INSTALL_ROOT:-/opt/glm52_speed_up_v2_stable_${DEPLOY_LABEL}}"

  export SSH_USER="${SSH_USER:-zhanghong}"
  export IFACE="${IFACE:-ens22f0}"
  export HOST_IFACE_OVERRIDES="${HOST_IFACE_OVERRIDES:-}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_6,mlx5_7}"
  export NCCL_PROTO="${NCCL_PROTO:-Simple}"
  export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc}"
  export UCX_DEVICES="${UCX_DEVICES:-mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1}"
  export UCX_NET_DEVICES="${UCX_NET_DEVICES:-${UCX_DEVICES}}"
  export UCX_MAX_RNDV_RAILS="${UCX_MAX_RNDV_RAILS:-4}"
  export UCX_MAX_EAGER_RAILS="${UCX_MAX_EAGER_RAILS:-4}"
  export UCX_CM_USE_ALL_DEVICES="${UCX_CM_USE_ALL_DEVICES:-y}"

  export SHAPE_LIST_FILE="${SHAPE_LIST_FILE:-${TASK_ROOT}/configs/warmup_shapes_stage_90k_v2.txt}"
  export EXPECTED_STAGE_90K_SHAPES="${EXPECTED_STAGE_90K_SHAPES:-92160,184320,200704}"
  export EXPECTED_STAGE_90K_SHAPES_SHA256="${EXPECTED_STAGE_90K_SHAPES_SHA256:-}"
  export PROXY_SOURCE="${PROXY_SOURCE:-${TASK_ROOT}/source/vllm_glm52_v1/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py}"
}
