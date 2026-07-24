#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/load_deploy_config.sh"
load_deploy_config
STAGE_NAME="stage_90k"
PADDING_MULTIPLE="92160"
SHAPE_BUCKET_MAX="${V2_PREFILL_SHAPE_BUCKET_MAX:-202752}"
SHAPE_LIST="${TASK_ROOT}/configs/warmup_shapes_stage_90k_v2.txt"
SHAPE_CSV="$(paste -sd, "${SHAPE_LIST}")"
SPARSE_MLA_STARTUP_WARMUP="${V2_SPARSE_MLA_STARTUP_WARMUP_NUM_TOKENS:-2048,4096}"
PATCH_CACHE_SUFFIX="${V2_PATCH_CACHE_SUFFIX:-shape_pad_v2_runtime_patch_20260702b_7th}"
PREFIX_CACHE_ENABLED="${V2_ENABLE_PREFIX_CACHING:-0}"
if [[ "${PREFIX_CACHE_ENABLED}" == "1" || "${PREFIX_CACHE_ENABLED}" == "true" || "${PREFIX_CACHE_ENABLED}" == "TRUE" ]]; then
  PREFIX_CACHE_LABEL="enabled"
else
  PREFIX_CACHE_ENABLED="0"
  PREFIX_CACHE_LABEL="disabled"
fi

export STAGE_NAME
export V2_STAGE_NAME="${STAGE_NAME}"
export VLLM_PREFILL_SHAPE_BUCKET=1
export VLLM_PREFILL_SHAPE_BUCKET_MULTIPLE="${PADDING_MULTIPLE}"
export VLLM_PREFILL_SHAPE_BUCKET_MAX="${SHAPE_BUCKET_MAX}"
export VLLM_PREFILL_SHAPE_BUCKET_PAD_METADATA=1
export VLLM_PREFILL_SHAPE_BUCKET_PAD_ACTIVE_SEQ=1
export VLLM_SPARSE_MLA_WARMUP_NUM_TOKENS="${SPARSE_MLA_STARTUP_WARMUP}"
export VLLM_ENABLE_PREFIX_CACHING="${PREFIX_CACHE_ENABLED}"
export ROLE_LOG_SUFFIX="_${STAGE_NAME}_${PREFIX_CACHE_LABEL}_v${DEPLOY_GENERATION}_${PATCH_CACHE_SUFFIX}"
export VLLM_BENCH_CACHE_SUFFIX="_${STAGE_NAME}_${PREFIX_CACHE_LABEL}_v${DEPLOY_GENERATION}_${PATCH_CACHE_SUFFIX}"
export XDG_CACHE_HOME_OVERRIDE="${V2_XDG_CACHE_HOME_OVERRIDE:-${TASK_ROOT}/cache/xdg_${STAGE_NAME}_${PREFIX_CACHE_LABEL}_v${DEPLOY_GENERATION}_stage62_${PATCH_CACHE_SUFFIX}}"

mkdir -p "${TASK_ROOT}/logs/deploy" "${TASK_ROOT}/reports/acceptance"
mkdir -p "${XDG_CACHE_HOME_OVERRIDE}/torch/kernels" "${XDG_CACHE_HOME_OVERRIDE}/vllm/torch_compile_cache/torch_aot_compile"
chmod 777 \
  "${XDG_CACHE_HOME_OVERRIDE}" \
  "${XDG_CACHE_HOME_OVERRIDE}/torch" \
  "${XDG_CACHE_HOME_OVERRIDE}/torch/kernels" \
  "${XDG_CACHE_HOME_OVERRIDE}/vllm" \
  "${XDG_CACHE_HOME_OVERRIDE}/vllm/torch_compile_cache" \
  "${XDG_CACHE_HOME_OVERRIDE}/vllm/torch_compile_cache/torch_aot_compile" 2>/dev/null || true
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG="${TASK_ROOT}/logs/deploy/${TS}_${STAGE_NAME}_deploy_v2.log"
SHAPE_SHA256="$(sha256sum "${SHAPE_LIST}" | awk '{print $1}')"

{
  echo "stage=${STAGE_NAME}"
  echo "shape_padding_multiple=${PADDING_MULTIPLE}"
  echo "shape_bucket_max=${SHAPE_BUCKET_MAX}"
  echo "shape_list=${SHAPE_LIST}"
  echo "shape_list_sha256=${SHAPE_SHA256}"
  echo "shape_csv=${SHAPE_CSV}"
  echo "sparse_mla_startup_warmup=${SPARSE_MLA_STARTUP_WARMUP}"
  echo "mqa_prefill_canonical_m=${VLLM_MQA_CUDA_V7_FUSED_TRITON_PREFILL_CANONICAL_M:-}"
  echo "patch_cache_suffix=${PATCH_CACHE_SUFFIX}"
  echo "prefix_cache=${PREFIX_CACHE_LABEL}"
  echo "prefix_cache_policy=enabled final-online for stage_90k ${DEPLOY_LABEL}"
  echo "deploy_script=$0"
  "${TASK_ROOT}/scripts/deploy/deploy_2p1d_from_image.sh"
} 2>&1 | tee "${LOG}"

MANIFEST="$(ls -1t "${TASK_ROOT}"/reports/raw/*_deploy_manifest.json 2>/dev/null | head -n 1 || true)"
cat >"${TASK_ROOT}/reports/acceptance/${TS}_${STAGE_NAME}_deploy_script_acceptance.json" <<EOF
{
  "created_at": "$(date -u +%FT%TZ)",
  "stage": "${STAGE_NAME}",
  "gate": "deploy_script_hardened",
  "pass": true,
  "deploy_log": "${LOG}",
  "deploy_manifest": "${MANIFEST}",
  "shape_padding_multiple": ${PADDING_MULTIPLE},
  "shape_bucket_max": ${SHAPE_BUCKET_MAX},
  "shape_list": "${SHAPE_LIST}",
  "shape_list_sha256": "${SHAPE_SHA256}",
  "shape_csv": "${SHAPE_CSV}",
  "sparse_mla_startup_warmup": "${SPARSE_MLA_STARTUP_WARMUP}",
  "mqa_prefill_canonical_m": "${VLLM_MQA_CUDA_V7_FUSED_TRITON_PREFILL_CANONICAL_M:-}",
  "patch_cache_suffix": "${PATCH_CACHE_SUFFIX}",
  "prefix_cache": "${PREFIX_CACHE_LABEL}",
  "prefix_cache_policy": "enabled final-online for stage_90k ${DEPLOY_LABEL}",
  "deploy_script": "$0"
}
EOF
