#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_ROOTFS="${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs"
OVERLAY_ROOTFS="${PROJECT_ROOT}/artifacts/phase6/20260731T0125Z_candidate_14c768b40_decode_metadata_v1/overlay_rootfs"
SOURCE_DIR="${OVERLAY_ROOTFS}/opt/vllm_glm52_v1"
BASE_SOURCE_DIR="${BASE_ROOTFS}/opt/vllm_glm52_v1"
ROTATION_ARTIFACT="${OVERLAY_ROOTFS}/opt/oscar_artifacts/rotation_fit_v2"
RUNTIME_EXPECTATION="${OVERLAY_ROOTFS}/opt/oscar_artifacts/oscar_runtime_expectation.json"
FROZEN_SUITE_DIR="${SUITE_DIR:-${PROJECT_ROOT}/artifacts/phase7/frozen_evaluator_v4_20260728/accuracy_v4_fc374ff4_4aec8ee8/suite}"
PORT="${PORT:-18082}"
LOCK_FILE="${PROJECT_ROOT}/artifacts/phase7/candidate_port_${PORT}.lock"

if [[ "${STAGE7_CANDIDATE_LOCK_HELD:-0}" != "1" ]]; then
  mkdir -p "$(dirname "${LOCK_FILE}")"
  exec 9>"${LOCK_FILE}"
  flock -n 9 || {
    echo "ERROR: another Stage 7 candidate process holds ${LOCK_FILE}" >&2
    exit 1
  }
  export STAGE7_CANDIDATE_LOCK_HELD=1
fi

export PYTHONDONTWRITEBYTECODE=1

native_links=(
  vllm/_C.abi3.so
  vllm/_C_stable_libtorch.abi3.so
  vllm/_moe_C.abi3.so
  vllm/cumem_allocator.abi3.so
  vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
  vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
)
created_links=()

cleanup_links() {
  local path
  for path in "${created_links[@]}"; do
    if [[ -L "${path}" ]]; then
      unlink "${path}"
    fi
  done
}
trap cleanup_links EXIT

for relative in "${native_links[@]}"; do
  runtime_path="${SOURCE_DIR}/${relative}"
  base_path="${BASE_SOURCE_DIR}/${relative}"
  [[ -f "${base_path}" ]] || {
    echo "ERROR: lower-layer native extension is missing: ${base_path}" >&2
    exit 1
  }
  if [[ -e "${runtime_path}" || -L "${runtime_path}" ]]; then
    [[ -L "${runtime_path}" && "$(readlink -f "${runtime_path}")" == "${base_path}" ]] || {
      echo "ERROR: unexpected candidate runtime native path: ${runtime_path}" >&2
      exit 1
    }
  else
    ln -s "${base_path}" "${runtime_path}"
    created_links+=("${runtime_path}")
  fi
done

export MANIFEST="${MANIFEST:-${PROJECT_ROOT}/configs/phase7/oscar_evaluation.json}"
export VERIFY_SCRIPT="${VERIFY_SCRIPT:-${SCRIPT_DIR}/verify_candidate_evaluation.py}"
export SOURCE_DIR
export SUITE_DIR="${FROZEN_SUITE_DIR}"
export RUN_KIND="${RUN_KIND:-oscar_candidate_tp8}"
export ARTIFACT_PHASE="${ARTIFACT_PHASE:-phase7}"
export SERVICE_LABEL="${SERVICE_LABEL:-OSCAR candidate TP=8}"
export PORT
export EXPECTED_MAIN_BRANCH="${EXPECTED_MAIN_BRANCH:-feat/glm52-model-load}"
export EXPECTED_SOURCE_BRANCH="${EXPECTED_SOURCE_BRANCH:-feat/glm52-oscar-integration}"
export EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:-14c768b406b3e39a2d4d5be77a9046ac7ccc26d1}"
export EXPECTED_KV_CACHE_DTYPE="${EXPECTED_KV_CACHE_DTYPE:-oscar_mla_int2}"
export DISABLE_ASYNC_SCHEDULING="${DISABLE_ASYNC_SCHEDULING:-1}"
export CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/artifacts/phase7/cache}"
export VLLM_OSCAR_MLA_ROTATION_ARTIFACT="${ROTATION_ARTIFACT}"
export VLLM_OSCAR_MLA_RUNTIME_EXPECTATION="${RUNTIME_EXPECTATION}"
export RUNTIME_SOURCE_COMMIT="${RUNTIME_SOURCE_COMMIT:-14c768b406b3e39a2d4d5be77a9046ac7ccc26d1}"
export CANDIDATE_MANIFEST_DIGEST="${CANDIDATE_MANIFEST_DIGEST:-sha256:52a74b155567c24ee9875f695f5be96be87397e74217bf04fa6c6582544468e4}"
export CANDIDATE_CONFIG_DIGEST="${CANDIDATE_CONFIG_DIGEST:-sha256:dbd78a779001300cca5f3802a4e69def3de5a82eb96746c614a5e6d7e14f0a99}"
export CANDIDATE_LAYER_DIGEST="${CANDIDATE_LAYER_DIGEST:-sha256:4b9070484b4441fbfd011b68a4d91f102b55d762fb3c0d15a46c500f68a83312}"

"${PROJECT_ROOT}/scripts/phase1/run_native_baseline.sh" "$@"
