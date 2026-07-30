#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_ROOTFS="${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs"
OVERLAY_ROOTFS="${PROJECT_ROOT}/artifacts/phase6/20260730T1928Z_candidate_98ddd3f4e_decode_fastpath/overlay_rootfs"
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
export EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:-98ddd3f4ef645bddec76d96cd86a11d17232aaa2}"
export EXPECTED_KV_CACHE_DTYPE="${EXPECTED_KV_CACHE_DTYPE:-oscar_mla_int2}"
export DISABLE_ASYNC_SCHEDULING="${DISABLE_ASYNC_SCHEDULING:-1}"
export CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/artifacts/phase7/cache}"
export VLLM_OSCAR_MLA_ROTATION_ARTIFACT="${ROTATION_ARTIFACT}"
export VLLM_OSCAR_MLA_RUNTIME_EXPECTATION="${RUNTIME_EXPECTATION}"
export RUNTIME_SOURCE_COMMIT="${RUNTIME_SOURCE_COMMIT:-98ddd3f4ef645bddec76d96cd86a11d17232aaa2}"
export CANDIDATE_MANIFEST_DIGEST="${CANDIDATE_MANIFEST_DIGEST:-sha256:96455a6e4d0eaccde087957f56061c3442db15a9a9c7a8a60395b9c7764a69f9}"
export CANDIDATE_CONFIG_DIGEST="${CANDIDATE_CONFIG_DIGEST:-sha256:e90f4a84cd0aa958b86236d4d982a735f8d6a903c82168fee1093ff357bdcbf7}"
export CANDIDATE_LAYER_DIGEST="${CANDIDATE_LAYER_DIGEST:-sha256:badb252b2e916f4d84d23253b28d72d90e74c52c7d009f73bdcbb0ca17f4f538}"

"${PROJECT_ROOT}/scripts/phase1/run_native_baseline.sh" "$@"
