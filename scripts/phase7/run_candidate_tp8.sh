#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_ROOTFS="${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs"
OVERLAY_ROOTFS="${PROJECT_ROOT}/artifacts/phase6/20260726T151933Z_candidate_7d317f1de/overlay_rootfs"
SOURCE_DIR="${OVERLAY_ROOTFS}/opt/vllm_glm52_v1"
BASE_SOURCE_DIR="${BASE_ROOTFS}/opt/vllm_glm52_v1"
ROTATION_ARTIFACT="${OVERLAY_ROOTFS}/opt/oscar_artifacts/rotation_fit_v2"

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

export MANIFEST="${PROJECT_ROOT}/configs/phase7/oscar_evaluation.json"
export VERIFY_SCRIPT="${SCRIPT_DIR}/verify_candidate_evaluation.py"
export SOURCE_DIR
export RUN_KIND="oscar_candidate_tp8"
export ARTIFACT_PHASE="phase7"
export SERVICE_LABEL="OSCAR candidate TP=8"
export PORT="${PORT:-18082}"
export EXPECTED_MAIN_BRANCH="feat/glm52-model-load"
export EXPECTED_SOURCE_BRANCH="feat/glm52-oscar-integration"
export EXPECTED_SOURCE_COMMIT="7d317f1dee21af9d49445878bcc9c2d181d041c9"
export EXPECTED_KV_CACHE_DTYPE="oscar_mla_int2"
export DISABLE_ASYNC_SCHEDULING=1
export CACHE_ROOT="${PROJECT_ROOT}/artifacts/phase7/cache"
export VLLM_OSCAR_MLA_ROTATION_ARTIFACT="${ROTATION_ARTIFACT}"
export VLLM_OSCAR_MLA_RUNTIME_EXPECTATION="${PROJECT_ROOT}/configs/phase5/oscar_runtime_expectation.json"
export RUNTIME_SOURCE_COMMIT="7d317f1dee21af9d49445878bcc9c2d181d041c9"
export CANDIDATE_MANIFEST_DIGEST="sha256:c2939feb779757c8f4c7a500300085b1ddee287975941588a19c305e3b602ec9"
export CANDIDATE_CONFIG_DIGEST="sha256:5ad3094114d68778cd743e971653aaf62f3c2e0464a2e69c62c28120e2145f7c"
export CANDIDATE_LAYER_DIGEST="sha256:8ad9ace913c624cee36efef0d514197c48ec0e8cadf01b8c7ffd253c46805225"

"${PROJECT_ROOT}/scripts/phase1/run_native_baseline.sh" "$@"
