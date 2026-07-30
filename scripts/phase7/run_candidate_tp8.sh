#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_ROOTFS="${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs"
OVERLAY_ROOTFS="${PROJECT_ROOT}/artifacts/phase6/20260730T2333Z_candidate_35ab18464_headgroup_v3/overlay_rootfs"
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
export EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:-35ab1846447fc86b4b2177e76c5939503cc3701b}"
export EXPECTED_KV_CACHE_DTYPE="${EXPECTED_KV_CACHE_DTYPE:-oscar_mla_int2}"
export DISABLE_ASYNC_SCHEDULING="${DISABLE_ASYNC_SCHEDULING:-1}"
export CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/artifacts/phase7/cache}"
export VLLM_OSCAR_MLA_ROTATION_ARTIFACT="${ROTATION_ARTIFACT}"
export VLLM_OSCAR_MLA_RUNTIME_EXPECTATION="${RUNTIME_EXPECTATION}"
export RUNTIME_SOURCE_COMMIT="${RUNTIME_SOURCE_COMMIT:-35ab1846447fc86b4b2177e76c5939503cc3701b}"
export CANDIDATE_MANIFEST_DIGEST="${CANDIDATE_MANIFEST_DIGEST:-sha256:a629a99ec78a90562ff506709c176e98c324f1c373378ac038a3ce0c06de9105}"
export CANDIDATE_CONFIG_DIGEST="${CANDIDATE_CONFIG_DIGEST:-sha256:6b5aeb4b1b26c8012062163d510fc5a58255c85a71a602d83f9affc386a7bb59}"
export CANDIDATE_LAYER_DIGEST="${CANDIDATE_LAYER_DIGEST:-sha256:2ec5ec198ccbdfb92712143087b0e0c57d6269e59b88d1724496f32986e206bd}"

"${PROJECT_ROOT}/scripts/phase1/run_native_baseline.sh" "$@"
