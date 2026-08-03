#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FROZEN_ACCURACY_SUITE="${PROJECT_ROOT}/artifacts/phase7/frozen_evaluator_v4_20260728/accuracy_v4_fc374ff4_4aec8ee8/suite"
STAGE9_PROFILE_DIR="${STAGE9_PROFILE_DIR:?set an absolute Stage 9 profiler directory}"
[[ "${STAGE9_PROFILE_DIR}" == /* ]] || {
  echo "ERROR: STAGE9_PROFILE_DIR must be absolute" >&2
  exit 1
}
STAGE9_PROFILE_DIR="$(realpath -m -- "${STAGE9_PROFILE_DIR}")"
[[ "${STAGE9_PROFILE_DIR}" == "${PROJECT_ROOT}/artifacts/"* ||
  "${STAGE9_PROFILE_DIR}" == /dev/shm/* ]] || {
  echo "ERROR: STAGE9_PROFILE_DIR must be under project artifacts/ or /dev/shm/" >&2
  exit 1
}
mkdir -p "${STAGE9_PROFILE_DIR}"
if [[ "${1:-}" == "serve" ]] &&
  find "${STAGE9_PROFILE_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "ERROR: STAGE9_PROFILE_DIR must be empty before a formal server run" >&2
  exit 1
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
[[ "${ARTIFACT_ROOT}" == /* ]] || {
  echo "ERROR: ARTIFACT_ROOT must be absolute" >&2
  exit 1
}
ARTIFACT_ROOT="$(realpath -m -- "${ARTIFACT_ROOT}")"
[[ "${ARTIFACT_ROOT}" == "${PROJECT_ROOT}/artifacts" ||
  "${ARTIFACT_ROOT}" == "${PROJECT_ROOT}/artifacts/"* ||
  "${ARTIFACT_ROOT}" == /dev/shm ||
  "${ARTIFACT_ROOT}" == /dev/shm/* ]] || {
  echo "ERROR: ARTIFACT_ROOT must be under project artifacts/ or /dev/shm/" >&2
  exit 1
}
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/artifacts/phase9/cache}"
[[ "${CACHE_ROOT}" == /* ]] || {
  echo "ERROR: CACHE_ROOT must be absolute" >&2
  exit 1
}
CACHE_ROOT="$(realpath -m -- "${CACHE_ROOT}")"
[[ "${CACHE_ROOT}" == "${PROJECT_ROOT}/artifacts/"* ||
  "${CACHE_ROOT}" == /dev/shm/* ]] || {
  echo "ERROR: CACHE_ROOT must be under project artifacts/ or /dev/shm/" >&2
  exit 1
}
if [[ -n "${RUN_ID:-}" && ! "${RUN_ID}" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]]; then
  echo "ERROR: RUN_ID contains unsafe characters" >&2
  exit 1
fi

export MANIFEST="${PROJECT_ROOT}/configs/phase1/native_baseline.json"
export VERIFY_SCRIPT="${SCRIPT_DIR}/verify_native_performance.py"
export RUN_KIND="native_stage9_tp8"
export ARTIFACT_PHASE="phase9"
export SERVICE_LABEL="Native Stage 9 TP=8"
export PORT="${PORT:-18083}"
export EXPECTED_MAIN_BRANCH="feat/glm52-model-load"
export EXPECTED_SOURCE_BRANCH="feat/glm52-oscar-integration"
export EXPECTED_SOURCE_COMMIT="ea8ae6b7758ae2b4db7cae44d638ae5de80148ac"
export EXPECTED_KV_CACHE_DTYPE="auto"
export DISABLE_ASYNC_SCHEDULING=1
export MAX_MODEL_LEN=131072
export ARTIFACT_ROOT
export CACHE_ROOT
export CANDIDATE_ROOTFS="${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs"
export SOURCE_DIR="${PROJECT_ROOT}/glm52_oscar_vllm"
export PROFILER_CONFIG="$(
  python3 "${SCRIPT_DIR}/build_profiler_config.py" \
    --profile-dir "${STAGE9_PROFILE_DIR}"
)"
export SUITE_DIR="${FROZEN_ACCURACY_SUITE}"
export OSCAR_RUNTIME_PROJECT_ROOT="${PROJECT_ROOT}"

"${PROJECT_ROOT}/scripts/phase1/run_native_baseline.sh" "$@"
