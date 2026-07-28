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

export MANIFEST="${PROJECT_ROOT}/configs/phase1/native_baseline.json"
export VERIFY_SCRIPT="${SCRIPT_DIR}/verify_native_performance.py"
export RUN_KIND="native_stage9_tp8"
export ARTIFACT_PHASE="phase9"
export SERVICE_LABEL="Native Stage 9 TP=8"
export PORT="${PORT:-18083}"
export EXPECTED_MAIN_BRANCH="feat/glm52-model-load"
export EXPECTED_SOURCE_BRANCH="feat/glm52-oscar-integration"
export EXPECTED_SOURCE_COMMIT="a3317695428819d41437b1cb144404b3bfc05a92"
export EXPECTED_KV_CACHE_DTYPE="auto"
export DISABLE_ASYNC_SCHEDULING=1
export MAX_MODEL_LEN=131072
export PROFILER_CONFIG="$(
  python3 "${SCRIPT_DIR}/build_profiler_config.py" \
    --profile-dir "${STAGE9_PROFILE_DIR}"
)"
export SUITE_DIR="${FROZEN_ACCURACY_SUITE}"
export CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/artifacts/phase9/cache}"
export OSCAR_RUNTIME_PROJECT_ROOT="${OSCAR_RUNTIME_PROJECT_ROOT:-${PROJECT_ROOT}}"

"${PROJECT_ROOT}/scripts/phase1/run_native_baseline.sh" "$@"
