#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MANIFEST="${PROJECT_ROOT}/configs/phase5/oscar_tp8.json"
export VERIFY_SCRIPT="${SCRIPT_DIR}/verify_oscar_tp8.py"
export SOURCE_DIR="${PROJECT_ROOT}/glm52_oscar_vllm"
export RUN_KIND="oscar_tp8"
export ARTIFACT_PHASE="phase5"
export SERVICE_LABEL="OSCAR TP=8"
export PORT="${PORT:-18081}"
export EXPECTED_MAIN_BRANCH="feat/glm52-model-load"
export EXPECTED_SOURCE_BRANCH="feat/glm52-oscar-integration"
export EXPECTED_SOURCE_COMMIT="c3823fda2ed1d82f92c99275b6e128bac9ba6220"
export EXPECTED_KV_CACHE_DTYPE="oscar_mla_int2"
export DISABLE_ASYNC_SCHEDULING=1
export CACHE_ROOT="${PROJECT_ROOT}/artifacts/phase5/cache"
export VLLM_OSCAR_MLA_ROTATION_ARTIFACT="${PROJECT_ROOT}/artifacts/phase2/20260726T1200Z_rotation_fit_v2"
export VLLM_OSCAR_MLA_RUNTIME_EXPECTATION="${PROJECT_ROOT}/configs/phase5/oscar_runtime_expectation.json"

exec "${PROJECT_ROOT}/scripts/phase1/run_native_baseline.sh" "$@"
