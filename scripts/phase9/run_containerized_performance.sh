#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_REPO="${PROJECT_ROOT}/glm52_oscar_vllm"
MODEL_ROOT="/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001"
CONTROL_IMAGE="${CONTROL_IMAGE:-oscar-glm-stage9-runtime:d0d22489b}"
BASE_IMAGE="glm52-oscar-a800-phase6-d0d22489b-0275043c:latest"
BASE_SOURCE_VOLUME="oscar-glm-phase0-source-fd3e0b3"
EXPECTED_SOURCE_COMMIT="d0d22489b265fc98f9f829dbcfca5e815543d337"
PERFORMANCE_CONFIG="${PROJECT_ROOT}/configs/phase9/performance_matrix.json"
HOST_OUTPUT_ROOT="${HOST_OUTPUT_ROOT:-/dev/shm/oscar-glm-stage9}"

usage() {
  cat <<'EOF'
Usage:
  run_containerized_performance.sh build-image
  RUN_ID=<safe-id> run_containerized_performance.sh \
    preflight-baseline|preflight-candidate
  FORMAL_RUN=1 RUN_ID=<safe-id> \
    run_containerized_performance.sh baseline|candidate
  FORMAL_RUN=1 STAGE9_ONLY_CELL=1024:1 RUN_ID=<safe-id> \
    run_containerized_performance.sh baseline|candidate
  FORMAL_RUN=1 RUN_ID=<safe-id> \
    run_containerized_performance.sh accuracy-smoke-candidate

The baseline and candidate use the same TP=8 source, model, server parameters,
random request matrix, warm-up count, and profiler. The selected variant changes
only the KV cache dtype/path. STAGE9_ONLY_CELL runs one exact frozen matrix cell
and does not run the candidate 128K check.
EOF
}

expected_control_image_id() {
  python3 - "${PERFORMANCE_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["runtime_container"]["image_id"])
PY
}

verify_control_image() {
  local actual expected
  expected="$(expected_control_image_id)"
  actual="$(docker image inspect "${CONTROL_IMAGE}" --format '{{.Id}}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "ERROR: unexpected Stage 9 control image ID: ${actual}" >&2
    echo "Expected: ${expected}" >&2
    exit 1
  }
}

require_safe_run_id() {
  [[ "${RUN_ID}" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]] || {
    echo "ERROR: RUN_ID contains unsafe characters" >&2
    exit 1
  }
}

require_clean_published_repositories() {
  local main_head source_head main_remote source_remote
  [[ -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all)" ]] || {
    echo "ERROR: main repository is not clean" >&2
    exit 1
  }
  [[ -z "$(git -C "${SOURCE_REPO}" status --porcelain --untracked-files=all)" ]] || {
    echo "ERROR: source repository is not clean" >&2
    exit 1
  }
  main_head="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
  source_head="$(git -C "${SOURCE_REPO}" rev-parse HEAD)"
  [[ "${source_head}" == "${EXPECTED_SOURCE_COMMIT}" ]] || {
    echo "ERROR: unexpected source commit: ${source_head}" >&2
    exit 1
  }
  main_remote="$(
    git -C "${PROJECT_ROOT}" ls-remote --heads origin \
      refs/heads/feat/glm52-model-load | awk 'NR == 1 {print $1}'
  )"
  source_remote="$(
    git -C "${SOURCE_REPO}" ls-remote --heads origin \
      refs/heads/feat/glm52-oscar-integration | awk 'NR == 1 {print $1}'
  )"
  [[ "${main_head}" == "${main_remote}" ]] || {
    echo "ERROR: main commit is not published" >&2
    exit 1
  }
  [[ "${source_head}" == "${source_remote}" ]] || {
    echo "ERROR: source commit is not published" >&2
    exit 1
  }
  export PREVERIFIED_MAIN_COMMIT="${main_head}"
  export PREVERIFIED_SOURCE_COMMIT="${source_head}"
}

build_control_image() {
  docker image inspect "${BASE_IMAGE}" >/dev/null
  docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --file "${PROJECT_ROOT}/docker/Dockerfile.phase9-runtime" \
    --tag "${CONTROL_IMAGE}" \
    "${PROJECT_ROOT}/docker"
  docker image inspect "${CONTROL_IMAGE}" \
    --format '{{.Id}} {{index .RepoDigests 0}}' 2>/dev/null ||
    docker image inspect "${CONTROL_IMAGE}" --format '{{.Id}}'
  verify_control_image
}

prepare_runtime_sources() {
  local overlay_source="${PROJECT_ROOT}/artifacts/phase6/20260802T154800Z_candidate_d0d22489b_inverse_fusion_v3/overlay_rootfs/opt/vllm_glm52_v1"
  local base_source="${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs/opt/vllm_glm52_v1"

  git config --global --add safe.directory "${PROJECT_ROOT}"
  git config --global --add safe.directory "${SOURCE_REPO}"

  find /opt/vllm_glm52_v1 -type f -name '*.pyc' -delete
  find /opt/vllm_glm52_v1 -depth -type d -name __pycache__ -empty -delete
  local relative
  for relative in \
    vllm/_C.abi3.so \
    vllm/_C_stable_libtorch.abi3.so \
    vllm/_moe_C.abi3.so \
    vllm/cumem_allocator.abi3.so \
    vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
    vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so; do
    [[ -f "${base_source}/${relative}" ]] || {
      echo "ERROR: base native extension is missing: ${relative}" >&2
      exit 1
    }
    unlink "/opt/vllm_glm52_v1/${relative}"
    ln -s "${base_source}/${relative}" "/opt/vllm_glm52_v1/${relative}"
  done
  mount --bind /opt/vllm_glm52_v1 "${overlay_source}"
}

select_variant() {
  local variant="$1"
  case "${variant}" in
    baseline)
      printf '%s\n' \
        "18083" \
        "${PROJECT_ROOT}/scripts/phase9/run_native_tp8.sh"
      ;;
    candidate)
      printf '%s\n' \
        "18084" \
        "${PROJECT_ROOT}/scripts/phase9/run_candidate_tp8.sh"
      ;;
    *)
      echo "ERROR: unsupported variant: ${variant}" >&2
      exit 2
      ;;
  esac
}

candidate_hf_overrides_json() {
  python3 - "${PERFORMANCE_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    overrides = json.load(handle)["candidate_hf_overrides"]
expected = {"index_topk": 1024}
if overrides != expected:
    raise SystemExit(f"unexpected candidate HF overrides: {overrides!r}")
print(json.dumps(overrides, separators=(",", ":"), sort_keys=True))
PY
}

candidate_prefill_sort_indices() {
  python3 - "${PERFORMANCE_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    environment = json.load(handle)["candidate_runtime_environment"]
expected = {
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND": "legacy",
    "VLLM_SPARSE_INDEXER_PREFILL_TOPK_TOKENS": "768",
    "VLLM_TOPK_PREFILL_SORT_INDICES": "1",
}
if environment != expected:
    raise SystemExit(f"unexpected candidate runtime environment: {environment!r}")
print(environment["VLLM_TOPK_PREFILL_SORT_INDICES"])
PY
}

candidate_decode_topk_backend() {
  python3 - "${PERFORMANCE_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    environment = json.load(handle)["candidate_runtime_environment"]
expected = {
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND": "legacy",
    "VLLM_SPARSE_INDEXER_PREFILL_TOPK_TOKENS": "768",
    "VLLM_TOPK_PREFILL_SORT_INDICES": "1",
}
if environment != expected:
    raise SystemExit(f"unexpected candidate runtime environment: {environment!r}")
print(environment["VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND"])
PY
}

candidate_prefill_topk_tokens() {
  python3 - "${PERFORMANCE_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    environment = json.load(handle)["candidate_runtime_environment"]
expected = {
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND": "legacy",
    "VLLM_SPARSE_INDEXER_PREFILL_TOPK_TOKENS": "768",
    "VLLM_TOPK_PREFILL_SORT_INDICES": "1",
}
if environment != expected:
    raise SystemExit(f"unexpected candidate runtime environment: {environment!r}")
print(environment["VLLM_SPARSE_INDEXER_PREFILL_TOPK_TOKENS"])
PY
}

inside_preflight() {
  local variant="$1"
  local profile_dir="${HOST_OUTPUT_ROOT}/preflight-profiles/${RUN_ID}"
  local selected port wrapper
  prepare_runtime_sources
  mapfile -t selected < <(select_variant "${variant}")
  port="${selected[0]}"
  wrapper="${selected[1]}"

  [[ ! -e "${profile_dir}" ]] || {
    echo "ERROR: preflight profiler directory already exists: ${profile_dir}" >&2
    exit 1
  }
  STAGE9_PROFILE_DIR="${profile_dir}" \
  ARTIFACT_ROOT="${HOST_OUTPUT_ROOT}" \
  CACHE_ROOT="${HOST_OUTPUT_ROOT}/cache/${variant}" \
  RUN_ID="${RUN_ID}" \
  PORT="${port}" \
  "${wrapper}" dry-run
}

inside_container() {
  local variant="$1"
  local server_run_dir="${HOST_OUTPUT_ROOT}/phase9/${RUN_ID}"
  local output_dir="${HOST_OUTPUT_ROOT}/results/${RUN_ID}"
  local profile_dir="${HOST_OUTPUT_ROOT}/profiles/${RUN_ID}"
  local selected port wrapper
  prepare_runtime_sources
  mapfile -t selected < <(select_variant "${variant}")
  port="${selected[0]}"
  wrapper="${selected[1]}"

  [[ ! -e "${server_run_dir}" ]] || {
    echo "ERROR: server run directory already exists: ${server_run_dir}" >&2
    exit 1
  }
  [[ ! -e "${output_dir}" ]] || {
    echo "ERROR: matrix output directory already exists: ${output_dir}" >&2
    exit 1
  }
  [[ ! -e "${profile_dir}" ]] || {
    echo "ERROR: profiler directory already exists: ${profile_dir}" >&2
    exit 1
  }
  mkdir -p "${HOST_OUTPUT_ROOT}"

  local wrapper_pid=""
  cleanup() {
    if [[ -n "${wrapper_pid:-}" ]] && kill -0 "${wrapper_pid}" 2>/dev/null; then
      kill -TERM -- "-${wrapper_pid}" 2>/dev/null || true
      wait "${wrapper_pid}" 2>/dev/null || true
    fi
  }
  trap 'cleanup; exit 130' INT
  trap 'cleanup; exit 143' TERM
  trap cleanup EXIT

  FORMAL_RUN=1 \
  PREVERIFIED_PUBLISHED_COMMITS=1 \
  PREVERIFIED_MAIN_COMMIT="${PREVERIFIED_MAIN_COMMIT}" \
  PREVERIFIED_SOURCE_COMMIT="${PREVERIFIED_SOURCE_COMMIT}" \
  STAGE9_PROFILE_DIR="${profile_dir}" \
  ARTIFACT_ROOT="${HOST_OUTPUT_ROOT}" \
  CACHE_ROOT="${HOST_OUTPUT_ROOT}/cache/${variant}" \
  RUN_ID="${RUN_ID}" \
  PORT="${port}" \
  setsid "${wrapper}" serve &
  wrapper_pid=$!

  local started elapsed next_progress=600
  started="$(date +%s)"
  while ! curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    kill -0 "${wrapper_pid}" 2>/dev/null || {
      wait "${wrapper_pid}" || true
      echo "ERROR: ${variant} server exited before readiness" >&2
      exit 1
    }
    sleep 30
    elapsed="$(( $(date +%s) - started ))"
    if ((elapsed >= next_progress)); then
      printf '%s variant=%s startup_elapsed_seconds=%s\n' \
        "$(date -u +%FT%TZ)" "${variant}" "${elapsed}" |
        tee -a "${HOST_OUTPUT_ROOT}/${RUN_ID}_orchestrator_progress_10min.log"
      next_progress="$((next_progress + 600))"
    fi
    ((elapsed <= 7200)) || {
      echo "ERROR: ${variant} server did not become ready within 7200 seconds" >&2
      exit 1
    }
  done

  local matrix_args=(
    "${PROJECT_ROOT}/scripts/phase9/run_performance_matrix.py"
    --variant "${variant}"
    --server-run-dir "${server_run_dir}"
    --output-dir "${output_dir}"
    --profile-dir "${profile_dir}"
    --base-url "http://127.0.0.1:${port}"
    --runtime-project-root "${PROJECT_ROOT}"
    --formal
  )
  if [[ -n "${STAGE9_ONLY_CELL:-}" ]]; then
    [[ "${STAGE9_ONLY_CELL}" =~ ^([0-9]+):([0-9]+)$ ]] || {
      echo "ERROR: STAGE9_ONLY_CELL must be INPUT_LENGTH:BATCH_SIZE" >&2
      exit 1
    }
    matrix_args+=(--only-cell "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
  elif [[ "${variant}" == "candidate" ]]; then
    matrix_args+=(--include-128k)
  fi
  "${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs/usr/bin/python3.12" \
    "${matrix_args[@]}"

  cleanup
  wrapper_pid=""
  trap - EXIT INT TERM
}

inside_accuracy_smoke() {
  local variant="$1"
  [[ "${variant}" == "candidate" ]] || {
    echo "ERROR: accuracy smoke only supports the candidate" >&2
    exit 2
  }
  prepare_runtime_sources
  HF_OVERRIDES_JSON="$(candidate_hf_overrides_json)" \
  VLLM_TOPK_PREFILL_SORT_INDICES="$(candidate_prefill_sort_indices)" \
  VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="$(candidate_decode_topk_backend)" \
  VLLM_SPARSE_INDEXER_PREFILL_TOPK_TOKENS="$(candidate_prefill_topk_tokens)" \
  FORMAL_RUN=1 \
  EVALUATION_ROLE=candidate \
  EVALUATION_TIER=fast \
  FAST_SAMPLE_COUNT=256 \
  FAST_CONCURRENCY=16 \
  ARTIFACT_ROOT="${HOST_OUTPUT_ROOT}" \
  STAGE7_RUN_ID="${RUN_ID}" \
  "${PROJECT_ROOT}/scripts/phase7/run_official_v5_gsm8k_isolated.sh" run
}

run_in_container() {
  local inside_mode="$1"
  local variant="$2"
  RUN_ID="${RUN_ID:?set RUN_ID}"
  require_safe_run_id
  verify_control_image
  docker volume inspect "${BASE_SOURCE_VOLUME}" >/dev/null
  mkdir -p "${HOST_OUTPUT_ROOT}"
  docker run --rm \
    --name "oscar-glm-stage9-${variant}-${RUN_ID}" \
    --gpus all \
    --ipc host \
    --ulimit memlock=-1 \
    --cap-add SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --mount "type=bind,src=${PROJECT_ROOT},dst=${PROJECT_ROOT}" \
    --mount "type=bind,src=${MODEL_ROOT},dst=${MODEL_ROOT},readonly" \
    --mount "type=volume,src=${BASE_SOURCE_VOLUME},dst=${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs/opt/vllm_glm52_v1" \
    --env FORMAL_RUN=1 \
    --env RUN_ID="${RUN_ID}" \
    --env HOST_OUTPUT_ROOT="${HOST_OUTPUT_ROOT}" \
    --env PREVERIFIED_MAIN_COMMIT="${PREVERIFIED_MAIN_COMMIT:-}" \
    --env PREVERIFIED_SOURCE_COMMIT="${PREVERIFIED_SOURCE_COMMIT:-}" \
    --env STAGE9_ONLY_CELL="${STAGE9_ONLY_CELL:-}" \
    --env PROJECT_ROOT="${PROJECT_ROOT}" \
    --env SOURCE_REPO="${SOURCE_REPO}" \
    --entrypoint /bin/bash \
    "${CONTROL_IMAGE}" \
    "${PROJECT_ROOT}/scripts/phase9/run_containerized_performance.sh" \
    "${inside_mode}" "${variant}"
}

run_preflight() {
  require_clean_published_repositories
  run_in_container inside-preflight "$1"
}

run_variant() {
  local variant="$1"
  [[ "${FORMAL_RUN:-0}" == "1" ]] || {
    echo "ERROR: formal performance run requires FORMAL_RUN=1" >&2
    exit 1
  }
  require_clean_published_repositories
  run_in_container inside "${variant}"
}

run_accuracy_smoke() {
  [[ "${FORMAL_RUN:-0}" == "1" ]] || {
    echo "ERROR: formal accuracy smoke requires FORMAL_RUN=1" >&2
    exit 1
  }
  require_clean_published_repositories
  run_in_container inside-accuracy-smoke candidate
}

mode="${1:-}"
case "${mode}" in
  build-image)
    build_control_image
    ;;
  preflight-baseline)
    run_preflight baseline
    ;;
  preflight-candidate)
    run_preflight candidate
    ;;
  baseline | candidate)
    run_variant "${mode}"
    ;;
  accuracy-smoke-candidate)
    run_accuracy_smoke
    ;;
  inside-preflight)
    inside_preflight "${2:?set variant}"
    ;;
  inside)
    inside_container "${2:?set variant}"
    ;;
  inside-accuracy-smoke)
    inside_accuracy_smoke "${2:?set variant}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
