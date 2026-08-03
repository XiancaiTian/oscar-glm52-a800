#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_REPO="${PROJECT_ROOT}/glm52_oscar_vllm"
CONFIG="${PROJECT_ROOT}/configs/phase7/official_v5_gsm8k.json"
VERIFY="${SCRIPT_DIR}/verify_official_v5_gsm8k.py"
FAST_CONFIG="${PROJECT_ROOT}/configs/phase7/official_v5_fast_gsm8k.json"
VERIFY_FAST="${SCRIPT_DIR}/verify_official_v5_fast.py"
FROZEN_ROOT="${PROJECT_ROOT}/artifacts/phase7/frozen_evaluator_v5_20260728"
EVAL_PYTHON="${FROZEN_ROOT}/.venv/bin/python"
SUITE_DIR="${FROZEN_ROOT}/accuracy_suites/model_agnostic_accuracy_official_v5"
STATIC_SUITE_DIR="${PROJECT_ROOT}/artifacts/phase7/frozen_evaluator_v4_20260728/accuracy_v4_fc374ff4_4aec8ee8/suite"
EXPECTED_SOURCE_COMMIT="83320e1205b65b551633eb4e32c4858987ba0516"
EXPECTED_MANIFEST_SHA256="ffc1d3b38f13a768ce76e2beb43709e5cf643b52b3a973c89fb976fb2207eb2b"
EVALUATION_TIER="${EVALUATION_TIER:-formal}"
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat <<'EOF'
Usage:
  run_official_v5_gsm8k_isolated.sh preflight
  FORMAL_RUN=1 EVALUATION_ROLE=native|candidate \
    run_official_v5_gsm8k_isolated.sh run

The run mode verifies published commits before creating a user+network
namespace. The model service and official_v5 runner then share only that
namespace's loopback interface and have no external network route.

Set EVALUATION_TIER=fast with FAST_SAMPLE_COUNT=256|1319 and
FAST_CONCURRENCY=8|16 for the Stage 7 8K/high screening protocol.
EOF
}

validate_artifact_root() {
  [[ "${ARTIFACT_ROOT}" == /* ]] || {
    echo "ERROR: ARTIFACT_ROOT must be absolute" >&2
    return 1
  }
  ARTIFACT_ROOT="$(realpath -m -- "${ARTIFACT_ROOT}")"
  [[ "${ARTIFACT_ROOT}" == "${PROJECT_ROOT}/artifacts" ||
    "${ARTIFACT_ROOT}" == "${PROJECT_ROOT}/artifacts/"* ||
    "${ARTIFACT_ROOT}" == /dev/shm ||
    "${ARTIFACT_ROOT}" == /dev/shm/* ]] || {
    echo "ERROR: ARTIFACT_ROOT must be under project artifacts/ or /dev/shm/" >&2
    return 1
  }
}

published_commit() {
  local repo="$1"
  local branch="$2"
  local label="$3"
  local head remote_url remote_head
  [[ -z "$(git -C "${repo}" status --porcelain --untracked-files=all)" ]] || {
    echo "ERROR: ${label} repository is not clean: ${repo}" >&2
    return 1
  }
  [[ "$(git -C "${repo}" symbolic-ref --quiet --short HEAD)" == "${branch}" ]] || {
    echo "ERROR: ${label} is not on ${branch}" >&2
    return 1
  }
  head="$(git -C "${repo}" rev-parse HEAD)"
  remote_url="$(git -C "${repo}" remote get-url origin)"
  remote_head="$(
    git ls-remote --heads "${remote_url}" "refs/heads/${branch}" |
      awk 'NR == 1 {print $1}'
  )"
  [[ "${head}" == "${remote_head}" ]] || {
    echo "ERROR: ${label} HEAD ${head} is not published" >&2
    return 1
  }
  printf '%s\n' "${head}"
}

static_preflight() {
  NLTK_DATA="${FROZEN_ROOT}/nltk_data" PYTHONPATH="${FROZEN_ROOT}" \
    "${EVAL_PYTHON}" "${VERIFY}" --config "${CONFIG}" >/dev/null
  case "${EVALUATION_TIER}" in
    formal) ;;
    fast)
      "${EVAL_PYTHON}" "${VERIFY_FAST}" --config "${FAST_CONFIG}" >/dev/null
      ;;
    *)
      echo "ERROR: EVALUATION_TIER must be formal or fast" >&2
      return 1
      ;;
  esac
  unshare -Urn --map-root-user bash -c '
    set -euo pipefail
    ip link set lo up
    ip -o -4 addr show dev lo | grep -q " 127.0.0.1/8 "
    ! ip route show | grep -q "^default "
    [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" == "8" ]]
  '
  echo "official_v5 GSM8K static and namespace preflight passed"
}

inside_namespace() {
  ip link set lo up
  local run_dir="${ARTIFACT_ROOT}/phase7/${STAGE7_RUN_ID}"
  mkdir -p "${run_dir}"
  {
    printf 'started_at_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'network_namespace=%s\n' "$(readlink /proc/self/ns/net)"
    printf 'user_namespace=%s\n' "$(readlink /proc/self/ns/user)"
    printf 'loopback_ipv4=%s\n' "$(ip -o -4 addr show dev lo | awk '{print $4}')"
    printf 'routes=%s\n' "$(ip route show | tr '\n' ';')"
    printf 'main_commit=%s\n' "${PREVERIFIED_MAIN_COMMIT}"
    printf 'source_commit=%s\n' "${PREVERIFIED_SOURCE_COMMIT}"
  } > "${run_dir}/network_isolation.txt"
  if ip route show | grep -q '^default '; then
    echo "ERROR: isolated namespace unexpectedly has a default route" >&2
    return 1
  fi
  set +e
  curl --noproxy '*' --connect-timeout 2 --max-time 3 \
    http://1.1.1.1/ >/dev/null 2>&1
  local egress_status=$?
  set -e
  printf 'external_ipv4_probe_exit=%s\n' "${egress_status}" \
    >> "${run_dir}/network_isolation.txt"
  [[ "${egress_status}" -ne 0 ]] || {
    echo "ERROR: isolated namespace unexpectedly reached external IPv4" >&2
    return 1
  }

  local server_wrapper port accuracy_wrapper evaluation_protocol
  local evaluation_scope max_model_len
  case "${EVALUATION_ROLE}" in
    native)
      server_wrapper="${PROJECT_ROOT}/scripts/phase1/run_native_baseline.sh"
      port=18080
      ;;
    candidate)
      server_wrapper="${PROJECT_ROOT}/scripts/phase7/run_candidate_tp8.sh"
      port=18082
      ;;
    *)
      echo "ERROR: EVALUATION_ROLE must be native or candidate" >&2
      return 1
      ;;
  esac
  case "${EVALUATION_TIER}" in
    formal)
      accuracy_wrapper="${PROJECT_ROOT}/scripts/phase7/run_official_v5_gsm8k.sh"
      evaluation_protocol="official_v5"
      evaluation_scope="current_stage_gsm8k"
      max_model_len=32768
      ;;
    fast)
      accuracy_wrapper="${PROJECT_ROOT}/scripts/phase7/run_official_v5_gsm8k_fast.sh"
      evaluation_protocol="official_v5_fast_screen"
      evaluation_scope="stage7_fast_gsm8k_${FAST_SAMPLE_COUNT}"
      max_model_len=8192
      ;;
    *)
      echo "ERROR: EVALUATION_TIER must be formal or fast" >&2
      return 1
      ;;
  esac

  wrapper_pid=""
  cleanup_server() {
    if [[ -n "${wrapper_pid}" ]] && kill -0 "${wrapper_pid}" 2>/dev/null; then
      kill -TERM -- "-${wrapper_pid}" 2>/dev/null || true
      wait "${wrapper_pid}" 2>/dev/null || true
    fi
  }
  trap 'cleanup_server; exit 130' INT
  trap 'cleanup_server; exit 143' TERM
  trap cleanup_server EXIT

  FORMAL_RUN=1 \
  PREVERIFIED_PUBLISHED_COMMITS=1 \
  PREVERIFIED_MAIN_COMMIT="${PREVERIFIED_MAIN_COMMIT}" \
  PREVERIFIED_SOURCE_COMMIT="${PREVERIFIED_SOURCE_COMMIT}" \
  EVALUATION_PROTOCOL="${evaluation_protocol}" \
  EVALUATION_SCOPE="${evaluation_scope}" \
  EVALUATION_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256}" \
  MAX_MODEL_LEN="${max_model_len}" \
  SUITE_DIR="${SUITE_DIR}" \
  STATIC_SUITE_DIR="${STATIC_SUITE_DIR}" \
  ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  ARTIFACT_PHASE=phase7 \
  RUN_ID="${STAGE7_RUN_ID}" \
  PORT="${port}" \
  setsid "${server_wrapper}" serve &
  wrapper_pid=$!
  printf '%s\n' "${wrapper_pid}" > "${run_dir}/server_wrapper.pid"

  local started elapsed next_progress=600
  started="$(date +%s)"
  while ! curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    kill -0 "${wrapper_pid}" 2>/dev/null || {
      wait "${wrapper_pid}" || true
      echo "ERROR: service wrapper exited before readiness" >&2
      return 1
    }
    sleep 30
    elapsed="$(( $(date +%s) - started ))"
    ((elapsed <= 7200)) || {
      echo "ERROR: service did not become ready within 7200 seconds" >&2
      return 1
    }
    if ((elapsed >= next_progress)); then
      printf '%s service_startup_elapsed_seconds=%s role=%s\n' \
        "$(date -u +%FT%TZ)" "${elapsed}" "${EVALUATION_ROLE}" |
        tee -a "${run_dir}/orchestrator_progress_10min.log"
      next_progress="$((next_progress + 600))"
    fi
  done

  EVALUATION_ROLE="${EVALUATION_ROLE}" \
  STAGE7_RUN_ID="${STAGE7_RUN_ID}" \
  ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  PORT="${port}" \
  FAST_SAMPLE_COUNT="${FAST_SAMPLE_COUNT:-}" \
  FAST_CONCURRENCY="${FAST_CONCURRENCY:-}" \
  "${accuracy_wrapper}"
  date -u +%FT%TZ > "${run_dir}/accuracy_completed_at_utc.txt"
  cleanup_server
  wrapper_pid=""
}

wait_for_gpu_release() {
  local attempt rows
  for attempt in $(seq 1 30); do
    rows="$(
      nvidia-smi --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )"
    if [[ "$(awk -F, '$1 + 0 != 0 || $2 + 0 != 0 {bad++} END {print bad+0}' <<<"${rows}")" == "0" ]]; then
      echo "GPU release check passed: 8/8 idle"
      return
    fi
    sleep 10
  done
  echo "ERROR: GPUs were not released within 300 seconds" >&2
  return 1
}

mode="${1:-}"
case "${mode}" in
  preflight)
    static_preflight
    ;;
  run)
    [[ "${FORMAL_RUN:-0}" == "1" ]] || {
      echo "ERROR: run requires FORMAL_RUN=1" >&2
      exit 1
    }
    EVALUATION_ROLE="${EVALUATION_ROLE:?set EVALUATION_ROLE}"
    case "${EVALUATION_ROLE}" in
      native | candidate) ;;
      *)
        echo "ERROR: EVALUATION_ROLE must be native or candidate" >&2
        exit 1
        ;;
    esac
    case "${EVALUATION_TIER}" in
      formal) ;;
      fast)
        case "${FAST_SAMPLE_COUNT:-}" in
          256 | 1319) ;;
          *)
            echo "ERROR: fast tier requires FAST_SAMPLE_COUNT=256 or 1319" >&2
            exit 1
            ;;
        esac
        case "${FAST_CONCURRENCY:-}" in
          8 | 16) ;;
          *)
            echo "ERROR: fast tier requires FAST_CONCURRENCY=8 or 16" >&2
            exit 1
            ;;
        esac
        ;;
      *)
        echo "ERROR: EVALUATION_TIER must be formal or fast" >&2
        exit 1
        ;;
    esac
    ARTIFACT_ROOT="${ARTIFACT_ROOT:-/dev/shm/oscar-glm-official-v5}"
    validate_artifact_root
    STAGE7_RUN_ID="${STAGE7_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${EVALUATION_ROLE}_${EVALUATION_TIER}_official_v5_gsm8k}"
    [[ "${STAGE7_RUN_ID}" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]] || {
      echo "ERROR: STAGE7_RUN_ID contains unsafe characters" >&2
      exit 1
    }
    [[ ! -e "${ARTIFACT_ROOT}/phase7/${STAGE7_RUN_ID}" ]] || {
      echo "ERROR: run directory already exists" >&2
      exit 1
    }
    static_preflight
    PREVERIFIED_MAIN_COMMIT="$(
      published_commit "${PROJECT_ROOT}" feat/glm52-model-load main
    )"
    PREVERIFIED_SOURCE_COMMIT="$(
      published_commit \
        "${SOURCE_REPO}" feat/glm52-oscar-integration source
    )"
    [[ "${PREVERIFIED_SOURCE_COMMIT}" == "${EXPECTED_SOURCE_COMMIT}" ]] || {
      echo "ERROR: source HEAD is not the frozen candidate commit" >&2
      exit 1
    }
    export PROJECT_ROOT SOURCE_REPO CONFIG VERIFY FAST_CONFIG VERIFY_FAST
    export FROZEN_ROOT EVAL_PYTHON
    export SUITE_DIR STATIC_SUITE_DIR EXPECTED_MANIFEST_SHA256
    export EVALUATION_ROLE EVALUATION_TIER ARTIFACT_ROOT
    export FAST_SAMPLE_COUNT FAST_CONCURRENCY
    export STAGE7_RUN_ID PREVERIFIED_MAIN_COMMIT PREVERIFIED_SOURCE_COMMIT
    unshare -Urn --map-root-user "$(realpath "${BASH_SOURCE[0]}")" inside
    wait_for_gpu_release
    ;;
  inside)
    inside_namespace
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
