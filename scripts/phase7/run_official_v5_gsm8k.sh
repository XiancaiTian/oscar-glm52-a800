#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/phase7/official_v5_gsm8k.json"
VERIFY="${SCRIPT_DIR}/verify_official_v5_gsm8k.py"
FROZEN_ROOT="${PROJECT_ROOT}/artifacts/phase7/frozen_evaluator_v5_20260728"
SUITE_DIR="${FROZEN_ROOT}/accuracy_suites/model_agnostic_accuracy_official_v5"
RUNNER="${FROZEN_ROOT}/tools/run_accuracy_suite.py"
EVAL_PYTHON="${FROZEN_ROOT}/.venv/bin/python"
NLTK_DATA="${FROZEN_ROOT}/nltk_data"
RUNTIME_EVAL_CONFIG="${PROJECT_ROOT}/configs/phase7/official_v5_eval_config_high_timeout_7200.json"
STAGE7_RUN_ID="${STAGE7_RUN_ID:?set STAGE7_RUN_ID}"
EVALUATION_ROLE="${EVALUATION_ROLE:?set EVALUATION_ROLE to native or candidate}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PORT="${PORT:?set PORT}"
ATTEMPT_ID="${ACCURACY_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_NAME="glm-5.2-fp8-pruned-reap-e154"
export PYTHONDONTWRITEBYTECODE=1

case "${EVALUATION_ROLE}" in
  native | candidate) ;;
  *)
    echo "ERROR: EVALUATION_ROLE must be native or candidate" >&2
    exit 1
    ;;
esac
[[ "${STAGE7_RUN_ID}" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]] || {
  echo "ERROR: STAGE7_RUN_ID contains unsafe characters" >&2
  exit 1
}
[[ "${ATTEMPT_ID}" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]] || {
  echo "ERROR: ACCURACY_ATTEMPT_ID contains unsafe characters" >&2
  exit 1
}
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

SERVER_RUN_DIR="${ARTIFACT_ROOT}/phase7/${STAGE7_RUN_ID}"
OUTPUT_DIR="${SERVER_RUN_DIR}/official_v5_gsm8k/${ATTEMPT_ID}"
RUNTIME_SUITE_DIR="${OUTPUT_DIR}/runtime_suite"
BASE_URL="http://127.0.0.1:${PORT}/v1"
RUNTIME_MANIFEST="${SERVER_RUN_DIR}/runtime_manifest.json"
SERVER_PID_FILE="${SERVER_RUN_DIR}/server.pid"
PREFLIGHT="${OUTPUT_DIR}/v5_preflight.json"

[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "ERROR: accuracy attempt already exists: ${OUTPUT_DIR}" >&2
  exit 1
}
mkdir -p "${OUTPUT_DIR}"
NLTK_DATA="${NLTK_DATA}" PYTHONPATH="${FROZEN_ROOT}" \
  "${EVAL_PYTHON}" "${VERIFY}" --config "${CONFIG}" --output "${PREFLIGHT}" \
  > "${OUTPUT_DIR}/v5_preflight.log"

[[ -f "${RUNTIME_MANIFEST}" ]] || {
  echo "ERROR: runtime manifest is missing: ${RUNTIME_MANIFEST}" >&2
  exit 1
}
"${EVAL_PYTHON}" - "${RUNTIME_MANIFEST}" "${EVALUATION_ROLE}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
role = sys.argv[2]
expected = {
    "evaluation_protocol": "official_v5",
    "evaluation_scope": "current_stage_gsm8k",
    "evaluation_manifest_sha256": (
        "ffc1d3b38f13a768ce76e2beb43709e5cf643b52b3a973c89fb976fb2207eb2b"
    ),
    "model_filename_size_mtime_ns_manifest_sha256": (
        "83eefdf08de8f489bee6f1d5b1bf4d2f3452d42757b4df580e76a40c3646acaf"
    ),
}
for name, value in expected.items():
    if manifest.get(name) != value:
        raise SystemExit(f"runtime identity mismatch: {name}")
expected_digest = {
    "native": "sha256:2fdfbe865aecc01eee15a01fcce58bf7581244dbbc53cbe3ef0e0cce44bc489d",
    "candidate": "sha256:57c03fca6636b9d0ebd97d5d8aecf94b82e872995c76057938738f3e469e484a",
}[role]
if manifest.get("candidate_manifest_digest") != expected_digest:
    raise SystemExit("runtime candidate manifest digest mismatch")
PY

server_pid="$(<"${SERVER_PID_FILE}")"
kill -0 "${server_pid}" 2>/dev/null || {
  echo "ERROR: server PID ${server_pid} is not running" >&2
  exit 1
}
curl -fsS "${BASE_URL%/v1}/health" >/dev/null

mkdir "${RUNTIME_SUITE_DIR}"
cp "${SUITE_DIR}/manifest.jsonl" "${RUNTIME_SUITE_DIR}/manifest.jsonl"
cp "${RUNTIME_EVAL_CONFIG}" "${RUNTIME_SUITE_DIR}/eval_config.json"
[[ "$(sha256sum "${RUNTIME_SUITE_DIR}/manifest.jsonl" | awk '{print $1}')" == \
  "ffc1d3b38f13a768ce76e2beb43709e5cf643b52b3a973c89fb976fb2207eb2b" ]] || {
  echo "ERROR: runtime suite manifest identity mismatch" >&2
  exit 1
}
[[ "$(sha256sum "${RUNTIME_SUITE_DIR}/eval_config.json" | awk '{print $1}')" == \
  "ef5784be4dfcfa28fb3064bb850de10458469f91f1856c716df1442a6557572a" ]] || {
  echo "ERROR: runtime evaluation config identity mismatch" >&2
  exit 1
}

command=(
  "${EVAL_PYTHON}" "${RUNNER}"
  --suite-dir "${RUNTIME_SUITE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --base-url "${BASE_URL}"
  --model "${MODEL_NAME}"
  --concurrency 8
  --benchmarks GSM8K
  --code-eval-isolated
  --resume
)
printf '%q ' "${command[@]}" > "${OUTPUT_DIR}/runner_command.txt"
printf '\n' >> "${OUTPUT_DIR}/runner_command.txt"
{
  printf 'started_at_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'evaluation_role=%s\n' "${EVALUATION_ROLE}"
  printf 'runner_sha256=%s\n' "$(sha256sum "${RUNNER}" | awk '{print $1}')"
  printf 'suite_manifest_sha256=%s\n' \
    "$(sha256sum "${SUITE_DIR}/manifest.jsonl" | awk '{print $1}')"
  printf 'suite_eval_config_sha256=%s\n' \
    "$(sha256sum "${SUITE_DIR}/eval_config.json" | awk '{print $1}')"
  printf 'runtime_suite_manifest_sha256=%s\n' \
    "$(sha256sum "${RUNTIME_SUITE_DIR}/manifest.jsonl" | awk '{print $1}')"
  printf 'runtime_suite_eval_config_sha256=%s\n' \
    "$(sha256sum "${RUNTIME_SUITE_DIR}/eval_config.json" | awk '{print $1}')"
  printf 'math_reasoning_timeout_seconds=7200\n'
  printf 'reasoning_effort=high\n'
  printf 'requirements_lock_sha256=%s\n' \
    "$(sha256sum "${FROZEN_ROOT}/requirements-lock.txt" | awk '{print $1}')"
  printf 'runtime_manifest_sha256=%s\n' \
    "$(sha256sum "${RUNTIME_MANIFEST}" | awk '{print $1}')"
  printf 'evaluator_python=%s\n' "$("${EVAL_PYTHON}" -VV | tr '\n' ' ')"
  printf 'network_namespace=%s\n' "$(readlink /proc/self/ns/net)"
  printf 'user_namespace=%s\n' "$(readlink /proc/self/ns/user)"
  printf 'code_eval_isolated=true\n'
} > "${OUTPUT_DIR}/runner_environment.txt"

server_success_total() {
  curl -fsS "${BASE_URL%/v1}/metrics" |
    awk '$1 ~ /^vllm:request_success_total\{/ { total += $2 } END { printf "%.0f\n", total }'
}

baseline_server_success="$(server_success_total)"
runner_pid=""
cleanup_runner() {
  if [[ -n "${runner_pid}" ]] && kill -0 "${runner_pid}" 2>/dev/null; then
    kill -TERM "${runner_pid}" 2>/dev/null || true
    wait "${runner_pid}" 2>/dev/null || true
  fi
}
trap 'cleanup_runner; exit 130' INT
trap 'cleanup_runner; exit 143' TERM
trap cleanup_runner EXIT

NLTK_DATA="${NLTK_DATA}" PYTHONPATH="${FROZEN_ROOT}" \
  "${command[@]}" > "${OUTPUT_DIR}/runner.log" 2>&1 &
runner_pid=$!
started="$(date +%s)"
next_progress=600
while kill -0 "${runner_pid}" 2>/dev/null; do
  sleep 60
  elapsed="$(( $(date +%s) - started ))"
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${runner_pid}" 2>/dev/null || true
    wait "${runner_pid}" 2>/dev/null || true
    echo "ERROR: model server exited during official_v5 GSM8K" >&2
    exit 1
  fi
  if ((elapsed >= next_progress)); then
    runner_completed="$(
      grep -Eo 'completed [0-9]+/1319' "${OUTPUT_DIR}/runner.log" |
        tail -1 || true
    )"
    current_server_success="$(server_success_total)"
    server_completed="$((current_server_success - baseline_server_success))"
    {
      printf '%s elapsed_seconds=%s %s server_completed=%s/1319\n' \
        "$(date -u +%FT%TZ)" "${elapsed}" \
        "${runner_completed:-completed unknown/1319}" "${server_completed}"
      nvidia-smi \
        --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
    } | tee -a "${OUTPUT_DIR}/progress_10min.log"
    next_progress="$((next_progress + 600))"
  fi
done

set +e
wait "${runner_pid}"
runner_status=$?
set -e
runner_pid=""
[[ "${runner_status}" -eq 0 ]] || {
  echo "ERROR: official_v5 GSM8K runner exited with status ${runner_status}" >&2
  exit "${runner_status}"
}

"${EVAL_PYTHON}" - "${OUTPUT_DIR}" "${EVALUATION_ROLE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
role = sys.argv[2]
runtime_manifest = output.parents[1] / "runtime_manifest.json"
summary = json.loads((output / "summary.json").read_text())
by_benchmark = json.loads((output / "summary_by_benchmark.json").read_text())
predictions = output / "predictions.jsonl"
rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
bad_statuses = {
    key: value
    for key, value in summary["status_counts"].items()
    if key != "scored" and value
}
gsm = summary.get("native_metrics", {}).get("GSM8K", {})
if (
    summary.get("protocol_version") != "official_v5"
    or summary.get("valid") is not True
    or summary.get("total") != 1319
    or summary.get("scored") != 1319
    or len(rows) != 1319
    or gsm.get("total") != 1319
    or len(by_benchmark) != 1
    or by_benchmark[0].get("benchmark") != "GSM8K"
):
    raise SystemExit("incomplete official_v5 GSM8K result")
if bad_statuses:
    raise SystemExit(f"non-scored statuses: {bad_statuses}")
fingerprints = {row.get("protocol_fingerprint") for row in rows}
if fingerprints != {summary["protocol_fingerprint"]}:
    raise SystemExit("prediction protocol fingerprints are inconsistent")
result = {
    "format_version": 1,
    "status": "passed",
    "evaluation_role": role,
    "protocol_version": "official_v5",
    "scope": "current_stage_gsm8k",
    "total": 1319,
    "scored": 1319,
    "request_failures": 0,
    "reasoning_effort": "high",
    "accuracy": gsm["accuracy"],
    "truncated_count": summary["truncated_count"],
    "truncation_rate": summary["truncation_rate"],
    "protocol_fingerprint": summary["protocol_fingerprint"],
    "summary_sha256": hashlib.sha256((output / "summary.json").read_bytes()).hexdigest(),
    "summary_by_benchmark_sha256": hashlib.sha256(
        (output / "summary_by_benchmark.json").read_bytes()
    ).hexdigest(),
    "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
    "runner_command_sha256": hashlib.sha256(
        (output / "runner_command.txt").read_bytes()
    ).hexdigest(),
    "runner_environment_sha256": hashlib.sha256(
        (output / "runner_environment.txt").read_bytes()
    ).hexdigest(),
    "runtime_manifest_sha256": hashlib.sha256(
        runtime_manifest.read_bytes()
    ).hexdigest(),
    "runtime_suite_manifest_sha256": hashlib.sha256(
        (output / "runtime_suite" / "manifest.jsonl").read_bytes()
    ).hexdigest(),
    "runtime_suite_eval_config_sha256": hashlib.sha256(
        (output / "runtime_suite" / "eval_config.json").read_bytes()
    ).hexdigest(),
}
(output / "validation.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
PY
