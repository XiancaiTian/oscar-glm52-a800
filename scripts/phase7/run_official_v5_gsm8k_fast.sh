#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FROZEN_ROOT="${PROJECT_ROOT}/artifacts/phase7/frozen_evaluator_v5_20260728"
SOURCE_SUITE="${FROZEN_ROOT}/accuracy_suites/model_agnostic_accuracy_official_v5"
FROZEN_RUNNER="${FROZEN_ROOT}/tools/run_accuracy_suite.py"
EVAL_PYTHON="${FROZEN_ROOT}/.venv/bin/python"
NLTK_DATA="${FROZEN_ROOT}/nltk_data"
FAST_RUNNER="${SCRIPT_DIR}/run_accuracy_suite_fast.py"
PREPARE_SUITE="${SCRIPT_DIR}/prepare_official_v5_fast_suite.py"
FAST_EVAL_CONFIG="${PROJECT_ROOT}/configs/phase7/official_v5_eval_config_fast_high_timeout_3600.json"
STAGE7_RUN_ID="${STAGE7_RUN_ID:?set STAGE7_RUN_ID}"
EVALUATION_ROLE="${EVALUATION_ROLE:?set EVALUATION_ROLE to native or candidate}"
FAST_SAMPLE_COUNT="${FAST_SAMPLE_COUNT:?set FAST_SAMPLE_COUNT to 256 or 1319}"
FAST_CONCURRENCY="${FAST_CONCURRENCY:?set FAST_CONCURRENCY to 8 or 16}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PORT="${PORT:?set PORT}"
ATTEMPT_ID="${ACCURACY_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_NAME="glm-5.2-fp8-pruned-reap-e154"
SELECTION_SEED="oscar-glm-stage7-fast-v1"
SOURCE_MANIFEST_SHA256="ffc1d3b38f13a768ce76e2beb43709e5cf643b52b3a973c89fb976fb2207eb2b"
export PYTHONDONTWRITEBYTECODE=1

case "${EVALUATION_ROLE}" in
  native | candidate) ;;
  *)
    echo "ERROR: EVALUATION_ROLE must be native or candidate" >&2
    exit 1
    ;;
esac
case "${FAST_SAMPLE_COUNT}" in
  256 | 1319) ;;
  *)
    echo "ERROR: FAST_SAMPLE_COUNT must be 256 or 1319" >&2
    exit 1
    ;;
esac
case "${FAST_CONCURRENCY}" in
  8 | 16) ;;
  *)
    echo "ERROR: FAST_CONCURRENCY must be 8 or 16" >&2
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
OUTPUT_DIR="$(
  printf '%s/official_v5_fast_gsm8k_%s_c%s/%s' \
    "${SERVER_RUN_DIR}" "${FAST_SAMPLE_COUNT}" "${FAST_CONCURRENCY}" "${ATTEMPT_ID}"
)"
RUNTIME_SUITE_DIR="${OUTPUT_DIR}/runtime_suite"
BASE_URL="http://127.0.0.1:${PORT}/v1"
RUNTIME_MANIFEST="${SERVER_RUN_DIR}/runtime_manifest.json"
SERVER_PID_FILE="${SERVER_RUN_DIR}/server.pid"

[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "ERROR: accuracy attempt already exists: ${OUTPUT_DIR}" >&2
  exit 1
}
mkdir -p "${OUTPUT_DIR}"

[[ "$(sha256sum "${SOURCE_SUITE}/manifest.jsonl" | awk '{print $1}')" == \
  "${SOURCE_MANIFEST_SHA256}" ]] || {
  echo "ERROR: source suite manifest identity mismatch" >&2
  exit 1
}
[[ -f "${RUNTIME_MANIFEST}" ]] || {
  echo "ERROR: runtime manifest is missing: ${RUNTIME_MANIFEST}" >&2
  exit 1
}
"${EVAL_PYTHON}" - "${RUNTIME_MANIFEST}" "${EVALUATION_ROLE}" \
  "${FAST_SAMPLE_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
role = sys.argv[2]
count = int(sys.argv[3])
expected = {
    "evaluation_protocol": "official_v5_fast_screen",
    "evaluation_scope": f"stage7_fast_gsm8k_{count}",
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
    "candidate": "sha256:fb8e914ca146adddaa1eca23134cc55f72243a1abe3eb79a64faa38b71c44a23",
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
"${EVAL_PYTHON}" - "${SERVER_RUN_DIR}/parsed_server_args.json" <<'PY'
import json
import sys
from pathlib import Path

args = json.loads(Path(sys.argv[1]).read_text())
if args.get("max_model_len") != 8192:
    raise SystemExit("fast screening server max_model_len must be 8192")
if args.get("max_num_seqs") != 16:
    raise SystemExit("fast screening server max_num_seqs must be 16")
if args.get("speculative_config") is not None:
    raise SystemExit("speculative decoding must remain disabled")
PY

"${EVAL_PYTHON}" "${PREPARE_SUITE}" \
  --source-manifest "${SOURCE_SUITE}/manifest.jsonl" \
  --eval-config "${FAST_EVAL_CONFIG}" \
  --output-dir "${RUNTIME_SUITE_DIR}" \
  --sample-count "${FAST_SAMPLE_COUNT}" \
  --selection-seed "${SELECTION_SEED}" \
  > "${OUTPUT_DIR}/prepare_suite.log"

command=(
  "${EVAL_PYTHON}" "${FAST_RUNNER}"
  --frozen-runner "${FROZEN_RUNNER}"
  --suite-dir "${RUNTIME_SUITE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --base-url "${BASE_URL}"
  --model "${MODEL_NAME}"
  --concurrency "${FAST_CONCURRENCY}"
  --checkpoint-every 20
  --code-eval-isolated
  --resume
)
printf '%q ' "${command[@]}" > "${OUTPUT_DIR}/runner_command.txt"
printf '\n' >> "${OUTPUT_DIR}/runner_command.txt"
{
  printf 'started_at_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'evaluation_role=%s\n' "${EVALUATION_ROLE}"
  printf 'screening_protocol=official_v5_fast_screen\n'
  printf 'sample_count=%s\n' "${FAST_SAMPLE_COUNT}"
  printf 'concurrency=%s\n' "${FAST_CONCURRENCY}"
  printf 'max_model_len=8192\n'
  printf 'reasoning_effort=high\n'
  printf 'math_reasoning_timeout_seconds=3600\n'
  printf 'frozen_runner_sha256=%s\n' \
    "$(sha256sum "${FROZEN_RUNNER}" | awk '{print $1}')"
  printf 'fast_runner_sha256=%s\n' \
    "$(sha256sum "${FAST_RUNNER}" | awk '{print $1}')"
  printf 'runtime_suite_manifest_sha256=%s\n' \
    "$(sha256sum "${RUNTIME_SUITE_DIR}/manifest.jsonl" | awk '{print $1}')"
  printf 'runtime_suite_eval_config_sha256=%s\n' \
    "$(sha256sum "${RUNTIME_SUITE_DIR}/eval_config.json" | awk '{print $1}')"
  printf 'selection_identity_sha256=%s\n' \
    "$(sha256sum "${RUNTIME_SUITE_DIR}/fast_suite_identity.json" | awk '{print $1}')"
  printf 'runtime_manifest_sha256=%s\n' \
    "$(sha256sum "${RUNTIME_MANIFEST}" | awk '{print $1}')"
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
    echo "ERROR: model server exited during fast GSM8K screening" >&2
    exit 1
  fi
  if ((elapsed >= next_progress)); then
    runner_completed="$(
      grep -Eo "completed [0-9]+/${FAST_SAMPLE_COUNT}" "${OUTPUT_DIR}/runner.log" |
        tail -1 || true
    )"
    current_server_success="$(server_success_total)"
    server_completed="$((current_server_success - baseline_server_success))"
    {
      printf '%s elapsed_seconds=%s %s server_completed=%s/%s\n' \
        "$(date -u +%FT%TZ)" "${elapsed}" \
        "${runner_completed:-completed unknown/${FAST_SAMPLE_COUNT}}" \
        "${server_completed}" "${FAST_SAMPLE_COUNT}"
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
  echo "ERROR: fast GSM8K runner exited with status ${runner_status}" >&2
  exit "${runner_status}"
}

"${EVAL_PYTHON}" - "${OUTPUT_DIR}" "${EVALUATION_ROLE}" \
  "${FAST_SAMPLE_COUNT}" "${FAST_CONCURRENCY}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
role = sys.argv[2]
expected_total = int(sys.argv[3])
concurrency = int(sys.argv[4])
summary = json.loads((output / "summary.json").read_text())
predictions = output / "predictions.jsonl"
rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
gsm = summary.get("native_metrics", {}).get("GSM8K", {})
if (
    summary.get("protocol_version") != "official_v5"
    or summary.get("screening_protocol") != "official_v5_fast_screen"
    or summary.get("valid") is not True
    or summary.get("total") != expected_total
    or summary.get("scored") != expected_total
    or len(rows) != expected_total
    or gsm.get("total") != expected_total
    or summary.get("status_counts") != {"scored": expected_total}
):
    raise SystemExit("incomplete official_v5 fast GSM8K result")
budgets = summary["benchmark_token_budgets"]["GSM8K"]
if (
    budgets["server_max_model_len"] != 8192
    or budgets["max_prompt_tokens"] != 218
    or budgets["fixed_output_limit"] != 7974
    or budgets["observed_max_prompt_tokens"] > 218
):
    raise SystemExit("fast result did not use the frozen 8K GSM8K token budget")
result = {
    "format_version": 1,
    "status": "passed",
    "evaluation_role": role,
    "protocol_version": "official_v5",
    "screening_protocol": "official_v5_fast_screen",
    "scope": f"stage7_fast_gsm8k_{expected_total}",
    "total": expected_total,
    "scored": expected_total,
    "request_failures": 0,
    "concurrency": concurrency,
    "server_max_model_len": 8192,
    "fixed_output_limit": budgets["fixed_output_limit"],
    "reasoning_effort": "high",
    "accuracy": gsm["accuracy"],
    "truncated_count": summary["truncated_count"],
    "truncation_rate": summary["truncation_rate"],
    "completion_tokens_mean": summary["completion_tokens_mean"],
    "requests_per_hour": summary["requests_per_hour"],
    "protocol_fingerprint": summary["protocol_fingerprint"],
    "final_full_evaluation_still_required": True,
}
evidence = {
    "summary_sha256": output / "summary.json",
    "summary_by_benchmark_sha256": output / "summary_by_benchmark.json",
    "summary_by_task_type_sha256": output / "summary_by_task_type.json",
    "predictions_sha256": output / "predictions.jsonl",
    "failed_cases_sha256": output / "failed_cases.jsonl",
    "fast_runner_state_sha256": output / "fast_runner_state.json",
    "runner_command_sha256": output / "runner_command.txt",
    "runner_environment_sha256": output / "runner_environment.txt",
    "runtime_manifest_sha256": output.parents[1] / "runtime_manifest.json",
    "runtime_suite_manifest_sha256": output / "runtime_suite" / "manifest.jsonl",
    "runtime_suite_eval_config_sha256": (
        output / "runtime_suite" / "eval_config.json"
    ),
    "fast_suite_identity_sha256": (
        output / "runtime_suite" / "fast_suite_identity.json"
    ),
}
for name, path in evidence.items():
    result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(output / "validation.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
PY
