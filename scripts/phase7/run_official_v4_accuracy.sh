#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAGE7_RUN_ID="${STAGE7_RUN_ID:?set STAGE7_RUN_ID to the running candidate server ID}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
SERVER_RUN_DIR="${ARTIFACT_ROOT}/phase7/${STAGE7_RUN_ID}"
ATTEMPT_ID="${ACCURACY_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${SERVER_RUN_DIR}/official_v4_accuracy/${ATTEMPT_ID}"
RUNTIME_SUITE_DIR="${OUTPUT_DIR}/runtime_suite"
EVAL_ROOT="/nfs/AE/txc/vllm_turbo_baseline_acc"
SUITE_DIR="${EVAL_ROOT}/accuracy_suites/model_agnostic_accuracy_official_v4"
RUNNER="${EVAL_ROOT}/tools/run_accuracy_suite.py"
EVAL_PYTHON="${PROJECT_ROOT}/artifacts/phase1-eval-venv/bin/python"
LOCK_FILE="${PROJECT_ROOT}/configs/phase1/evaluator-requirements.lock.txt"
BASE_URL="${BASE_URL:-http://127.0.0.1:18082/v1}"
MODEL_NAME="glm-5.2-fp8-pruned-reap-e154"

expected_runner_sha="fc374ff4c4715e37d515d37aa794b3e649dc1710034d3355c69c21efa1a8aeff"
actual_runner_sha="$(sha256sum "${RUNNER}" | awk '{print $1}')"
[[ "${actual_runner_sha}" == "${expected_runner_sha}" ]] || {
  echo "ERROR: official_v4 runner changed: ${actual_runner_sha}" >&2
  exit 1
}
requirement_sha="$(sha256sum "${EVAL_ROOT}/requirements-accuracy-suite.txt" | awk '{print $1}')"
[[ "${requirement_sha}" == "1667a30b31747c80f286ee7a33dc7131c88aa0ff09ae89d1dd2cb6ec35b779c6" ]] || {
  echo "ERROR: evaluator requirements changed: ${requirement_sha}" >&2
  exit 1
}
[[ -x "${EVAL_PYTHON}" ]] || {
  echo "ERROR: evaluator venv is missing: ${EVAL_PYTHON}" >&2
  exit 1
}
diff -u "${LOCK_FILE}" <(uv pip freeze --python "${EVAL_PYTHON}")

runtime_manifest="${SERVER_RUN_DIR}/runtime_manifest.json"
[[ -f "${runtime_manifest}" ]] || {
  echo "ERROR: candidate runtime manifest is missing" >&2
  exit 1
}
"${EVAL_PYTHON}" - "${runtime_manifest}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "runtime_source_commit": "a3317695428819d41437b1cb144404b3bfc05a92",
    "candidate_manifest_digest": (
        "sha256:1d3d26262fd6abe51ee271d99584091fcef3ca2cd35a585204ac14c6340f0ea6"
    ),
    "candidate_config_digest": (
        "sha256:dd7b4f47a900dfc59f599cd99ca9c3e25456d9fd5de29753d1f85fd4f256ca70"
    ),
    "candidate_layer_digest": (
        "sha256:189f55db6bd54114fc1a86f956704e7e4eb13b80f4d1f826add697aad40182d0"
    ),
}
for name, value in expected.items():
    if manifest.get(name) != value:
        raise SystemExit(f"candidate runtime identity mismatch: {name}")
PY
server_pid="$(<"${SERVER_RUN_DIR}/server.pid")"
kill -0 "${server_pid}" 2>/dev/null || {
  echo "ERROR: server PID ${server_pid} is not running" >&2
  exit 1
}
curl -fsS "${BASE_URL%/v1}/health" >/dev/null

[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "ERROR: accuracy attempt already exists: ${OUTPUT_DIR}" >&2
  exit 1
}
mkdir -p "${RUNTIME_SUITE_DIR}"
ln -s "${SUITE_DIR}/manifest.jsonl" "${RUNTIME_SUITE_DIR}/manifest.jsonl"
"${EVAL_PYTHON}" - "${SUITE_DIR}/eval_config.json" \
  "${RUNTIME_SUITE_DIR}/eval_config.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
config = json.loads(source.read_text(encoding="utf-8"))
expected = {
    "code": 600,
    "instruction_following": 300,
    "math_reasoning": 300,
}
for name, value in expected.items():
    if config["timeouts_seconds"][name] != value:
        raise SystemExit(f"unexpected source timeout {name}")
config["timeouts_seconds"]["code"] = 3600
config["timeouts_seconds"]["instruction_following"] = 1800
config["timeouts_seconds"]["math_reasoning"] = 1800
output.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
command=(
  "${EVAL_PYTHON}" "${RUNNER}"
  --suite-dir "${RUNTIME_SUITE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --base-url "${BASE_URL}"
  --model "${MODEL_NAME}"
  --concurrency 8
  --benchmarks GSM8K "LiveCodeBench v6" "MultiPL-E" IFEval
)
printf '%q ' "${command[@]}" > "${OUTPUT_DIR}/runner_command.txt"
printf '\n' >> "${OUTPUT_DIR}/runner_command.txt"
{
  printf 'started_at_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'runner_sha256=%s\n' "${actual_runner_sha}"
  printf 'suite_manifest_sha256=%s\n' \
    "$(sha256sum "${SUITE_DIR}/manifest.jsonl" | awk '{print $1}')"
  printf 'source_eval_config_sha256=%s\n' \
    "$(sha256sum "${SUITE_DIR}/eval_config.json" | awk '{print $1}')"
  printf 'runtime_eval_config_sha256=%s\n' \
    "$(sha256sum "${RUNTIME_SUITE_DIR}/eval_config.json" | awk '{print $1}')"
  printf 'runtime_code_timeout_seconds=3600\n'
  printf 'runtime_instruction_following_timeout_seconds=1800\n'
  printf 'runtime_math_timeout_seconds=1800\n'
  printf 'evaluator_python=%s\n' "$("${EVAL_PYTHON}" -VV | tr '\n' ' ')"
} > "${OUTPUT_DIR}/runner_environment.txt"

server_success_total() {
  curl -fsS "${BASE_URL%/v1}/metrics" |
    awk '$1 ~ /^vllm:request_success_total\{/ { total += $2 } END { printf "%.0f\n", total }'
}

baseline_server_success="$(server_success_total)"
printf 'baseline_server_success=%s\n' "${baseline_server_success}" \
  >> "${OUTPUT_DIR}/runner_environment.txt"

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
    echo "ERROR: model server exited during official_v4 accuracy" >&2
    exit 1
  fi
  if ((elapsed >= next_progress)); then
    runner_completed="$(
      grep -Eo 'completed [0-9]+/2360' "${OUTPUT_DIR}/runner.log" |
        tail -1 || true
    )"
    current_server_success="$(server_success_total)"
    server_completed="$((current_server_success - baseline_server_success))"
    {
      printf '%s elapsed_seconds=%s %s server_completed=%s/2360\n' \
        "$(date -u +%FT%TZ)" "${elapsed}" \
        "${runner_completed:-completed unknown/2360}" "${server_completed}"
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
  echo "ERROR: official_v4 runner exited with status ${runner_status}" >&2
  exit "${runner_status}"
}

"${EVAL_PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
summary = json.loads((output / "summary.json").read_text())
predictions = output / "predictions.jsonl"
rows = sum(1 for _ in predictions.open("rb"))
bad_statuses = {
    key: value
    for key, value in summary["status_counts"].items()
    if key != "scored" and value
}
if summary["total"] != 2360 or summary["scored"] != 2360 or rows != 2360:
    raise SystemExit(
        f"incomplete official_v4 result: total={summary['total']} "
        f"scored={summary['scored']} rows={rows}"
    )
if bad_statuses:
    raise SystemExit(f"non-scored statuses: {bad_statuses}")
result = {
    "status": "passed",
    "total": 2360,
    "scored": 2360,
    "accuracy": summary["accuracy"],
    "status_counts": summary["status_counts"],
    "duration_seconds": summary["duration_seconds"],
    "predictions_rows": rows,
    "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
}
(output / "validation.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
PY
