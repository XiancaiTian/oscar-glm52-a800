#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHASE1_RUN_ID="${PHASE1_RUN_ID:?set PHASE1_RUN_ID to the running server run ID}"
SERVER_RUN_DIR="${PROJECT_ROOT}/artifacts/phase1/${PHASE1_RUN_ID}"
ATTEMPT_ID="${ACCURACY_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${SERVER_RUN_DIR}/official_v4_accuracy"
OUTPUT_DIR="${OUTPUT_ROOT}/${ATTEMPT_ID}"
RUNTIME_SUITE_DIR="${OUTPUT_DIR}/runtime_suite"
EVAL_ROOT="/nfs/AE/txc/vllm_turbo_baseline_acc"
SUITE_DIR="${EVAL_ROOT}/accuracy_suites/model_agnostic_accuracy_official_v4"
RUNNER="${EVAL_ROOT}/tools/run_accuracy_suite.py"
EVAL_PYTHON="${PROJECT_ROOT}/artifacts/phase1-eval-venv/bin/python"
LOCK_FILE="${PROJECT_ROOT}/configs/phase1/evaluator-requirements.lock.txt"
BASE_URL="${BASE_URL:-http://127.0.0.1:18080/v1}"
MODEL_NAME="glm-5.2-fp8-pruned-staticgate-e154"

expected_runner_sha="fc374ff4c4715e37d515d37aa794b3e649dc1710034d3355c69c21efa1a8aeff"
actual_runner_sha="$(sha256sum "${RUNNER}" | awk '{print $1}')"
[[ "${actual_runner_sha}" == "${expected_runner_sha}" ]] || {
  echo "ERROR: official_v4 runner changed: ${actual_runner_sha}" >&2
  exit 1
}
requirement_sha="$(sha256sum "${EVAL_ROOT}/requirements-accuracy-suite.txt" | awk '{print $1}')"
[[ "${requirement_sha}" == "1667a30b31747c80f286ee7a33dc7131c88aa0ff09ae89d1dd2cb6ec35b779c6" ]] || {
  echo "ERROR: evaluator requirements file changed: ${requirement_sha}" >&2
  exit 1
}
[[ -x "${EVAL_PYTHON}" ]] || {
  echo "ERROR: evaluator venv is missing: ${EVAL_PYTHON}" >&2
  exit 1
}
diff -u "${LOCK_FILE}" <(
  uv pip freeze --python "${EVAL_PYTHON}"
)
[[ -f "${SERVER_RUN_DIR}/runtime_manifest.json" ]] || {
  echo "ERROR: server runtime manifest is missing" >&2
  exit 1
}
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
if config["timeouts_seconds"]["code"] != 600:
    raise SystemExit(
        f"unexpected source code timeout: {config['timeouts_seconds']['code']}"
    )
config["timeouts_seconds"]["code"] = 900
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
  printf 'runtime_code_timeout_seconds=900\n'
  printf 'evaluator_python=%s\n' "$("${EVAL_PYTHON}" -VV | tr '\n' ' ')"
} > "${OUTPUT_DIR}/runner_environment.txt"

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
    completed="$(grep -Eo 'completed [0-9]+/2360' "${OUTPUT_DIR}/runner.log" | tail -1 || true)"
    {
      printf '%s elapsed_seconds=%s %s\n' \
        "$(date -u +%FT%TZ)" "${elapsed}" "${completed:-completed unknown/2360}"
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
if summary["total"] != 2360:
    raise SystemExit(f"unexpected total: {summary['total']}")
if summary["scored"] != 2360:
    raise SystemExit(f"unexpected scored: {summary['scored']}")
if rows != 2360:
    raise SystemExit(f"unexpected predictions rows: {rows}")
if bad_statuses:
    raise SystemExit(f"non-scored statuses: {bad_statuses}")
digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
result = {
    "status": "passed",
    "total": summary["total"],
    "scored": summary["scored"],
    "accuracy": summary["accuracy"],
    "status_counts": summary["status_counts"],
    "duration_seconds": summary["duration_seconds"],
    "predictions_rows": rows,
    "predictions_sha256": digest,
}
(output / "validation.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
PY
