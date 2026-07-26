#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHASE1_RUN_ID="${PHASE1_RUN_ID:?set PHASE1_RUN_ID to the running server run ID}"
SOURCE_ATTEMPT_ID="${SOURCE_ATTEMPT_ID:?set SOURCE_ATTEMPT_ID to the failed full attempt ID}"
RETRY_ATTEMPT_ID="${RETRY_ATTEMPT_ID:-retry_$(date -u +%Y%m%dT%H%M%SZ)}"
MERGED_ATTEMPT_ID="${MERGED_ATTEMPT_ID:-merged_$(date -u +%Y%m%dT%H%M%SZ)}"

SERVER_RUN_DIR="${PROJECT_ROOT}/artifacts/phase1/${PHASE1_RUN_ID}"
OUTPUT_ROOT="${SERVER_RUN_DIR}/official_v4_accuracy"
SOURCE_DIR="${OUTPUT_ROOT}/${SOURCE_ATTEMPT_ID}"
RETRY_DIR="${OUTPUT_ROOT}/${RETRY_ATTEMPT_ID}"
MERGED_DIR="${OUTPUT_ROOT}/${MERGED_ATTEMPT_ID}"
RUNTIME_SUITE_DIR="${RETRY_DIR}/runtime_suite"
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
diff -u "${LOCK_FILE}" <(uv pip freeze --python "${EVAL_PYTHON}")

[[ -f "${SOURCE_DIR}/predictions.jsonl" ]] || {
  echo "ERROR: source predictions are missing: ${SOURCE_DIR}" >&2
  exit 1
}
[[ -f "${SOURCE_DIR}/summary.json" ]] || {
  echo "ERROR: source summary is missing: ${SOURCE_DIR}" >&2
  exit 1
}
[[ ! -e "${RETRY_DIR}" ]] || {
  echo "ERROR: retry attempt already exists: ${RETRY_DIR}" >&2
  exit 1
}
[[ ! -e "${MERGED_DIR}" ]] || {
  echo "ERROR: merged attempt already exists: ${MERGED_DIR}" >&2
  exit 1
}

server_pid="$(<"${SERVER_RUN_DIR}/server.pid")"
kill -0 "${server_pid}" 2>/dev/null || {
  echo "ERROR: server PID ${server_pid} is not running" >&2
  exit 1
}
curl -fsS "${BASE_URL%/v1}/health" >/dev/null

mkdir -p "${RUNTIME_SUITE_DIR}"
"${EVAL_PYTHON}" - \
  "${SUITE_DIR}/manifest.jsonl" \
  "${SOURCE_DIR}/predictions.jsonl" \
  "${SOURCE_DIR}/summary.json" \
  "${SOURCE_DIR}/runtime_suite/eval_config.json" \
  "${RUNTIME_SUITE_DIR}/manifest.jsonl" \
  "${RUNTIME_SUITE_DIR}/eval_config.json" \
  "${RETRY_DIR}/failure_metadata.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    manifest_path,
    predictions_path,
    summary_path,
    source_config_path,
    retry_manifest_path,
    retry_config_path,
    metadata_path,
) = map(Path, sys.argv[1:])


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = read_jsonl(predictions_path)
if summary["total"] != 2360 or len(rows) != 2360:
    raise SystemExit("source attempt does not contain exactly 2360 rows")
if summary["scored"] != 2353:
    raise SystemExit(f"unexpected source scored count: {summary['scored']}")
if summary["status_counts"] != {"request_failed": 7, "scored": 2353}:
    raise SystemExit(f"unexpected source statuses: {summary['status_counts']}")
if len({row["id"] for row in rows}) != 2360:
    raise SystemExit("source prediction IDs are not unique")

failed = [row for row in rows if row["evaluator_status"] != "scored"]
expected_ids = [f"gsm8k:{index:06d}" for index in range(1311, 1318)]
if [row["id"] for row in failed] != expected_ids:
    raise SystemExit(f"unexpected failed IDs: {[row['id'] for row in failed]}")
for row in failed:
    if row["evaluator_status"] != "request_failed":
        raise SystemExit(f"unexpected failed status for {row['id']}")
    if "read timeout=300" not in row["error_message"]:
        raise SystemExit(f"unexpected failure reason for {row['id']}")

manifest = read_jsonl(manifest_path)
manifest_by_id = {row["id"]: row for row in manifest}
if len(manifest_by_id) != len(manifest):
    raise SystemExit("official manifest IDs are not unique")
retry_rows = [manifest_by_id[sample_id] for sample_id in expected_ids]
for sample, failed_row in zip(retry_rows, failed, strict=True):
    prompt_hash = hashlib.sha256(sample["prompt"].encode("utf-8")).hexdigest()
    if prompt_hash != failed_row["prompt_hash"]:
        raise SystemExit(f"prompt hash mismatch for {sample['id']}")

config = json.loads(source_config_path.read_text(encoding="utf-8"))
if config["timeouts_seconds"]["math_reasoning"] != 300:
    raise SystemExit("source math timeout is not 300 seconds")
config["timeouts_seconds"]["math_reasoning"] = 900

retry_manifest_path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retry_rows),
    encoding="utf-8",
)
retry_config_path.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
metadata = {
    "source_total": 2360,
    "source_scored": 2353,
    "source_predictions_sha256": hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest(),
    "retry_ids": expected_ids,
    "source_failures": [
        {
            "id": row["id"],
            "prompt_hash": row["prompt_hash"],
            "error_message": row["error_message"],
        }
        for row in failed
    ],
    "source_math_timeout_seconds": 300,
    "retry_math_timeout_seconds": 900,
}
metadata_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

command=(
  "${EVAL_PYTHON}" "${RUNNER}"
  --suite-dir "${RUNTIME_SUITE_DIR}"
  --output-dir "${RETRY_DIR}"
  --base-url "${BASE_URL}"
  --model "${MODEL_NAME}"
  --concurrency 7
  --benchmarks GSM8K
)
printf '%q ' "${command[@]}" > "${RETRY_DIR}/runner_command.txt"
printf '\n' >> "${RETRY_DIR}/runner_command.txt"
{
  printf 'started_at_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'source_attempt_id=%s\n' "${SOURCE_ATTEMPT_ID}"
  printf 'runner_sha256=%s\n' "${actual_runner_sha}"
  printf 'retry_manifest_sha256=%s\n' \
    "$(sha256sum "${RUNTIME_SUITE_DIR}/manifest.jsonl" | awk '{print $1}')"
  printf 'runtime_eval_config_sha256=%s\n' \
    "$(sha256sum "${RUNTIME_SUITE_DIR}/eval_config.json" | awk '{print $1}')"
  printf 'runtime_math_timeout_seconds=900\n'
  printf 'evaluator_python=%s\n' "$("${EVAL_PYTHON}" -VV | tr '\n' ' ')"
} > "${RETRY_DIR}/runner_environment.txt"

"${command[@]}" > "${RETRY_DIR}/runner.log" 2>&1 &
runner_pid=$!
started="$(date +%s)"
next_progress=600
while kill -0 "${runner_pid}" 2>/dev/null; do
  sleep 60
  elapsed="$(( $(date +%s) - started ))"
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${runner_pid}" 2>/dev/null || true
    wait "${runner_pid}" 2>/dev/null || true
    echo "ERROR: model server exited during official_v4 retry" >&2
    exit 1
  fi
  if ((elapsed >= next_progress)); then
    {
      printf '%s elapsed_seconds=%s\n' "$(date -u +%FT%TZ)" "${elapsed}"
      nvidia-smi \
        --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
    } | tee -a "${RETRY_DIR}/progress_10min.log"
    next_progress="$((next_progress + 600))"
  fi
done
wait "${runner_pid}"

mkdir -p "${MERGED_DIR}"
"${EVAL_PYTHON}" - \
  "${RUNNER}" \
  "${SUITE_DIR}/manifest.jsonl" \
  "${SOURCE_DIR}" \
  "${RETRY_DIR}" \
  "${MERGED_DIR}" <<'PY'
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

runner_path, manifest_path, source_dir, retry_dir, merged_dir = map(
    Path, sys.argv[1:]
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("official_v4_runner", runner_path)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)

manifest = read_jsonl(manifest_path)
source_rows = read_jsonl(source_dir / "predictions.jsonl")
retry_rows = read_jsonl(retry_dir / "predictions.jsonl")
retry_summary = json.loads((retry_dir / "summary.json").read_text(encoding="utf-8"))
if retry_summary["total"] != 7 or retry_summary["scored"] != 7:
    raise SystemExit(f"retry is incomplete: {retry_summary}")
if retry_summary["status_counts"] != {"scored": 7}:
    raise SystemExit(f"retry has non-scored rows: {retry_summary['status_counts']}")
if len(retry_rows) != 7 or len({row["id"] for row in retry_rows}) != 7:
    raise SystemExit("retry predictions do not contain 7 unique rows")

retry_by_id = {row["id"]: row for row in retry_rows}
source_failed_ids = {
    row["id"] for row in source_rows if row["evaluator_status"] != "scored"
}
if set(retry_by_id) != source_failed_ids:
    raise SystemExit("retry IDs do not exactly match source failed IDs")

merged_rows = [
    retry_by_id.get(row["id"], row)
    for row in source_rows
]
if len(merged_rows) != 2360 or len({row["id"] for row in merged_rows}) != 2360:
    raise SystemExit("merged predictions do not contain 2360 unique rows")
if [row["id"] for row in merged_rows] != [row["id"] for row in manifest]:
    raise SystemExit("merged prediction order does not match official manifest")
for sample, row in zip(manifest, merged_rows, strict=True):
    prompt_hash = hashlib.sha256(sample["prompt"].encode("utf-8")).hexdigest()
    if row["prompt_hash"] != prompt_hash:
        raise SystemExit(f"merged prompt hash mismatch for {row['id']}")
    if row["evaluator_status"] != "scored" or row["score"] is None:
        raise SystemExit(f"merged row is not scored: {row['id']}")

runner.write_jsonl(merged_dir / "predictions.jsonl", merged_rows)
by_benchmark = runner.summarize(merged_rows, "benchmark")
by_task_type = runner.summarize(merged_rows, "task_type")
runner.write_json(merged_dir / "summary_by_benchmark.json", by_benchmark)
runner.write_json(merged_dir / "summary_by_task_type.json", by_task_type)
runner.write_csv(merged_dir / "summary_by_benchmark.csv", by_benchmark)
runner.write_csv(merged_dir / "summary_by_task_type.csv", by_task_type)
failed = [
    row
    for row in merged_rows
    if row["evaluator_status"] != "scored" or row.get("score") != 1.0
]
runner.write_jsonl(merged_dir / "failed_cases.jsonl", failed)

source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
overall = {
    "total": 2360,
    "scored": 2360,
    "accuracy": sum(row["score"] for row in merged_rows) / 2360,
    "status_counts": dict(Counter(row["evaluator_status"] for row in merged_rows)),
    "started_at_unix": source_summary["started_at_unix"],
    "ended_at_unix": retry_summary["ended_at_unix"],
    "duration_seconds": (
        source_summary["duration_seconds"] + retry_summary["duration_seconds"]
    ),
}
runner.write_json(merged_dir / "summary.json", overall)

provenance = {
    "merge_policy": "replace_only_source_request_failed_rows_by_exact_id",
    "source_attempt": source_dir.name,
    "retry_attempt": retry_dir.name,
    "source_predictions_sha256": sha256(source_dir / "predictions.jsonl"),
    "retry_predictions_sha256": sha256(retry_dir / "predictions.jsonl"),
    "replaced_ids": sorted(source_failed_ids),
    "composite_duration_policy": "sum_of_source_and_retry_runner_durations",
}
runner.write_json(merged_dir / "merge_provenance.json", provenance)

validation = {
    "status": "passed",
    "total": 2360,
    "scored": 2360,
    "accuracy": overall["accuracy"],
    "status_counts": overall["status_counts"],
    "predictions_rows": 2360,
    "predictions_sha256": sha256(merged_dir / "predictions.jsonl"),
    "summary_sha256": sha256(merged_dir / "summary.json"),
    "retry_total": 7,
    "retry_scored": 7,
    "replaced_ids": sorted(source_failed_ids),
}
runner.write_json(merged_dir / "validation.json", validation)
print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
PY

sha256sum \
  "${RETRY_DIR}/predictions.jsonl" \
  "${RETRY_DIR}/summary.json" \
  "${MERGED_DIR}/predictions.jsonl" \
  "${MERGED_DIR}/summary.json" \
  "${MERGED_DIR}/validation.json"
