#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${TASK_ROOT}/scripts/deploy/deploy_common_env.sh"
ENDPOINT="${ENDPOINT:-${MESSAGES_ENDPOINT}}"
MODEL="${MODEL:-${MODEL_ID}}"
RUN_ID="${RUN_ID:-task2_messages_cache_usage_$(date -u +%Y%m%d_%H%M%S)}"
RAW_DIR="${TASK_ROOT}/reports/raw/${RUN_ID}"
LOG_DIR="${TASK_ROOT}/logs/api"
ACCEPTANCE="${TASK_ROOT}/reports/acceptance/${RUN_ID}_acceptance.json"
SUMMARY="${TASK_ROOT}/reports/api/${RUN_ID}_summary.json"
CMD_FILE="${LOG_DIR}/${RUN_ID}.cmd"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

mkdir -p "${RAW_DIR}" "${LOG_DIR}" "$(dirname "${ACCEPTANCE}")" "$(dirname "${SUMMARY}")"
printf 'ENDPOINT=%q MODEL=%q RUN_ID=%q %q\n' "${ENDPOINT}" "${MODEL}" "${RUN_ID}" "$0" > "${CMD_FILE}"

export ENDPOINT MODEL RUN_ID RAW_DIR ACCEPTANCE SUMMARY LOG_FILE
python3 - <<'PY' 2>&1 | tee "${LOG_FILE}"
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

endpoint = os.environ["ENDPOINT"]
model = os.environ["MODEL"]
run_id = os.environ["RUN_ID"]
raw_dir = Path(os.environ["RAW_DIR"])
acceptance_path = Path(os.environ["ACCEPTANCE"])
summary_path = Path(os.environ["SUMMARY"])

common_prefix = "\n".join(
    [
        f"{model} prefix cache verification shared context.",
        "The following numbered facts are intentionally stable across all requests.",
        *[
            f"Stable fact {i:03d}: this sentence is part of the identical public prefix used to verify vLLM prefix cache reuse."
            for i in range(1, 121)
        ],
        "The shared prefix ends after this line; the final question differs per request.",
    ]
)
questions = [
    "Request one warms the prefix cache. Reply with exactly: warmup-ok",
    "Request two checks cache reuse. Reply with exactly: cache-check-ok",
    "Request three checks cache reuse again. Reply with exactly: cache-check-again-ok",
]


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def post_messages(index: int, question: str) -> dict:
    payload = {
        "model": model,
        "system": "You are a concise assistant. Keep reasoning enabled and answer briefly.",
        "messages": [
            {
                "role": "user",
                "content": f"{common_prefix}\n\nFinal question {index}: {question}",
            }
        ],
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
    }
    payload_path = raw_dir / f"{run_id}_request_{index}.json"
    raw_path = raw_dir / f"{run_id}_response_{index}.json"
    write_json(payload_path, payload)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    started = time.perf_counter()
    status = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
            response = json.loads(body)
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
        try:
            response = json.loads(body)
        except Exception:
            response = {"raw_error": body}
    elapsed = round(time.perf_counter() - started, 3)
    write_json(raw_path, response)
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    item = {
        "index": index,
        "status": status,
        "elapsed_sec": elapsed,
        "payload_path": str(payload_path),
        "raw_response_path": str(raw_path),
        "usage": usage,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
    }
    print(
        f"request={index} status={status} elapsed_sec={elapsed} "
        f"usage={json.dumps(usage, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    return item


items = []
for idx, question in enumerate(questions, start=1):
    items.append(post_messages(idx, question))
    time.sleep(1)

required = [
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
]

checks = {
    "three_requests_completed": len(items) == 3,
    "all_http_200": all(item["status"] == 200 for item in items),
    "all_usage_has_four_fields": all(
        all(field in item["usage"] for field in required) for item in items
    ),
    "all_usage_fields_integer": all(
        all(isinstance(item["usage"].get(field), int) for field in required)
        for item in items
    ),
    "at_least_one_cache_hit": any(
        isinstance(item["usage"].get("cache_read_input_tokens"), int)
        and item["usage"].get("cache_read_input_tokens") > 0
        for item in items
    ),
    "all_total_input_explainable": all(
        isinstance(item["usage"].get("input_tokens"), int)
        and isinstance(item["usage"].get("cache_creation_input_tokens"), int)
        and isinstance(item["usage"].get("cache_read_input_tokens"), int)
        and (
            item["usage"]["input_tokens"]
            + item["usage"]["cache_creation_input_tokens"]
            + item["usage"]["cache_read_input_tokens"]
        )
        > 0
        for item in items
    ),
}
total_cache_creation_tokens = sum(
    item["usage"].get("cache_creation_input_tokens", 0)
    for item in items
    if isinstance(item.get("usage"), dict)
    and isinstance(item["usage"].get("cache_creation_input_tokens"), int)
)
total_cache_read_tokens = sum(
    item["usage"].get("cache_read_input_tokens", 0)
    for item in items
    if isinstance(item.get("usage"), dict)
    and isinstance(item["usage"].get("cache_read_input_tokens"), int)
)
cache_observed_tokens = total_cache_creation_tokens + total_cache_read_tokens
cache_hit_rate = (
    total_cache_read_tokens / cache_observed_tokens
    if cache_observed_tokens > 0
    else 0.0
)
request_cache_hit_rate = (
    sum(
        1
        for item in items
        if isinstance(item.get("usage"), dict)
        and isinstance(item["usage"].get("cache_read_input_tokens"), int)
        and item["usage"]["cache_read_input_tokens"] > 0
    )
    / len(items)
    if items
    else 0.0
)
result = {
    "run_id": run_id,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "endpoint": endpoint,
    "model": model,
    "max_tokens": 32,
    "prefix_cache": "enabled",
    "requests": items,
    "cache_hit_rate": cache_hit_rate,
    "request_cache_hit_rate": request_cache_hit_rate,
    "total_cache_creation_input_tokens": total_cache_creation_tokens,
    "total_cache_read_input_tokens": total_cache_read_tokens,
    "checks": checks,
    "pass": all(checks.values()),
}
write_json(summary_path, result)
write_json(acceptance_path, result)
print(f"summary={summary_path}")
print(f"acceptance={acceptance_path}")
raise SystemExit(0 if result["pass"] else 1)
PY
