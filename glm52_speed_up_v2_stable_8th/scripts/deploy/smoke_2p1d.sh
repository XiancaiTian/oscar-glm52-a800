#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy_common_env.sh"
source "${TASK_ROOT}/state/2p1d_endpoint.env"

mkdir -p "${TASK_ROOT}/logs/deploy" "${TASK_ROOT}/reports/raw" "${TASK_ROOT}/reports/acceptance"
RUN_ID="2p1d_smoke_$(date -u +%Y%m%d_%H%M%S)"
LOG="${TASK_ROOT}/logs/deploy/${RUN_ID}.log"
ACCEPTANCE="${TASK_ROOT}/reports/acceptance/${RUN_ID}_acceptance.json"
RAW_DIR="${TASK_ROOT}/reports/raw/${RUN_ID}"
mkdir -p "${RAW_DIR}"

request() {
  local name="$1"
  shift
  echo "[$(date -u +%FT%TZ)] ${name}: $*" | tee -a "${LOG}"
  "$@" > "${RAW_DIR}/${name}.out" 2> "${RAW_DIR}/${name}.err"
}

request proxy_health curl -fsS --max-time 30 "http://${DECODE_HEAD}:${PROXY_PORT}/healthcheck"
request proxy_models curl -fsS --max-time 30 "http://${DECODE_HEAD}:${PROXY_PORT}/v1/models"
request p0_models curl -fsS --max-time 30 "http://${PREFILL0_HEAD}:${PREFILL_PORT}/v1/models"
request p1_models curl -fsS --max-time 30 "http://${PREFILL1_HEAD}:${PREFILL_PORT}/v1/models"
request d0_models curl -fsS --max-time 30 "http://${DECODE_HEAD}:${DECODE_PORT}/v1/models"

request chat_completion curl -fsS --max-time 240 \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{"model":"'"${MODEL_ID}"'","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8,"temperature":0,"stream":false,"chat_template_kwargs":{"enable_thinking":true}}' \
  "${BASE_URL}/chat/completions"

request messages curl -fsS --max-time 240 \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{"model":"'"${MODEL_ID}"'","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8,"temperature":0,"stream":false}' \
  "${BASE_URL}/messages"

python3 - "${RUN_ID}" "${RAW_DIR}" "${ACCEPTANCE}" "${BASE_URL}" "${MODEL_ID}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_id, raw_dir, acceptance, base_url, model = sys.argv[1:6]
raw = Path(raw_dir)
checks = {}
for name in [
    "proxy_health",
    "proxy_models",
    "p0_models",
    "p1_models",
    "d0_models",
    "chat_completion",
    "messages",
]:
    out = raw / f"{name}.out"
    err = raw / f"{name}.err"
    checks[name] = out.exists() and out.stat().st_size > 0 and (
        not err.exists() or err.stat().st_size == 0
    )

for name in ["proxy_models", "p0_models", "p1_models", "d0_models"]:
    try:
        body = json.loads((raw / f"{name}.out").read_text(encoding="utf-8"))
        checks[f"{name}_has_model"] = any(
            isinstance(item, dict) and item.get("id") == model
            for item in body.get("data", [])
        )
    except Exception:
        checks[f"{name}_has_model"] = False

for name in ["chat_completion", "messages"]:
    try:
        body = json.loads((raw / f"{name}.out").read_text(encoding="utf-8"))
        checks[f"{name}_json"] = isinstance(body, dict) and not isinstance(
            body.get("error"), dict
        )
    except Exception:
        checks[f"{name}_json"] = False

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "gate": "2p1d_smoke",
    "pass": all(checks.values()),
    "base_url": base_url,
    "model": model,
    "checks": checks,
    "raw_dir": str(raw),
}
Path(acceptance).write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["pass"] else 1)
PY

record_acceptance "2p1d_smoke" "PASS" "run_id=${RUN_ID} acceptance=${ACCEPTANCE}"
echo "smoke_run_id=${RUN_ID}"
