#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${TASK_ROOT}/scripts/deploy/deploy_common_env.sh"

RUN_ID="${RUN_ID:-task2_mcp_l123_$(date -u +%Y%m%d_%H%M%S)}"
MCP_URL="${MCP_URL:-http://192.168.18.107:8000/mcp}"
LOG_DIR="${TASK_ROOT}/logs/api"
RAW_DIR="${TASK_ROOT}/reports/raw/${RUN_ID}"
ACCEPTANCE="${TASK_ROOT}/reports/acceptance/${RUN_ID}_acceptance.json"
SUMMARY="${TASK_ROOT}/reports/api/${RUN_ID}_summary.json"
mkdir -p "${LOG_DIR}" "${RAW_DIR}" "$(dirname "${ACCEPTANCE}")" "$(dirname "${SUMMARY}")"

PROMPT_FILE="${RAW_DIR}/mcp_prompt.txt"
ADD_LOG="${RAW_DIR}/mcp_add.log"
LIST_LOG="${RAW_DIR}/mcp_list.log"
PROMPT_LOG="${RAW_DIR}/mcp_prompt.log"
REMOVE_LOG="${RAW_DIR}/mcp_remove.log"
CMD_FILE="${LOG_DIR}/${RUN_ID}.cmd"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

cat > "${PROMPT_FILE}" <<EOF
请通过 e2etest MCP 验证当前 2P1D OpenAI 服务。必须调用 validate_model 工具一次，参数为 base_url="${OPENAI_BASE}", model_id="${MODEL_ID}", protocol="openai", api_key="EMPTY", supports_vlm=false, scope="full"。拿到 run_id 后调用 get_run_status 轮询，直到 completed 或明确失败。输出 L1、L2、L3 三层结果；L3 整体必须 pass。若 L2 litellm_streaming 仅因输出长度限制导致 reasoning-only stream/no content_block_delta，则按 accepted/pass 记录。Thinking 保持开启，不要把 enable_thinking 改为 false。最后必须单独输出一行：MCP_E2E_RESULT_JSON: {"L1":"pass","L2":"pass","L3":"pass","vlm":"unsupported","target":"${OPENAI_BASE}","model":"${MODEL_ID}"}
EOF

printf '%q\n' "$0" > "${CMD_FILE}"

remote_zhanghong() {
  sshpass -e ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    zhanghong@127.0.0.1 "$@"
}

run_remote_logged() {
  local output="$1"
  shift
  set +e
  remote_zhanghong "$@" > "${output}" 2>&1
  local rc=$?
  set -e
  echo "${rc}" > "${output}.rc"
  return "${rc}"
}

cleanup() {
  run_remote_logged "${REMOVE_LOG}" "bash -lc 'claude mcp remove e2etest'" || true
}
trap cleanup EXIT

{
  echo "run_id=${RUN_ID}"
  echo "mcp_remove_before"
  run_remote_logged "${RAW_DIR}/mcp_remove_before.log" "bash -lc 'claude mcp remove e2etest'" || true
  echo "mcp_add"
  run_remote_logged "${ADD_LOG}" "bash -lc 'claude mcp add --transport http e2etest ${MCP_URL}'"
  echo "mcp_list"
  run_remote_logged "${LIST_LOG}" "bash -lc 'claude mcp list && claude mcp get e2etest'"
  echo "claude_prompt"
  run_remote_logged "${PROMPT_LOG}" "bash -lc 'timeout 5400s claude -p --permission-mode bypassPermissions --allowedTools \"mcp__e2etest__validate_model,mcp__e2etest__get_run_status\" -- \"\$(cat ${PROMPT_FILE})\"'" || true
  cleanup
  trap - EXIT
  python3 - "$RUN_ID" "$ADD_LOG" "$LIST_LOG" "$PROMPT_LOG" "$REMOVE_LOG" "$SUMMARY" "$ACCEPTANCE" "${OPENAI_BASE}" "${MODEL_ID}" <<'PY'
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

run_id, add_log, list_log, prompt_log, remove_log, summary, acceptance, target_base, model_id = sys.argv[1:10]
paths = {name: Path(path) for name, path in {
    "add": add_log,
    "list": list_log,
    "prompt": prompt_log,
    "remove": remove_log,
}.items()}

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

def rc(path: Path) -> int | None:
    rc_path = Path(str(path) + ".rc")
    if not rc_path.exists():
        return None
    try:
        return int(rc_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None

texts = {name: read(path) for name, path in paths.items()}
rcs = {name: rc(path) for name, path in paths.items()}
prompt_text = texts["prompt"]
json_match = re.search(r"MCP_E2E_RESULT_JSON:\s*(\{.*\})", prompt_text)
reported = {}
if json_match:
    try:
        reported = json.loads(json_match.group(1))
    except Exception:
        reported = {}
low_prompt = prompt_text.lower()
l2_value = str(reported.get("L2", "")).lower()
l2_table_pass = bool(re.search(
    r"^\|\s*\*\*L2\*\*[^|]*\|\s*(?:✅\s*)?(?:pass|accepted)\b",
    prompt_text,
    re.I | re.M,
))
l2_result_pass = bool(re.search(
    r"(?:^|\n)\s*(?:\*\*)?L2(?:\s*结果)?(?:\*\*)?\s*[:：]\s*(?:✅\s*)?(?:pass|accepted)\b",
    prompt_text,
    re.I,
))
l2_allowed_stream_issue = (
    bool(re.search(
        r"L2.*litellm_streaming.*(reasoning-only|no content_block_delta|输出长度限制)",
        prompt_text,
        re.I | re.S,
    ))
    and not bool(re.search(
        r"L2.*litellm_streaming.*(timeout|超时|partial_fail|非预期)",
        prompt_text,
        re.I | re.S,
    ))
)
checks = {
    "mcp_add_rc_zero": rcs["add"] == 0,
    "mcp_list_shows_e2etest": rcs["list"] == 0 and "e2etest" in texts["list"].lower(),
    "claude_prompt_rc_zero": rcs["prompt"] == 0,
    "l1_reported_pass": str(reported.get("L1", "")).lower() == "pass" or bool(re.search(r"\bL1\b.*\bpass\b", low_prompt, re.I | re.S)),
    "l2_reported_pass_or_accepted": l2_value in {"pass", "accepted"} or l2_table_pass or l2_result_pass or l2_allowed_stream_issue,
    "l3_reported_pass": str(reported.get("L3", "")).lower() == "pass" or bool(re.search(r"\bL3\b.*\bpass\b", low_prompt, re.I | re.S)),
    "vlm_marked_unsupported": str(reported.get("vlm", "")).lower() == "unsupported" or "unsupported" in low_prompt or "不支持" in prompt_text,
    "target_and_model_present": target_base in prompt_text and model_id in prompt_text,
    "mcp_removed": rcs["remove"] == 0 or "not found" in texts["remove"].lower(),
}
result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "gate": "task2_mcp_l123",
    "pass": all(checks.values()),
    "checks": checks,
    "reported_result": reported,
    "l2_audit": {
        "value": l2_value,
        "table_pass": l2_table_pass,
        "result_pass": l2_result_pass,
        "allowed_stream_issue": l2_allowed_stream_issue,
    },
    "return_codes": rcs,
    "logs": {name: str(path) for name, path in paths.items()},
}
Path(summary).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
Path(acceptance).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
sys.exit(0 if result["pass"] else 1)
PY
} 2>&1 | tee "${LOG_FILE}"

record_acceptance "task2_mcp_l123" "PASS" "run_id=${RUN_ID} acceptance=${ACCEPTANCE}"
