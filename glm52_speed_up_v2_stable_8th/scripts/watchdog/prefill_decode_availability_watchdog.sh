#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${TASK_ROOT}/scripts/deploy/load_deploy_config.sh"
load_deploy_config
WRAPPER="${WRAPPER:-${TASK_ROOT}/scripts/deploy/deploy_8th_claude204_current_verified.sh}"

INTERVAL_SEC="${AVAIL_WATCHDOG_INTERVAL_SEC:-1800}"
CONFIRM_ATTEMPTS="${AVAIL_WATCHDOG_CONFIRM_ATTEMPTS:-5}"
CONFIRM_INTERVAL_SEC="${AVAIL_WATCHDOG_CONFIRM_INTERVAL_SEC:-60}"
STATE_DIR="${TASK_ROOT}/state/watchdog"
LOG_DIR="${TASK_ROOT}/logs/watchdog"
RAW_DIR="${TASK_ROOT}/reports/raw"
ACCEPT_DIR="${TASK_ROOT}/reports/acceptance"
PID_FILE="${STATE_DIR}/prefill_decode_availability_watchdog.pid"
LOCK_FILE="${STATE_DIR}/prefill_decode_availability_recover.lock"
EVENTS_FILE="${STATE_DIR}/prefill_decode_availability_events.jsonl"

MESSAGES_ENDPOINT="${MESSAGES_ENDPOINT:-${PROXY_ROOT}/v1/messages}"
P0_PREFILL="${P0_PREFILL:-${PREFILL0_HEAD}:${PREFILL_PORT}}"
P1_PREFILL="${P1_PREFILL:-${PREFILL1_HEAD}:${PREFILL_PORT}}"
ALLOWED_HOSTS_CSV="${ALLOWED_HOSTS_CSV:-${ALLOWED_HOSTS}}"
FORBIDDEN_HOSTS_CSV="${FORBIDDEN_HOSTS_CSV:-}"

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${RAW_DIR}" "${ACCEPT_DIR}"

json_event() {
  local event_type="$1"
  local status="$2"
  local detail="${3:-}"
  local extra_json="${4:-}"
  if [[ -z "${extra_json}" ]]; then
    extra_json="{}"
  fi
  python3 - "${EVENTS_FILE}" "${event_type}" "${status}" "${detail}" "${extra_json}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
event_type, status, detail, extra_json = sys.argv[2:6]
try:
    extra = json.loads(extra_json)
except Exception:
    extra = {"extra_parse_error": extra_json}
row = {
    "time_utc": datetime.now(timezone.utc).isoformat(),
    "event_type": event_type,
    "status": status,
    "detail": detail,
}
row.update(extra)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

assert_safety() {
  local text
  text="${PROXY_ROOT} ${MESSAGES_ENDPOINT} ${P0_PREFILL} ${P1_PREFILL} ${WRAPPER} ${TASK_ROOT}"
  if [[ -n "${FORBIDDEN_HOSTS_CSV}" ]]; then
    IFS=',' read -r -a forbidden_hosts <<< "${FORBIDDEN_HOSTS_CSV}"
    for host in "${forbidden_hosts[@]}"; do
      [[ -n "${host}" ]] || continue
      if grep -Fq "${host}" <<< "${text}"; then
        echo "ERROR: forbidden host appears in availability watchdog config: ${host}" >&2
        return 98
      fi
    done
  fi
  while IFS= read -r host; do
    [[ -n "${host}" ]] || continue
    case ",${ALLOWED_HOSTS_CSV}," in
      *",${host},"*) ;;
      *) echo "ERROR: non-allowed host appears in availability watchdog config: ${host}" >&2; return 96 ;;
    esac
  done < <(grep -Eo '192\.168\.16\.[0-9]+' <<< "${text}" | sort -u)
  [[ "${WRAPPER}" == "${TASK_ROOT}/scripts/deploy/deploy_8th_claude204_current_verified.sh" ]]
}

run_canary() {
  local run_id="$1"
  local raw_dir="${RAW_DIR}/${run_id}"
  local accept_path="${ACCEPT_DIR}/${run_id}_acceptance.json"
  mkdir -p "${raw_dir}"
  python3 - \
    "${run_id}" \
    "${MODEL_ID}" \
    "${MESSAGES_ENDPOINT}" \
    "${P0_PREFILL}" \
    "${P1_PREFILL}" \
    "${raw_dir}" \
    "${accept_path}" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

run_id, model, endpoint, p0, p1, raw_dir, accept_path = sys.argv[1:8]
raw_dir = Path(raw_dir)
accept_path = Path(accept_path)

def post_prefill(group: str, target: str) -> dict:
    stream_payload = {
        "model": model,
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{run_id} {group} Prefill+Decode stream-release canary. "
                    "Generate a short answer and finish normally."
                ),
            }
        ],
        "_prefill_target": target,
    }
    started = time.time()
    stream_req = urllib.request.Request(
        endpoint,
        data=json.dumps(stream_payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
            "x-request-id": f"{run_id}_{group}_stream",
        },
        method="POST",
    )
    status = None
    error = None
    message_stop_seen = False
    event_count = 0
    byte_count = 0
    first_event_ms = None
    last_event_ms = None
    usage = {}
    try:
        with urllib.request.urlopen(stream_req, timeout=180) as resp, (
            raw_dir / f"{group.lower()}_stream_events.jsonl"
        ).open("w", encoding="utf-8") as out:
            status = resp.status
            current_event = None
            for raw_line in resp:
                now_ms = round((time.time() - started) * 1000, 3)
                line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                byte_count += len(raw_line)
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                raw_data = line.split(":", 1)[1].strip()
                try:
                    parsed = json.loads(raw_data)
                except Exception:
                    parsed = {"raw": raw_data}
                event_count += 1
                if first_event_ms is None:
                    first_event_ms = now_ms
                last_event_ms = now_ms
                out.write(
                    json.dumps(
                        {"elapsed_ms": now_ms, "event": current_event, "data": parsed},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if isinstance(parsed, dict):
                    if parsed.get("type") == "message_stop":
                        message_stop_seen = True
                    if isinstance(parsed.get("usage"), dict):
                        usage.update(parsed["usage"])
                    message = parsed.get("message")
                    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
                        usage.update(message["usage"])
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", "replace")[:1000]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    stream_elapsed_ms = round((time.time() - started) * 1000, 3)

    follow_payload = {
        "model": model,
        "max_tokens": 1,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": f"{run_id} {group} follow-up after stream release.",
            }
        ],
        "_prefill_target": target,
    }
    follow_started = time.time()
    follow_status = None
    follow_error = None
    follow_body = ""
    follow_usage = None
    follow_req = urllib.request.Request(
        endpoint,
        data=json.dumps(follow_payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
            "x-request-id": f"{run_id}_{group}_followup",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(follow_req, timeout=120) as resp:
            follow_body = resp.read().decode("utf-8", "replace")
            follow_status = resp.status
    except urllib.error.HTTPError as exc:
        follow_body = exc.read().decode("utf-8", "replace")
        follow_status = exc.code
        follow_error = follow_body[:1000]
    except Exception as exc:  # noqa: BLE001
        follow_error = f"{type(exc).__name__}: {exc}"
    follow_elapsed_ms = round((time.time() - follow_started) * 1000, 3)
    try:
        follow_parsed = json.loads(follow_body) if follow_body else None
    except Exception:
        follow_parsed = None
    follow_output_present = bool(follow_body) and follow_status == 200
    if isinstance(follow_parsed, dict):
        if follow_parsed.get("error"):
            follow_output_present = False
        content = follow_parsed.get("content")
        if isinstance(content, list):
            follow_output_present = follow_output_present and bool(content)
        follow_usage = follow_parsed.get("usage")
    stream_pass = bool(
        status == 200
        and event_count > 0
        and message_stop_seen
        and error is None
    )
    followup_pass = bool(follow_status == 200 and follow_output_present and not follow_error)
    row = {
        "group": group,
        "target_prefill": target,
        "status": status,
        "pass": bool(stream_pass and followup_pass),
        "stream_pass": stream_pass,
        "stream_status": status,
        "stream_elapsed_ms": stream_elapsed_ms,
        "message_stop_seen": message_stop_seen,
        "event_count": event_count,
        "byte_count": byte_count,
        "first_event_ms": first_event_ms,
        "last_event_ms": last_event_ms,
        "usage": usage,
        "error": error,
        "followup_pass": followup_pass,
        "followup_status": follow_status,
        "followup_elapsed_ms": follow_elapsed_ms,
        "followup_usage": follow_usage,
        "followup_error": follow_error,
        "followup_body_excerpt": follow_body[:1000],
    }
    (raw_dir / f"{group.lower()}_request.json").write_text(
        json.dumps(stream_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (raw_dir / f"{group.lower()}_followup_request.json").write_text(
        json.dumps(follow_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (raw_dir / f"{group.lower()}_response.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return row

checks = [post_prefill("P0", p0), post_prefill("P1", p1)]
payload_out = {
    "schema_version": "1.0",
    "gate": "prefill_decode_real_inference_canary",
    "pass": all(row["pass"] for row in checks),
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "run_id": run_id,
    "endpoint": endpoint,
    "model": model,
    "checks": checks,
    "failed_groups": [row["group"] for row in checks if not row["pass"]],
    "raw_dir": str(raw_dir),
}
accept_path.write_text(
    json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload_out, ensure_ascii=False, indent=2))
raise SystemExit(0 if payload_out["pass"] else 10)
PY
}

latest_acceptance_for() {
  local run_id="$1"
  printf '%s\n' "${ACCEPT_DIR}/${run_id}_acceptance.json"
}

confirm_unavailability() {
  local base_run_id="$1"
  local summary_path="${ACCEPT_DIR}/${base_run_id}_confirmation_acceptance.json"
  local paths=()
  local i confirm_run accept_path rc

  for ((i = 1; i <= CONFIRM_ATTEMPTS; i++)); do
    confirm_run="${base_run_id}_confirm_${i}"
    set +e
    run_canary "${confirm_run}" > "${LOG_DIR}/${confirm_run}.log" 2>&1
    rc=$?
    set -e
    accept_path="$(latest_acceptance_for "${confirm_run}")"
    paths+=("${accept_path}")
    json_event \
      "availability_confirm_attempt" \
      "$([[ "${rc}" == "0" ]] && echo pass || echo fail)" \
      "attempt=${i}/${CONFIRM_ATTEMPTS}" \
      "{\"acceptance\":\"${accept_path}\"}"
    if ((i < CONFIRM_ATTEMPTS)); then
      sleep "${CONFIRM_INTERVAL_SEC}"
    fi
  done

  python3 - "${summary_path}" "${CONFIRM_ATTEMPTS}" "${CONFIRM_INTERVAL_SEC}" "${paths[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path(sys.argv[1])
attempts = int(sys.argv[2])
interval_sec = int(sys.argv[3])
paths = [Path(p) for p in sys.argv[4:]]

rows = []
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    checks = {row["group"]: row for row in data.get("checks", [])}
    rows.append(
        {
            "acceptance": str(path),
            "pass": bool(data.get("pass")),
            "P0": bool(checks.get("P0", {}).get("pass")),
            "P1": bool(checks.get("P1", {}).get("pass")),
            "checks": data.get("checks", []),
        }
    )

p0_pass = sum(1 for row in rows if row["P0"])
p1_pass = sum(1 for row in rows if row["P1"])
p0_fail = attempts - p0_pass
p1_fail = attempts - p1_pass

decision = "uncertain_no_restart"
reason = "confirmation results are mixed; do not restart"
if p0_fail == attempts and p1_pass == attempts:
    decision = "recover_P0_only"
    reason = "P0 failed every confirmation canary while P1 passed every confirmation canary"
elif p1_fail == attempts and p0_pass == attempts:
    decision = "recover_P1_only"
    reason = "P1 failed every confirmation canary while P0 passed every confirmation canary"
elif p0_fail == attempts and p1_fail == attempts:
    decision = "full_restart_warmup"
    reason = "both Prefill+Decode canary paths failed every confirmation attempt"

summary = {
    "schema_version": "1.0",
    "gate": "prefill_decode_unavailability_confirmation",
    "pass": decision != "uncertain_no_restart",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "attempts": attempts,
    "interval_sec": interval_sec,
    "decision": decision,
    "reason": reason,
    "counts": {
        "P0_pass": p0_pass,
        "P0_fail": p0_fail,
        "P1_pass": p1_pass,
        "P1_fail": p1_fail,
    },
    "attempt_acceptance_paths": [str(p) for p in paths],
    "attempts_detail": rows,
}
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(str(summary_path))
PY
}

recover_once() {
  assert_safety
  local ts run_id accept_path p0_pass p1_pass fail_count recover_log post_run post_accept confirmation_path decision
  ts="$(date -u +%Y%m%d_%H%M%S)"
  run_id="prefill_decode_availability_${ts}"
  set +e
  run_canary "${run_id}" | tee "${LOG_DIR}/${run_id}.log"
  local canary_rc=${PIPESTATUS[0]}
  set -e
  accept_path="$(latest_acceptance_for "${run_id}")"
  if [[ "${canary_rc}" == "0" ]]; then
    json_event "availability_check" "pass" "P0/P1 Prefill+Decode canary passed" "{\"acceptance\":\"${accept_path}\"}"
    return 0
  fi
  p0_pass="$(jq -r '.checks[] | select(.group=="P0") | .pass' "${accept_path}")"
  p1_pass="$(jq -r '.checks[] | select(.group=="P1") | .pass' "${accept_path}")"
  fail_count="$(jq -r '.failed_groups | length' "${accept_path}")"
  json_event "availability_check" "fail" "failed_groups=${fail_count}" "{\"acceptance\":\"${accept_path}\"}"

  confirmation_path="$(confirm_unavailability "${run_id}")"
  decision="$(jq -r '.decision' "${confirmation_path}")"
  json_event \
    "availability_confirmation" \
    "$([[ "${decision}" == "uncertain_no_restart" ]] && echo warn || echo fail)" \
    "decision=${decision}" \
    "{\"confirmation\":\"${confirmation_path}\"}"
  case "${decision}" in
    recover_P0_only)
      p0_pass=false
      p1_pass=true
      ;;
    recover_P1_only)
      p0_pass=true
      p1_pass=false
      ;;
    full_restart_warmup)
      p0_pass=false
      p1_pass=false
      ;;
    *)
      json_event "availability_recover_skip" "warn" "unavailability was not confirmed; no restart" "{\"confirmation\":\"${confirmation_path}\"}"
      return 0
      ;;
  esac

  (
    flock -n 9 || {
      json_event "availability_recover_skip" "locked" "another recovery is running" "{\"acceptance\":\"${accept_path}\"}"
      exit 1
    }
    if [[ "${p0_pass}" == "false" && "${p1_pass}" == "true" ]]; then
      recover_log="${LOG_DIR}/${run_id}_recover_P0.log"
      json_event "single_prefill_recover_start" "running" "P0 failed; P1+Decode still usable" "{\"recover_log\":\"${recover_log}\"}"
      if "${WRAPPER}" prefill-recover P0 "availability_${run_id}" > "${recover_log}" 2>&1; then
        post_run="${run_id}_post_P0_recover"
        run_canary "${post_run}" > "${LOG_DIR}/${post_run}.log" 2>&1 && {
          post_accept="$(latest_acceptance_for "${post_run}")"
          json_event "single_prefill_recover_complete" "pass" "P0 recovered" "{\"recover_log\":\"${recover_log}\",\"post_acceptance\":\"${post_accept}\"}"
          exit 0
        }
      fi
      json_event "single_prefill_recover_complete" "fail" "P0 single recover failed; escalate to full restart" "{\"recover_log\":\"${recover_log}\"}"
    elif [[ "${p1_pass}" == "false" && "${p0_pass}" == "true" ]]; then
      recover_log="${LOG_DIR}/${run_id}_recover_P1.log"
      json_event "single_prefill_recover_start" "running" "P1 failed; P0+Decode still usable" "{\"recover_log\":\"${recover_log}\"}"
      if "${WRAPPER}" prefill-recover P1 "availability_${run_id}" > "${recover_log}" 2>&1; then
        post_run="${run_id}_post_P1_recover"
        run_canary "${post_run}" > "${LOG_DIR}/${post_run}.log" 2>&1 && {
          post_accept="$(latest_acceptance_for "${post_run}")"
          json_event "single_prefill_recover_complete" "pass" "P1 recovered" "{\"recover_log\":\"${recover_log}\",\"post_acceptance\":\"${post_accept}\"}"
          exit 0
        }
      fi
      json_event "single_prefill_recover_complete" "fail" "P1 single recover failed; escalate to full restart" "{\"recover_log\":\"${recover_log}\"}"
    else
      json_event "full_recover_start" "running" "both Prefill canaries failed or Decode path unavailable" "{\"acceptance\":\"${accept_path}\"}"
    fi

    recover_log="${LOG_DIR}/${run_id}_full_restart_warmup.log"
    if "${WRAPPER}" full-restart-warmup "availability_${run_id}" > "${recover_log}" 2>&1; then
      post_run="${run_id}_post_full_recover"
      if run_canary "${post_run}" > "${LOG_DIR}/${post_run}.log" 2>&1; then
        post_accept="$(latest_acceptance_for "${post_run}")"
        json_event "full_recover_complete" "pass" "full restart and warmup completed" "{\"recover_log\":\"${recover_log}\",\"post_acceptance\":\"${post_accept}\"}"
        exit 0
      fi
      post_accept="$(latest_acceptance_for "${post_run}")"
      json_event "full_recover_complete" "fail" "full restart completed but post-canary failed" "{\"recover_log\":\"${recover_log}\",\"post_acceptance\":\"${post_accept}\"}"
      exit 1
    fi
    json_event "full_recover_complete" "fail" "full restart or warmup failed" "{\"recover_log\":\"${recover_log}\"}"
    exit 1
  ) 9>"${LOCK_FILE}"
}

loop() {
  assert_safety
  echo "$$" > "${PID_FILE}"
  json_event "availability_loop_start" "running" "interval=${INTERVAL_SEC}" "{\"pid\":$$}"
  while true; do
    recover_once || true
    sleep "${INTERVAL_SEC}"
  done
}

start() {
  assert_safety
  if [[ -f "${PID_FILE}" ]] && ps -p "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "already_running_pid=$(cat "${PID_FILE}")"
    return 0
  fi
  local ts log_path
  ts="$(date -u +%Y%m%d_%H%M%S)"
  log_path="${LOG_DIR}/prefill_decode_availability_loop_${ts}.log"
  nohup bash "$0" loop > "${log_path}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "started_pid=${pid}"
  echo "loop_log=${log_path}"
}

stop() {
  if [[ -f "${PID_FILE}" ]]; then
    kill "$(cat "${PID_FILE}")" 2>/dev/null || true
    rm -f "${PID_FILE}"
  fi
  echo "stopped"
}

status() {
  assert_safety
  if [[ -f "${PID_FILE}" ]]; then
    echo "pid=$(cat "${PID_FILE}")"
    ps -p "$(cat "${PID_FILE}")" -o pid=,etime=,stat=,cmd= || true
  else
    echo "pid_file_missing"
  fi
  echo "interval_sec=${INTERVAL_SEC}"
  echo "confirm_attempts=${CONFIRM_ATTEMPTS}"
  echo "confirm_interval_sec=${CONFIRM_INTERVAL_SEC}"
  echo "events=${EVENTS_FILE}"
  tail -20 "${EVENTS_FILE}" 2>/dev/null || true
}

cmd="${1:-status}"
case "${cmd}" in
  once)
    recover_once
    ;;
  loop)
    loop
    ;;
  start)
    start
    ;;
  stop)
    stop
    ;;
  status)
    status
    ;;
  *)
    echo "usage: $0 once|loop|start|stop|status" >&2
    exit 2
    ;;
esac
