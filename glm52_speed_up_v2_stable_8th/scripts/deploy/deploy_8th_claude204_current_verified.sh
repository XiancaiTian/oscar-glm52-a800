#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/load_deploy_config.sh"
load_deploy_config

MODE="${1:-precheck}"
RUN_ID="${RUN_ID:-claude204_${DEPLOY_LABEL}_current_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${TASK_ROOT}/logs/deploy"
RAW_DIR="${TASK_ROOT}/reports/raw"
ACCEPT_DIR="${TASK_ROOT}/reports/acceptance"
CLAUDE204_LOG_DIR="${TASK_ROOT}/logs/claudecli_204"
export MODEL_ID OPENAI_BASE MESSAGES_ENDPOINT MESSAGES_COUNT_ENDPOINT

usage() {
  cat <<'EOF'
Usage:
  deploy_8th_claude204_current_verified.sh MODE

Modes:
  print-config       Print effective 8th ClaudeCLI-2.1.204 proxy compatibility config.
  precheck           Validate hosts, shape list, proxy patch, and scripts.
  proxy-restart      Restart only the Proxy; retain P0/P1/D0 vLLM model processes.
  prefill-restart    Restart one Prefill group only: P0 or P1.
  prefill-warmup     Run stage_90k shape warmup through one Prefill group: P0 or P1.
  prefill-recover    Mark one Prefill unavailable, restart it, run targeted warmup, then mark healthy.
  warmup             Run stage_90k shape warmup through both Prefill groups.
  full-restart-warmup
                     Full 2P1D restart followed by stage_90k shape warmup.
  availability-watchdog-start|stop|status|once
                     Manage the 30-minute real-inference availability watchdog.
  verify-http        Verify /v1/messages/count_tokens and /v1/messages system-role handling.
  verify             Run healthcheck and verify-http.
  run                precheck -> proxy-restart -> verify-http.
EOF
}

actual_hosts_csv() {
  printf '%s\n' \
    ${TARGET_HOSTS} \
    "${BUILD_HOST}" \
    "${PREFILL_HEAD}" "${PREFILL_WORKER}" \
    "${PREFILL0_HEAD}" "${PREFILL0_WORKER}" \
    "${PREFILL1_HEAD}" "${PREFILL1_WORKER}" \
    "${DECODE_HEAD}" "${DECODE_WORKER}" |
    sed '/^$/d' | sort -u | paste -sd, -
}

expected_hosts_csv() {
  printf '%s' "${EXPECTED_HOSTS_CSV}" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -
}

allowed_hosts_csv() {
  printf '%s' "${ALLOWED_HOSTS}" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -
}

shape_values_csv() {
  paste -sd, "${SHAPE_LIST_FILE}"
}

assert_allowed_hosts() {
  local actual expected allowed
  actual="$(actual_hosts_csv)"
  expected="$(expected_hosts_csv)"
  allowed="$(allowed_hosts_csv)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR: deploy target hosts must match EXPECTED_HOSTS_CSV; expected=${expected} actual=${actual}" >&2
    exit 2
  fi
  if [[ "${allowed}" != "${expected}" ]]; then
    echo "ERROR: ALLOWED_HOSTS must match EXPECTED_HOSTS_CSV; expected=${expected} actual=${allowed}" >&2
    exit 2
  fi
  [[ -z "${FORBIDDEN_HOSTS_CSV}" ]] && return 0
  IFS=',' read -r -a forbidden_hosts <<< "${FORBIDDEN_HOSTS_CSV}"
  for host in "${forbidden_hosts[@]}"; do
    [[ -n "${host}" ]] || continue
    if grep -Fq "${host}" <<< "${actual},${ALLOWED_HOSTS},${PROXY_ROOT},${OPENAI_BASE}"; then
      echo "ERROR: forbidden host appears in active ${DEPLOY_LABEL} Claude204 config: ${host}" >&2
      exit 2
    fi
  done
}

assert_shape_list() {
  if [[ ! -f "${SHAPE_LIST_FILE}" ]]; then
    echo "ERROR: missing shape list file: ${SHAPE_LIST_FILE}" >&2
    exit 3
  fi
  local actual_sha actual_values
  actual_sha="$(sha256sum "${SHAPE_LIST_FILE}" | awk '{print $1}')"
  actual_values="$(shape_values_csv)"
  if [[ "${actual_values}" != "${EXPECTED_STAGE_90K_SHAPES}" ]]; then
    echo "ERROR: shape list mismatch: expected=${EXPECTED_STAGE_90K_SHAPES} actual=${actual_values}" >&2
    exit 3
  fi
  if [[ "${actual_sha}" != "${EXPECTED_STAGE_90K_SHAPES_SHA256}" ]]; then
    echo "ERROR: shape checksum mismatch: expected=${EXPECTED_STAGE_90K_SHAPES_SHA256} actual=${actual_sha}" >&2
    exit 3
  fi
}

assert_proxy_claude204_patch() {
  if [[ ! -f "${PROXY_SOURCE}" ]]; then
    echo "ERROR: missing proxy source: ${PROXY_SOURCE}" >&2
    exit 4
  fi
  grep -Fq "def _normalize_anthropic_system_messages" "${PROXY_SOURCE}"
  grep -Fq "PROXY_ANTHROPIC_SYSTEM_NORMALIZED" "${PROXY_SOURCE}"
  grep -Fq "PROXY_PREFILL_INTERNAL_WARMUP_ALLOWED" "${PROXY_SOURCE}"
  grep -Fq 'endpoint == "/messages/count_tokens"' "${PROXY_SOURCE}"
  grep -Fq 'api == "/messages"' "${PROXY_SOURCE}"
  python3 -m py_compile "${PROXY_SOURCE}"
}

precheck() {
  assert_allowed_hosts
  assert_shape_list
  assert_proxy_claude204_patch
  assert_runtime_patch_source
  bash -n "${SCRIPT_DIR}/deploy_8th_current_verified.sh"
  bash -n "${SCRIPT_DIR}/deploy_stage_90k_v8_prefix_cache.sh"
  bash -n "${SCRIPT_DIR}/deploy_stage_90k_v2.sh"
  "${SCRIPT_DIR}/deploy_8th_current_verified.sh" --print-config >/tmp/deploy_8th_current_verified.$$.config
  grep -Fq "target_hosts=$(expected_hosts_csv)" /tmp/deploy_8th_current_verified.$$.config
  grep -Fq "context_length_guard=1" /tmp/deploy_8th_current_verified.$$.config
  rm -f /tmp/deploy_8th_current_verified.$$.config
  echo "precheck_pass=1"
}

print_config() {
  cat <<EOF
run_id=${RUN_ID}
task_root=${TASK_ROOT}
deploy_config_file=${DEPLOY_CONFIG_FILE}
deploy_label=${DEPLOY_LABEL}
scheme=${SCHEME}
model_id=${MODEL_ID}
model_path=${MODEL_PATH}
image_tag=${IMAGE_TAG}
image_tar=${IMAGE_TAR}
runtime_patch_source_dir=${RUNTIME_PATCH_SOURCE_DIR}
proxy_root=${PROXY_ROOT}
openai_base=${OPENAI_BASE}
messages_endpoint=${MESSAGES_ENDPOINT}
messages_count_endpoint=${MESSAGES_COUNT_ENDPOINT}
target_hosts=$(actual_hosts_csv)
allowed_hosts=$(allowed_hosts_csv)
forbidden_hosts=${FORBIDDEN_HOSTS_CSV}
prefill0=${PREFILL0_HEAD},${PREFILL0_WORKER}
prefill1=${PREFILL1_HEAD},${PREFILL1_WORKER}
decode=${DECODE_HEAD},${DECODE_WORKER}
iface=${IFACE}
host_iface_overrides=${HOST_IFACE_OVERRIDES}
nccl_ib_hca=${NCCL_IB_HCA}
ucx_tls=${UCX_TLS}
ucx_devices=${UCX_DEVICES}
ucx_net_devices=${UCX_NET_DEVICES}
containers=${PREFILL_HEAD_CONTAINER},${PREFILL_WORKER_CONTAINER},${PREFILL1_HEAD_CONTAINER},${PREFILL1_WORKER_CONTAINER},${DECODE_HEAD_CONTAINER},${DECODE_WORKER_CONTAINER}
shape_values=$(shape_values_csv)
shape_list=${SHAPE_LIST_FILE}
shape_list_sha256=${EXPECTED_STAGE_90K_SHAPES_SHA256}
context_length_guard=${PREFILL_PROXY_CONTEXT_LENGTH_GUARD}
proxy_max_model_len=${PREFILL_PROXY_MAX_MODEL_LEN}
claude204_proxy_compat=enabled
responses_api_verification=skipped_by_instruction
proxy_source=${PROXY_SOURCE}
proxy_patch_marker=PROXY_ANTHROPIC_SYSTEM_NORMALIZED
prefill_internal_token_set=1
EOF
}

health() {
  curl -fsS --max-time 15 "${PROXY_ROOT}/healthcheck" >/dev/null
  curl -fsS --max-time 15 "${OPENAI_BASE}/models" >/dev/null
}

latest_proxy_log() {
  ls -1t "${LOG_DIR}/${SCHEME}"/*proxy*.log 2>/dev/null | head -n 1 || true
}

marker_count() {
  local log_path="$1"
  if [[ -n "${log_path}" && -f "${log_path}" ]]; then
    rg -c "PROXY_ANTHROPIC_SYSTEM_NORMALIZED" "${log_path}" 2>/dev/null || true
  else
    printf '0\n'
  fi
}

proxy_restart() {
  precheck
  mkdir -p "${LOG_DIR}" "${RAW_DIR}" "${ACCEPT_DIR}"
  DEPLOY_ACTION=proxy_restart "${SCRIPT_DIR}/deploy_stage_90k_v2.sh"
}

prefill_group_target() {
  case "${1:-}" in
    P0|p0|0)
      printf '%s\n' "P0 ${PREFILL0_HEAD}:${PREFILL_PORT} restart_prefill0_only"
      ;;
    P1|p1|1)
      printf '%s\n' "P1 ${PREFILL1_HEAD}:${PREFILL_PORT} restart_prefill1_only"
      ;;
    *)
      echo "ERROR: expected Prefill group P0 or P1; got '${1:-}'" >&2
      return 2
      ;;
  esac
}

set_prefill_status() {
  local group="$1"
  local status="$2"
  local code="$3"
  local reason="$4"
  python3 - \
    "${TASK_ROOT}/state/watchdog/prefill_status.json" \
    "${group}" "${status}" "${code}" "${reason}" \
    "${PREFILL0_HEAD}:${PREFILL_PORT}" "${PREFILL0_HEAD}" \
    "${PREFILL1_HEAD}:${PREFILL_PORT}" "${PREFILL1_HEAD}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
group, status, code, reason = sys.argv[2:6]
p0_endpoint, p0_host, p1_endpoint, p1_host = sys.argv[6:10]
now = datetime.now(timezone.utc).isoformat()
prefill_map = {
    "P0": [p0_endpoint, p0_host],
    "P1": [p1_endpoint, p1_host],
    "all": [p0_endpoint, p0_host, p1_endpoint, p1_host],
}
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    data = {}
prefills = data.setdefault("prefills", {})
for key in prefill_map[group]:
    prefills[key] = {
        "status": status,
        "error_code": code,
        "reason": reason,
        "updated_at": now,
        "source": "deploy_8th_claude204_current_verified",
    }
data["updated_at"] = now
data["source"] = "deploy_8th_claude204_current_verified"
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

recovery_cache_env() {
  local label="${1:-recovery}"
  local ts
  ts="$(date -u +%Y%m%d_%H%M%S)"
  export V2_PATCH_CACHE_SUFFIX="shape_pad_v2_runtime_patch_20260702b_${DEPLOY_LABEL}_recovery_${label}_${ts}"
  export V2_XDG_CACHE_HOME_OVERRIDE="${TASK_ROOT}/cache/xdg_stage_90k_enabled_v${DEPLOY_GENERATION}_stage62_${V2_PATCH_CACHE_SUFFIX}"
  mkdir -p \
    "${V2_XDG_CACHE_HOME_OVERRIDE}/torch/kernels" \
    "${V2_XDG_CACHE_HOME_OVERRIDE}/vllm/torch_compile_cache/torch_aot_compile"
  chmod 777 \
    "${V2_XDG_CACHE_HOME_OVERRIDE}" \
    "${V2_XDG_CACHE_HOME_OVERRIDE}/torch" \
    "${V2_XDG_CACHE_HOME_OVERRIDE}/torch/kernels" \
    "${V2_XDG_CACHE_HOME_OVERRIDE}/vllm" \
    "${V2_XDG_CACHE_HOME_OVERRIDE}/vllm/torch_compile_cache" \
    "${V2_XDG_CACHE_HOME_OVERRIDE}/vllm/torch_compile_cache/torch_aot_compile" \
    2>/dev/null || true
  echo "recovery_cache_suffix=${V2_PATCH_CACHE_SUFFIX}"
  echo "recovery_xdg_cache_home=${V2_XDG_CACHE_HOME_OVERRIDE}"
}

warmup_stage90k() {
  local group="${1:-all}"
  local target_prefill=""
  local run_group="${group}"
  if [[ "${group}" != "all" ]]; then
    read -r run_group target_prefill _ < <(prefill_group_target "${group}")
  fi
  precheck
  mkdir -p "${LOG_DIR}" "${RAW_DIR}" "${ACCEPT_DIR}"
  local ts run_id log_path
  ts="$(date -u +%Y%m%d_%H%M%S)"
  run_id="${RUN_ID}_${run_group}_stage90k_warmup_${ts}"
  log_path="${LOG_DIR}/${run_id}.log"
  (
    cd "${TASK_ROOT}"
    export RUN_ID="${run_id}"
    export STAGE_NAME="stage_90k"
    export V2_STAGE_NAME="stage_90k"
    export PREFIX_CACHE_MODE="enabled"
    export ALLOW_PREFIX_CACHE_ENABLED_FINAL="1"
    export REQUIRED_PER_PREFILL="${REQUIRED_PER_PREFILL:-3}"
    export MAX_ATTEMPTS_PER_LENGTH="${MAX_ATTEMPTS_PER_LENGTH:-18}"
    export MAX_TOKENS="1"
    export MODEL="${MODEL_ID}"
    export MESSAGES_ENDPOINT="${MESSAGES_ENDPOINT}"
    export MESSAGES_COUNT_ENDPOINT="${MESSAGES_COUNT_ENDPOINT}"
    export PREFILL_PROXY_INTERNAL_TOKEN="${PREFILL_PROXY_INTERNAL_TOKEN}"
    if [[ -n "${target_prefill}" ]]; then
      export TARGET_PREFILL="${target_prefill}"
      export PREFILLS="${target_prefill}"
    else
      export PREFILLS="${PREFILL0_HEAD}:${PREFILL_PORT},${PREFILL1_HEAD}:${PREFILL_PORT}"
    fi
    export DECODE="${DECODE_HEAD}:${DECODE_PORT}"
    python3 "${TASK_ROOT}/scripts/api/messages_stage_shape_warmup_v2.py"
  ) 2>&1 | tee "${log_path}"
}

prefill_restart() {
  local group target action
  read -r group target action < <(prefill_group_target "${1:-}")
  precheck
  recovery_cache_env "${group}_restart"
  DEPLOY_ACTION="${action}" "${SCRIPT_DIR}/deploy_stage_90k_v2.sh"
}

prefill_recover() {
  local group target action reason
  read -r group target action < <(prefill_group_target "${1:-}")
  reason="${2:-manual_prefill_recover_${group}}"
  precheck
  set_prefill_status "${group}" "restarting" "prefill_node_restarting" "${reason}: restarting ${target}"
  recovery_cache_env "${group}_recover"
  if DEPLOY_ACTION="${action}" "${SCRIPT_DIR}/deploy_stage_90k_v2.sh"; then
    # The proxy gates warming/unavailable Prefill paths. Re-open the restarted
    # path after /v1/models readiness so shape warmup can exercise it.
    set_prefill_status "${group}" "healthy" "" "${reason}: restart passed; warmup gate opened for ${target}"
    if warmup_stage90k "${group}"; then
      set_prefill_status "${group}" "healthy" "" "${reason}: single Prefill recover and warmup pass"
      health
      return 0
    fi
  fi
  set_prefill_status "${group}" "unavailable" "current_prefill_node_unavailable" "${reason}: single Prefill recover failed"
  return 1
}

full_restart_warmup() {
  local reason="${1:-manual_full_restart_warmup}"
  precheck
  set_prefill_status all restarting prefill_node_restarting "${reason}: full restart"
  recovery_cache_env "full_restart"
  if DEPLOY_ACTION=restart "${SCRIPT_DIR}/deploy_stage_90k_v2.sh"; then
    # The proxy treats warming as unavailable. Once deploy_stage_90k_v2.sh has
    # confirmed model readiness, mark both Prefill paths healthy before warmup.
    set_prefill_status all healthy "" "${reason}: full restart passed; warmup gate opened"
    if warmup_stage90k all; then
      set_prefill_status all healthy "" "${reason}: full restart and warmup pass"
      health
      return 0
    fi
  fi
  set_prefill_status all unavailable current_prefill_node_unavailable "${reason}: full restart or warmup failed"
  return 1
}

availability_watchdog() {
  local subcmd="${1:-status}"
  shift || true
  "${TASK_ROOT}/scripts/watchdog/prefill_decode_availability_watchdog.sh" "${subcmd}" "$@"
}

verify_http() {
  precheck
  mkdir -p "${RAW_DIR}" "${ACCEPT_DIR}" "${CLAUDE204_LOG_DIR}"
  local before_log before_count raw_dir accept_path
  before_log="$(latest_proxy_log)"
  before_count="$(marker_count "${before_log}")"
  raw_dir="${RAW_DIR}/${RUN_ID}_claude204_http"
  accept_path="${ACCEPT_DIR}/${RUN_ID}_claude204_http_acceptance.json"
  mkdir -p "${raw_dir}"
  python3 - \
    "${RUN_ID}" \
    "${MODEL_ID}" \
    "${MESSAGES_COUNT_ENDPOINT}" \
    "${MESSAGES_ENDPOINT}" \
    "${raw_dir}" \
    "${accept_path}" \
    "${LOG_DIR}/${SCHEME}" \
    "${before_count}" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

run_id, model, count_url, messages_url, raw_dir, accept_path, proxy_log_dir, before_count = sys.argv[1:9]
raw_dir = Path(raw_dir)
accept_path = Path(accept_path)
proxy_log_dir = Path(proxy_log_dir)
before_count = int(before_count or "0")
payload = {
    "model": model,
    "max_tokens": 8,
    "messages": [
        {"role": "system", "content": "When asked for the marker, answer OK."},
        {"role": "user", "content": "Return the marker."},
    ],
}

def post_json(url: str, body: dict) -> tuple[int, str, dict | None]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-request-id": run_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            text = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        status = exc.code
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    return status, text, parsed

count_status, count_text, count_json = post_json(count_url, payload)
(raw_dir / "system_role_count_tokens_response.json").write_text(
    count_text + "\n", encoding="utf-8"
)
messages_status, messages_text, messages_json = post_json(messages_url, payload)
(raw_dir / "system_role_messages_response.json").write_text(
    messages_text + "\n", encoding="utf-8"
)
(raw_dir / "system_role_request.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

after_count = 0
proxy_log = None
text = ""
proxy_tail_path = raw_dir / "proxy_marker_tail.log"
deadline = time.time() + 60
while time.time() < deadline:
    proxy_logs = sorted(
        proxy_log_dir.glob("*proxy*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    proxy_log = proxy_logs[0] if proxy_logs else None
    if proxy_log:
        text = proxy_log.read_text(encoding="utf-8", errors="replace")
        after_count = text.count("PROXY_ANTHROPIC_SYSTEM_NORMALIZED")
        if after_count > before_count:
            break
    time.sleep(0.5)
marker_lines = [
    line for line in text.splitlines()
    if "PROXY_ANTHROPIC_SYSTEM_NORMALIZED" in line
][-20:]
proxy_tail_path.write_text("\n".join(marker_lines) + ("\n" if marker_lines else ""), encoding="utf-8")

errors = []
if count_status != 200:
    errors.append(f"count_tokens_status={count_status}")
if messages_status != 200:
    errors.append(f"messages_status={messages_status}")
if after_count <= before_count:
    errors.append(
        f"proxy_normalization_marker_not_increased before={before_count} after={after_count}"
    )
if "context_length_precheck_failed" in count_text + messages_text:
    errors.append("context_length_precheck_failed_seen")

payload_out = {
    "schema_version": "1.0",
    "gate": "claudecli_204_http_compat",
    "pass": not errors,
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "run_id": run_id,
    "model": model,
    "anthropic_base_url": messages_url.rsplit("/v1/messages", 1)[0],
    "system_role_count_tokens_status": count_status,
    "system_role_messages_status": messages_status,
    "proxy_system_normalized_marker_present": after_count > before_count,
    "proxy_marker_count_before": before_count,
    "proxy_marker_count_after": after_count,
    "context_length_precheck_false_positive": "context_length_precheck_failed" in count_text + messages_text,
    "responses_api_validation": "not_executed",
    "raw_evidence_paths": [
        str(raw_dir / "system_role_request.json"),
        str(raw_dir / "system_role_count_tokens_response.json"),
        str(raw_dir / "system_role_messages_response.json"),
        str(proxy_tail_path),
    ],
    "proxy_log_path": str(proxy_log) if proxy_log else None,
    "errors": errors,
}
accept_path.write_text(
    json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload_out, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit(1)
PY
}

verify() {
  health
  verify_http
}

case "${MODE}" in
  print-config|--print-config)
    assert_allowed_hosts
    assert_shape_list
    assert_proxy_claude204_patch
    assert_runtime_patch_source
    print_config
    ;;
  precheck)
    precheck
    ;;
  proxy-restart|proxy_restart)
    proxy_restart
    ;;
  prefill-restart|prefill_restart)
    prefill_restart "${2:-}"
    ;;
  prefill-warmup|prefill_warmup)
    warmup_stage90k "${2:-all}"
    ;;
  prefill-recover|prefill_recover)
    prefill_recover "${2:-}" "${3:-manual_prefill_recover}"
    ;;
  warmup)
    warmup_stage90k all
    ;;
  full-restart-warmup|full_restart_warmup)
    full_restart_warmup "${2:-manual_full_restart_warmup}"
    ;;
  availability-watchdog-start)
    availability_watchdog start
    ;;
  availability-watchdog-stop)
    availability_watchdog stop
    ;;
  availability-watchdog-status)
    availability_watchdog status
    ;;
  availability-watchdog-once)
    availability_watchdog once
    ;;
  verify-http)
    verify_http
    ;;
  verify)
    verify
    ;;
  run)
    precheck
    proxy_restart
    verify_http
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
