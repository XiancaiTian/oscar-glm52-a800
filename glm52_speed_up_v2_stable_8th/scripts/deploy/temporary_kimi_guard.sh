#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/deploy/deploy_common_env.sh"

DURATION_SECONDS="${DURATION_SECONDS:-7200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"
GUARD_HOSTS="${GUARD_HOSTS:-${TARGET_HOSTS}}"
STOP_FILE="${STOP_FILE:-${ROOT}/state/temporary_kimi_guard.stop}"

deadline=$(( $(date +%s) + DURATION_SECONDS ))
rm -f "${STOP_FILE}"

while [[ "$(date +%s)" -lt "${deadline}" ]]; do
  [[ -f "${STOP_FILE}" ]] && exit 0
  for host in ${GUARD_HOSTS}; do
    remote "${host}" "SUDO_PASS=$(printf '%q' "${SSHPASS}") bash -s" <<'REMOTE' || true
set +e
pats='[w]atchdog_service_kimi|[c]heck_service_kimi|[r]un_kimi|[k]imi_k2_6|[k]imi_k2|[s]glang::scheduler'
printf '%s\n' "${SUDO_PASS}" | sudo -S -p '' pkill -KILL -f "${pats}" >/dev/null 2>&1 || true
for c in $(docker ps -aq --format '{{.ID}} {{.Names}} {{.Image}}' | awk 'tolower($0) ~ /kimi|sglang/ {print $1}'); do
  docker rm -f "${c}" >/dev/null 2>&1 || true
done
REMOTE
  done
  sleep "${SLEEP_SECONDS}"
done
