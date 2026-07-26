#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${PHASE5_RUN_ID:?set PHASE5_RUN_ID to the running OSCAR server run ID}"
RUN_DIR="${PROJECT_ROOT}/artifacts/phase5/${RUN_ID}"
CANDIDATE_ROOTFS="${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs"
PYTHON_BIN="${CANDIDATE_ROOTFS}/usr/bin/python3.12"
VENV_DIR="${CANDIDATE_ROOTFS}/opt/fp8_speed_up_v4_venv"
PORT="${PORT:-18081}"
SERVER_PID="$(<"${RUN_DIR}/server.pid")"

kill -0 "${SERVER_PID}" 2>/dev/null || {
  echo "ERROR: OSCAR server PID ${SERVER_PID} is not running" >&2
  exit 1
}
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

export PYTHONHOME="${CANDIDATE_ROOTFS}/usr"
export VIRTUAL_ENV="${VENV_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/glm52_oscar_vllm:${VENV_DIR}/lib/python3.12/site-packages:${CANDIDATE_ROOTFS}/usr/local/lib/python3.12/dist-packages:${CANDIDATE_ROOTFS}/usr/lib/python3/dist-packages"
export HF_HUB_OFFLINE=1

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/phase1/run_native_smoke.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --output "${RUN_DIR}/oscar_single_smoke.json" \
  2>&1 | tee "${RUN_DIR}/oscar_single_smoke.log"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_oscar_multi_smoke.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --requests 8 \
  --output "${RUN_DIR}/oscar_multi_smoke.json" \
  2>&1 | tee "${RUN_DIR}/oscar_multi_smoke.log"

{
  grep -F "OSCAR MLA pools:" "${RUN_DIR}/server.log"
  grep -F "OSCAR MLA rotation artifact loaded:" "${RUN_DIR}/server.log"
  grep -F "OSCAR MLA three-pool write active; no full BF16 latent history" \
    "${RUN_DIR}/server.log"
  grep -F "OSCAR MLA recent-to-INT2 demotion active" "${RUN_DIR}/server.log"
  grep -F "OSCAR MLA DSA-selected mixed prefix/recent/INT2 read active" \
    "${RUN_DIR}/server.log"
} > "${RUN_DIR}/oscar_runtime_evidence.log"

nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits \
  > "${RUN_DIR}/gpu_after_smoke.txt"
