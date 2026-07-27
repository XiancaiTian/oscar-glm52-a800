#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${PHASE5_RUN_ID:?set PHASE5_RUN_ID to the running OSCAR server run ID}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
RUN_DIR="${ARTIFACT_ROOT}/phase5/${RUN_ID}"
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

"${PYTHON_BIN}" - "${RUN_DIR}/server.log" \
  >> "${RUN_DIR}/oscar_runtime_evidence.log" <<'PY'
import pathlib
import re
import sys

lines = pathlib.Path(sys.argv[1]).read_text(errors="replace").splitlines()
pool_lines = [line for line in lines if "OSCAR MLA pools:" in line]
if not pool_lines:
    raise SystemExit("missing OSCAR MLA pool observability")
pool_line = pool_lines[-1]
pool_match = re.search(
    r"logical capacity=(\d+) tokens.*BF16 history=absent, "
    r"compression ratios theoretical=([0-9.]+)x "
    r"padded=([0-9.]+)x allocated=([0-9.]+)x",
    pool_line,
)
if pool_match is None:
    raise SystemExit("incomplete OSCAR MLA pool observability")
capacity, theoretical, padded, allocated = (
    int(pool_match.group(1)),
    float(pool_match.group(2)),
    float(pool_match.group(3)),
    float(pool_match.group(4)),
)
if capacity < 32768:
    raise SystemExit(f"OSCAR logical capacity is below 32K: {capacity}")
if abs(theoretical - 6.4) > 1e-9 or abs(padded - 6.4) > 1e-9:
    raise SystemExit(
        "unexpected latent compression ratios: "
        f"theoretical={theoretical}, padded={padded}"
    )
if allocated <= 1.0:
    raise SystemExit(f"OSCAR allocated capacity ratio is not a gain: {allocated}")

count_lines = [line for line in lines if "OSCAR MLA call counts:" in line]
if not count_lines:
    raise SystemExit("missing OSCAR MLA call counts")
count_line = count_lines[-1]
count_match = re.search(
    r"layers=(\d+), store=(\d+) \(min=(\d+) max=(\d+)\), "
    r"demotion=(\d+) \(min=(\d+) max=(\d+)\), "
    r"read=(\d+) \(min=(\d+) max=(\d+)\)",
    count_line,
)
if count_match is None:
    raise SystemExit("invalid OSCAR MLA call-count format")
values = tuple(int(value) for value in count_match.groups())
layers = values[0]
if layers != 78:
    raise SystemExit(f"expected 78 OSCAR MLA layers, got {layers}")
for name, (total, minimum, maximum) in zip(
    ("store", "demotion", "read"),
    (values[1:4], values[4:7], values[7:10]),
    strict=True,
):
    if minimum <= 0 or minimum != maximum or total != layers * minimum:
        raise SystemExit(
            f"inconsistent OSCAR {name} counts: "
            f"total={total}, min={minimum}, max={maximum}, layers={layers}"
        )

print(pool_line)
print(count_line)
PY

nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits \
  > "${RUN_DIR}/gpu_after_smoke.txt"
