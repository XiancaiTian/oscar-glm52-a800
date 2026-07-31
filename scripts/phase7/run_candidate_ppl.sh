#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAGE7_PPL_RUN_ID="${STAGE7_PPL_RUN_ID:?set STAGE7_PPL_RUN_ID for the PPL run}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
[[ "${ARTIFACT_ROOT}" == /* ]] || {
  echo "ERROR: ARTIFACT_ROOT must be absolute" >&2
  exit 1
}
ARTIFACT_ROOT="$(realpath -m -- "${ARTIFACT_ROOT}")"
[[ "${ARTIFACT_ROOT}" == "${PROJECT_ROOT}/artifacts" ||
  "${ARTIFACT_ROOT}" == "${PROJECT_ROOT}/artifacts/"* ||
  "${ARTIFACT_ROOT}" == /dev/shm ||
  "${ARTIFACT_ROOT}" == /dev/shm/* ]] || {
  echo "ERROR: ARTIFACT_ROOT must be under project artifacts/ or /dev/shm/" >&2
  exit 1
}
[[ "${STAGE7_PPL_RUN_ID}" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]] || {
  echo "ERROR: STAGE7_PPL_RUN_ID contains unsafe characters" >&2
  exit 1
}
RUN_DIR="${ARTIFACT_ROOT}/phase7/${STAGE7_PPL_RUN_ID}"
BASE_ROOTFS="${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs"
OVERLAY_ROOTFS="${PROJECT_ROOT}/artifacts/phase6/20260731T0125Z_candidate_14c768b40_decode_metadata_v1/overlay_rootfs"
SOURCE_DIR="${OVERLAY_ROOTFS}/opt/vllm_glm52_v1"
BASE_SOURCE_DIR="${BASE_ROOTFS}/opt/vllm_glm52_v1"
VENV_DIR="${BASE_ROOTFS}/opt/fp8_speed_up_v4_venv"
PYTHON_BIN="${BASE_ROOTFS}/usr/bin/python3.12"
VENV_SITE_PACKAGES="${VENV_DIR}/lib/python3.12/site-packages"
ROOTFS_LOCAL_SITE_PACKAGES="${BASE_ROOTFS}/usr/local/lib/python3.12/dist-packages"
ROOTFS_DIST_PACKAGES="${BASE_ROOTFS}/usr/lib/python3/dist-packages"
MODEL_PATH="/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001"
RUNTIME_PROJECT_ROOT="${PROJECT_ROOT}"
FROZEN_EVALUATOR_ROOT="${RUNTIME_PROJECT_ROOT}/artifacts/phase7/frozen_evaluator_v4_20260728"
SUITE_DIR="${FROZEN_EVALUATOR_ROOT}"
PPL_RUNNER="${FROZEN_EVALUATOR_ROOT}/run_vllm_perplexity_suite.py"
OUTPUT_DIR="${RUN_DIR}/wikitext2_ppl"
NATIVE_LIB="${BASE_ROOTFS}/opt/glm52_speed_up_v1_stable/artifacts/native_ext/stage50_sparse_mla_m1_splitmerge_final_ops.so"
ROTATION_ARTIFACT="${OVERLAY_ROOTFS}/opt/oscar_artifacts/rotation_fit_v2"
RUNTIME_EXPECTATION="${OVERLAY_ROOTFS}/opt/oscar_artifacts/oscar_runtime_expectation.json"
LOCK_FILE="${PROJECT_ROOT}/artifacts/phase7/candidate_port_18082.lock"
CACHE_ROOT="${CACHE_ROOT:-${ARTIFACT_ROOT}/phase7/cache}"
[[ "${CACHE_ROOT}" == /* ]] || {
  echo "ERROR: CACHE_ROOT must be absolute" >&2
  exit 1
}
CACHE_ROOT="$(realpath -m -- "${CACHE_ROOT}")"
[[ "${CACHE_ROOT}" == "${PROJECT_ROOT}/artifacts/"* ||
  "${CACHE_ROOT}" == /dev/shm/* ]] || {
  echo "ERROR: CACHE_ROOT must be under project artifacts/ or /dev/shm/" >&2
  exit 1
}

[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "ERROR: PPL output already exists: ${OUTPUT_DIR}" >&2
  exit 1
}

[[ "$(sha256sum "${SUITE_DIR}/identity.json" | awk '{print $1}')" == \
  "fba1421ed512dd8551195e0856db69b9fbed75b71cffeddddb223e8ffbeaee44" ]] || {
  echo "ERROR: frozen PPL identity changed" >&2
  exit 1
}
[[ "$(sha256sum "${SUITE_DIR}/manifest.jsonl" | awk '{print $1}')" == \
  "56e1caa23dd79555caec2fa8e8586bfc2a9622e26d7bf630764511ac5689423e" ]] || {
  echo "ERROR: frozen PPL manifest changed" >&2
  exit 1
}
[[ "$(sha256sum "${PPL_RUNNER}" | awk '{print $1}')" == \
  "eec6b1a4be99a068f80b2e9c0d684392f1bf1a669cae4fbc881581d6baa25668" ]] || {
  echo "ERROR: frozen PPL runner changed" >&2
  exit 1
}

mkdir -p "$(dirname "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
flock -n 9 || {
  echo "ERROR: another Stage 7 candidate process holds ${LOCK_FILE}" >&2
  exit 1
}
export STAGE7_CANDIDATE_LOCK_HELD=1
export OSCAR_RUNTIME_PROJECT_ROOT="${PROJECT_ROOT}"

FORMAL_RUN=1 \
  RUN_ID="${STAGE7_PPL_RUN_ID}" \
  ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  MANIFEST="${PROJECT_ROOT}/configs/phase7/oscar_evaluation.json" \
  VERIFY_SCRIPT="${SCRIPT_DIR}/verify_candidate_evaluation.py" \
  "${SCRIPT_DIR}/run_candidate_tp8.sh" formal-preflight

native_links=(
  vllm/_C.abi3.so
  vllm/_C_stable_libtorch.abi3.so
  vllm/_moe_C.abi3.so
  vllm/cumem_allocator.abi3.so
  vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
  vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
)
created_links=()
cleanup_links() {
  local path
  for path in "${created_links[@]}"; do
    if [[ -L "${path}" ]]; then
      unlink "${path}"
    fi
  done
}
trap cleanup_links EXIT
for relative in "${native_links[@]}"; do
  runtime_path="${SOURCE_DIR}/${relative}"
  base_path="${BASE_SOURCE_DIR}/${relative}"
  [[ ! -e "${runtime_path}" && ! -L "${runtime_path}" ]] || {
    echo "ERROR: candidate runtime link already exists: ${runtime_path}" >&2
    exit 1
  }
  ln -s "${base_path}" "${runtime_path}"
  created_links+=("${runtime_path}")
done

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLM52_CANDIDATE_ROOTFS="${BASE_ROOTFS}"
export PYTHONHOME="${BASE_ROOTFS}/usr"
export VIRTUAL_ENV="${VENV_DIR}"
export PYTHONPATH="${SOURCE_DIR}:${VENV_SITE_PACKAGES}:${ROOTFS_LOCAL_SITE_PACKAGES}:${ROOTFS_DIST_PACKAGES}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${XDG_CACHE_HOME}/hf"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_DISABLE_INDUCTOR_AUTOTUNE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export VLLM_SPARSE_INDEXER_MQA_LOGITS_BACKEND=cuda_v7
export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS=1
export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE=8192
export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES=2
export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS=4
export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES=2
export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR=1
export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS=1
export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE=1
export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND=persistent
export VLLM_SPARSE_INDEXER_DECODE_FP8_LUT=1
export VLLM_TOPK_ENV_CACHE=1
export VLLM_MQA_CUDA_V7_FUSED_TRITON=1
export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_M_MAX=4
export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_M=1
export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N=512
export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_NUM_WARPS=4
export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_NUM_STAGES=3
export VLLM_MQA_CUDA_V7_FUSED_TRITON_PREFILL_CANONICAL_M=512
export VLLM_SPARSE_MLA_FORCE_KV_SPLITS=1
export VLLM_SPARSE_MLA_FINAL_STATIC_BY_TOKENS=1
export VLLM_SPARSE_MLA_ASSUME_VALID_DYNAMIC=1
export VLLM_SPARSE_MLA_ASSUME_VALID_NOMASK=1
export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N=256
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL=1
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_UNSAFE_ENABLE=1
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_DECODE_ONLY=1
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_NUM_SPLITS=32
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_LIB="${NATIVE_LIB}"
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_DEBUG_LOGS=16
export VLLM_OSCAR_MLA_ROTATION_ARTIFACT="${ROTATION_ARTIFACT}"
export VLLM_OSCAR_MLA_RUNTIME_EXPECTATION="${RUNTIME_EXPECTATION}"

command=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_candidate_ppl_wrapper.py"
  --runner "${PPL_RUNNER}"
  --suite-dir "${SUITE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --model-path "${MODEL_PATH}"
)
printf '%q ' "${command[@]}" > "${OUTPUT_DIR}/runner_command.txt"
printf '\n' >> "${OUTPUT_DIR}/runner_command.txt"
env | LC_ALL=C sort | awk -F= '
  $1 == "CUDA_VISIBLE_DEVICES" ||
  $1 == "GLM52_CANDIDATE_ROOTFS" ||
  $1 == "HF_HOME" ||
  $1 == "HF_HUB_OFFLINE" ||
  $1 == "PYTHONHOME" ||
  $1 == "PYTHONPATH" ||
  $1 == "VIRTUAL_ENV" ||
  $1 == "XDG_CACHE_HOME" ||
  $1 ~ /^VLLM_/ {print}
' > "${OUTPUT_DIR}/runtime_environment.txt"
{
  printf 'frozen_evaluator_identity_sha256=%s\n' \
    "$(sha256sum "${SUITE_DIR}/identity.json" | awk '{print $1}')"
  printf 'frozen_evaluator_manifest_sha256=%s\n' \
    "$(sha256sum "${SUITE_DIR}/manifest.jsonl" | awk '{print $1}')"
  printf 'frozen_ppl_runner_sha256=%s\n' \
    "$(sha256sum "${PPL_RUNNER}" | awk '{print $1}')"
} >> "${OUTPUT_DIR}/runtime_environment.txt"

(
  cd "${SOURCE_DIR}"
  "${command[@]}"
) > "${OUTPUT_DIR}/runner.log" 2>&1 &
runner_pid=$!
started="$(date +%s)"
next_progress=600
while kill -0 "${runner_pid}" 2>/dev/null; do
  sleep 60
  elapsed="$(( $(date +%s) - started ))"
  if ((elapsed >= next_progress)); then
    {
      printf '%s runner_pid=%s elapsed_seconds=%s\n' \
        "$(date -u +%FT%TZ)" "${runner_pid}" "${elapsed}"
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
  echo "ERROR: WikiText-2 runner exited with status ${runner_status}" >&2
  exit "${runner_status}"
}

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${SUITE_DIR}" "${PPL_RUNNER}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
suite = Path(sys.argv[2])
runner = Path(sys.argv[3])
summary_path = output / "summary.json"
summary = json.loads(summary_path.read_text())
if summary["total"] != 1 or summary["scored"] != 1:
    raise SystemExit(f"invalid perplexity summary: {summary}")
result = summary["results"][0]
if result["evaluator_status"] != "scored":
    raise SystemExit(f"invalid evaluator status: {result['evaluator_status']}")
validation = {
    "status": "passed",
    "total": 1,
    "scored": 1,
    "perplexity": result["perplexity"],
    "mean_nll": result["mean_nll"],
    "evaluated_tokens": result["evaluated_tokens"],
    "windows": result["windows"],
    "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "frozen_evaluator_identity_sha256": hashlib.sha256(
        (suite / "identity.json").read_bytes()
    ).hexdigest(),
    "frozen_evaluator_manifest_sha256": hashlib.sha256(
        (suite / "manifest.jsonl").read_bytes()
    ).hexdigest(),
    "frozen_ppl_runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    "runner_command_sha256": hashlib.sha256(
        (output / "runner_command.txt").read_bytes()
    ).hexdigest(),
    "runtime_environment_sha256": hashlib.sha256(
        (output / "runtime_environment.txt").read_bytes()
    ).hexdigest(),
}
(output / "validation.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
PY
