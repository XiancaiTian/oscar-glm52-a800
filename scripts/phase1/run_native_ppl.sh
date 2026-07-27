#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHASE1_RUN_ID="${PHASE1_RUN_ID:?set PHASE1_RUN_ID for the PPL run}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
RUN_DIR="${ARTIFACT_ROOT}/phase1/${PHASE1_RUN_ID}"
CANDIDATE_ROOTFS="${CANDIDATE_ROOTFS:-${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_ROOT}/glm52_oscar_vllm}"
VENV_DIR="${CANDIDATE_ROOTFS}/opt/fp8_speed_up_v4_venv"
PYTHON_BIN="${CANDIDATE_ROOTFS}/usr/bin/python3.12"
VENV_SITE_PACKAGES="${VENV_DIR}/lib/python3.12/site-packages"
ROOTFS_LOCAL_SITE_PACKAGES="${CANDIDATE_ROOTFS}/usr/local/lib/python3.12/dist-packages"
ROOTFS_DIST_PACKAGES="${CANDIDATE_ROOTFS}/usr/lib/python3/dist-packages"
CACHE_ROOT="${CACHE_ROOT:-${ARTIFACT_ROOT}/phase1/cache}"
MODEL_PATH="/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001"
EVAL_ROOT="/nfs/AE/txc/vllm_turbo_baseline_acc"
SUITE_DIR="${EVAL_ROOT}/accuracy_suites/model_agnostic_accuracy_official_v4"
PPL_RUNNER="${EVAL_ROOT}/tools/run_vllm_perplexity_suite.py"
OUTPUT_DIR="${RUN_DIR}/wikitext2_ppl"
NATIVE_LIB="${CANDIDATE_ROOTFS}/opt/glm52_speed_up_v1_stable/artifacts/native_ext/stage50_sparse_mla_m1_splitmerge_final_ops.so"

FORMAL_RUN=1 RUN_ID="${PHASE1_RUN_ID}" ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  CANDIDATE_ROOTFS="${CANDIDATE_ROOTFS}" SOURCE_DIR="${SOURCE_DIR}" \
  CACHE_ROOT="${CACHE_ROOT}" \
  "${SCRIPT_DIR}/run_native_baseline.sh" formal-preflight

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLM52_CANDIDATE_ROOTFS="${CANDIDATE_ROOTFS}"
export PYTHONHOME="${CANDIDATE_ROOTFS}/usr"
export VIRTUAL_ENV="${VENV_DIR}"
export PYTHONPATH="${SOURCE_DIR}:${VENV_SITE_PACKAGES}:${ROOTFS_LOCAL_SITE_PACKAGES}:${ROOTFS_DIST_PACKAGES}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}/hf"
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
export VLLM_SPARSE_MLA_WARMUP_NUM_TOKENS=2048,4096
export VLLM_SPARSE_MLA_ASSUME_VALID_DYNAMIC=1
export VLLM_SPARSE_MLA_ASSUME_VALID_NOMASK=1
export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N=256
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL=1
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_UNSAFE_ENABLE=1
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_DECODE_ONLY=1
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_NUM_SPLITS=32
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_LIB="${NATIVE_LIB}"
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_DEBUG_LOGS=16

command=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_native_ppl_wrapper.py"
  --runner "${PPL_RUNNER}"
  --suite-dir "${SUITE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --model-path "${MODEL_PATH}"
)
printf '%q ' "${command[@]}" > "${OUTPUT_DIR}/runner_command.txt"
printf '\n' >> "${OUTPUT_DIR}/runner_command.txt"
env | LC_ALL=C sort | awk -F= '
  $1 == "CUDA_VISIBLE_DEVICES" ||
  $1 == "FLASHINFER_DISABLE_VERSION_CHECK" ||
  $1 == "GLM52_CANDIDATE_ROOTFS" ||
  $1 == "HF_HOME" ||
  $1 == "HF_HUB_OFFLINE" ||
  $1 == "PYTHONHOME" ||
  $1 == "PYTHONPATH" ||
  $1 == "PYTHONDONTWRITEBYTECODE" ||
  $1 == "TRANSFORMERS_CACHE" ||
  $1 == "VIRTUAL_ENV" ||
  $1 == "XDG_CACHE_HOME" ||
  $1 ~ /^VLLM_/ {print}
' > "${OUTPUT_DIR}/runtime_environment.txt"

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

"${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
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
}
(output / "validation.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
PY
