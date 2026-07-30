#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/configs/phase1/native_baseline.json}"
CANDIDATE_ROOTFS="${CANDIDATE_ROOTFS:-${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs}"
SOURCE_REPO="${PROJECT_ROOT}/glm52_oscar_vllm"
SOURCE_DIR="${SOURCE_DIR:-${SOURCE_REPO}}"
VERIFY_SCRIPT="${VERIFY_SCRIPT:-${SCRIPT_DIR}/verify_native_baseline.py}"
VENV_DIR="${CANDIDATE_ROOTFS}/opt/fp8_speed_up_v4_venv"
PYTHON_BIN="${CANDIDATE_ROOTFS}/usr/bin/python3.12"
VENV_SITE_PACKAGES="${VENV_DIR}/lib/python3.12/site-packages"
ROOTFS_LOCAL_SITE_PACKAGES="${CANDIDATE_ROOTFS}/usr/local/lib/python3.12/dist-packages"
ROOTFS_DIST_PACKAGES="${CANDIDATE_ROOTFS}/usr/lib/python3/dist-packages"
CANDIDATE_PYTHONPATH="${SOURCE_DIR}:${VENV_SITE_PACKAGES}:${ROOTFS_LOCAL_SITE_PACKAGES}:${ROOTFS_DIST_PACKAGES}"
MODEL_PATH="/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001"
SUITE_DIR="${SUITE_DIR:-/nfs/AE/txc/vllm_turbo_baseline_acc/accuracy_suites/model_agnostic_accuracy_official_v4}"
STATIC_SUITE_DIR="${STATIC_SUITE_DIR:-${SUITE_DIR}}"
EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL:-official_v4}"
EVALUATION_SCOPE="${EVALUATION_SCOPE:-full}"
EVALUATION_MANIFEST_SHA256="${EVALUATION_MANIFEST_SHA256:-4aec8ee85bee5eb73ce99c2009fcaedc79804bde1433f855fb77276ffccacfa5}"
NATIVE_LIB="${CANDIDATE_ROOTFS}/opt/glm52_speed_up_v1_stable/artifacts/native_ext/stage50_sparse_mla_m1_splitmerge_final_ops.so"
RUN_KIND="${RUN_KIND:-native_tp8}"
ARTIFACT_PHASE="${ARTIFACT_PHASE:-phase1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${RUN_KIND}}"
RUN_DIR="${ARTIFACT_ROOT}/${ARTIFACT_PHASE}/${RUN_ID}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
SERVICE_LABEL="${SERVICE_LABEL:-Native TP=8}"
EXPECTED_MAIN_BRANCH="${EXPECTED_MAIN_BRANCH:-feat/glm52-model-load}"
EXPECTED_SOURCE_BRANCH="${EXPECTED_SOURCE_BRANCH:-feat/glm52-oscar-integration}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:-a94b1f640fe504be3d741a1070e43f806eaad894}"
EXPECTED_KV_CACHE_DTYPE="${EXPECTED_KV_CACHE_DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DISABLE_ASYNC_SCHEDULING="${DISABLE_ASYNC_SCHEDULING:-0}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/artifacts/phase1/cache}"
RUNTIME_SOURCE_COMMIT="${RUNTIME_SOURCE_COMMIT:-fd3e0b3772e989cf0d0d73a3d19b252ab82e9cdd}"
CANDIDATE_MANIFEST_DIGEST="${CANDIDATE_MANIFEST_DIGEST:-sha256:2fdfbe865aecc01eee15a01fcce58bf7581244dbbc53cbe3ef0e0cce44bc489d}"
CANDIDATE_CONFIG_DIGEST="${CANDIDATE_CONFIG_DIGEST:-sha256:58a853ee730c263968dcc6b76401e85e4b510a140777c4ffc791747edd8ea42d}"
CANDIDATE_LAYER_DIGEST="${CANDIDATE_LAYER_DIGEST:-sha256:352d47f649171770e32edb7e1112e8a31f6a5aead0f6160a30ad3a8eec6659c3}"

export GLM52_CANDIDATE_ROOTFS="${CANDIDATE_ROOTFS}"
export PYTHONHOME="${CANDIDATE_ROOTFS}/usr"
export VIRTUAL_ENV="${VENV_DIR}"
export PYTHONPATH="${CANDIDATE_PYTHONPATH}"
export DISABLE_ASYNC_SCHEDULING
export EXPECTED_KV_CACHE_DTYPE
export MAX_MODEL_LEN

usage() {
  cat <<'EOF'
Usage:
  run_native_baseline.sh preflight
  run_native_baseline.sh dry-run
  run_native_baseline.sh check-gpus
  FORMAL_RUN=1 run_native_baseline.sh formal-preflight
  FORMAL_RUN=1 run_native_baseline.sh serve

The serve mode fails closed unless both repositories are clean and their
current feature-branch commits are present on their configured origin remotes.
The official_v5 isolated orchestrator may pass already verified remote commits;
the serve process still rechecks local branch, HEAD, and cleanliness offline.
EOF
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || {
    echo "ERROR: required file is missing: ${path}" >&2
    return 1
  }
}

run_static_preflight() {
  require_file "${MANIFEST}"
  require_file "${PYTHON_BIN}"
  require_file "${NATIVE_LIB}"
  require_file "${MODEL_PATH}/config.json"
  require_file "${SUITE_DIR}/manifest.jsonl"
  require_file "${STATIC_SUITE_DIR}/manifest.jsonl"

  mkdir -p "${RUN_DIR}"
  "${PYTHON_BIN}" "${VERIFY_SCRIPT}" \
    --manifest "${MANIFEST}" \
    --suite-dir "${STATIC_SUITE_DIR}" \
    --output "${RUN_DIR}/static_preflight.json"

  (
    cd "${SOURCE_DIR}"
    "${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

import tokenizers
import torch
import transformers
import triton
import vllm
import vllm._C

rootfs = Path(os.environ["GLM52_CANDIDATE_ROOTFS"]).resolve()
python = Path(sys.executable).resolve()
if python != rootfs / "usr/bin/python3.12":
    raise SystemExit(f"unexpected Python interpreter: {python}")
for optional_package in ("flash_attn", "triton_kernels"):
    if importlib.util.find_spec(optional_package) is not None:
        raise SystemExit(
            f"host system package leaked into candidate runtime: {optional_package}"
        )

print(json.dumps({
    "python": sys.version.split()[0],
    "python_executable": str(python),
    "python_prefix": sys.prefix,
    "torch": torch.__version__,
    "triton": triton.__version__,
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "flashinfer_python": importlib.metadata.version("flashinfer-python"),
    "flashinfer_jit_cache": importlib.metadata.version(
        "flashinfer-jit-cache"
    ),
    "flash_attn_spec": None,
    "triton_kernels_spec": None,
    "vllm_reported_version": vllm.__version__,
    "vllm_source": vllm.__file__,
    "vllm_C": vllm._C.__file__,
    "cuda_initialized": torch.cuda.is_initialized(),
}, sort_keys=True))
PY
  ) | tee "${RUN_DIR}/fixed_environment_import.json"
}

gpu_snapshot() {
  local label="$1"
  local output="${RUN_DIR}/gpu_${label}.txt"
  {
    date -u +%FT%TZ
    nvidia-smi
    nvidia-smi \
      --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits
  } > "${output}"

  mapfile -t rows < <(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  [[ "${#rows[@]}" -eq 8 ]] || {
    echo "ERROR: expected exactly 8 visible GPUs, got ${#rows[@]}" >&2
    return 1
  }
  local row index memory utilization
  for row in "${rows[@]}"; do
    IFS=, read -r index memory utilization <<<"${row}"
    index="${index//[[:space:]]/}"
    memory="${memory//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    [[ "${index}" =~ ^[0-7]$ && "${memory}" == "0" && "${utilization}" == "0" ]] || {
      echo "ERROR: GPU is not idle at ${label}: ${row}" >&2
      return 1
    }
  done
  if nvidia-smi \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "ERROR: a compute process is using a visible GPU at ${label}" >&2
    return 1
  fi
  echo "GPU check ${label}: 8/8 idle"
}

check_gpus_twice() {
  mkdir -p "${RUN_DIR}"
  gpu_snapshot first
  echo "Waiting 60 seconds before the required second GPU check."
  sleep 60
  gpu_snapshot second
}

published_commit() {
  local repo="$1"
  local label="$2"
  local expected_branch="$3"
  local branch head remote_url remote_head

  [[ -z "$(git -C "${repo}" status --porcelain --untracked-files=all)" ]] || {
    echo "ERROR: ${label} repository is not clean: ${repo}" >&2
    return 1
  }
  branch="$(git -C "${repo}" symbolic-ref --quiet --short HEAD)"
  [[ "${branch}" == "${expected_branch}" ]] || {
    echo "ERROR: ${label} branch is ${branch}; expected ${expected_branch}" >&2
    return 1
  }
  head="$(git -C "${repo}" rev-parse HEAD)"
  if [[ "${PREVERIFIED_PUBLISHED_COMMITS:-0}" == "1" ]]; then
    local preverified
    case "${label}" in
      main)
        preverified="${PREVERIFIED_MAIN_COMMIT:?missing PREVERIFIED_MAIN_COMMIT}"
        ;;
      source)
        preverified="${PREVERIFIED_SOURCE_COMMIT:?missing PREVERIFIED_SOURCE_COMMIT}"
        ;;
      *)
        echo "ERROR: unsupported preverified repository label: ${label}" >&2
        return 1
        ;;
    esac
    [[ "${head}" == "${preverified}" ]] || {
      echo "ERROR: ${label} HEAD ${head} differs from preverified ${preverified}" >&2
      return 1
    }
    printf '%s\n' "${head}"
    return
  fi
  remote_url="$(git -C "${repo}" remote get-url origin)"
  remote_head="$(
    git ls-remote --heads "${remote_url}" "refs/heads/${branch}" |
      awk 'NR == 1 {print $1}'
  )"
  [[ "${remote_head}" == "${head}" ]] || {
    echo "ERROR: ${label} HEAD ${head} is not published at origin/${branch}" >&2
    return 1
  }
  printf '%s\n' "${head}"
}

write_runtime_manifest() {
  local main_commit="$1"
  local source_commit="$2"
  local command_file="$3"
  local environment_file="$4"
  MAIN_COMMIT="${main_commit}" SOURCE_COMMIT="${source_commit}" \
    RUNTIME_SOURCE_COMMIT="${RUNTIME_SOURCE_COMMIT}" \
    CANDIDATE_MANIFEST_DIGEST="${CANDIDATE_MANIFEST_DIGEST}" \
    CANDIDATE_CONFIG_DIGEST="${CANDIDATE_CONFIG_DIGEST}" \
    CANDIDATE_LAYER_DIGEST="${CANDIDATE_LAYER_DIGEST}" \
    EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL}" \
    EVALUATION_SCOPE="${EVALUATION_SCOPE}" \
    EVALUATION_MANIFEST_SHA256="${EVALUATION_MANIFEST_SHA256}" \
    RUN_IDENTIFIER="${RUN_ID}" RUNTIME_MANIFEST="${RUN_DIR}/runtime_manifest.json" \
    COMMAND_FILE="${command_file}" ENVIRONMENT_FILE="${environment_file}" \
    "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
import time
from pathlib import Path

command_file = Path(os.environ["COMMAND_FILE"])
environment_file = Path(os.environ["ENVIRONMENT_FILE"])
payload = {
    "format_version": 1,
    "status": "starting",
    "run_id": os.environ["RUN_IDENTIFIER"],
    "started_at_unix": time.time(),
    "main_commit": os.environ["MAIN_COMMIT"],
    "source_repository_commit": os.environ["SOURCE_COMMIT"],
    "runtime_source_commit": os.environ["RUNTIME_SOURCE_COMMIT"],
    "candidate_manifest_digest": os.environ["CANDIDATE_MANIFEST_DIGEST"],
    "candidate_config_digest": os.environ["CANDIDATE_CONFIG_DIGEST"],
    "candidate_layer_digest": os.environ["CANDIDATE_LAYER_DIGEST"],
    "model_filename_size_mtime_ns_manifest_sha256": "83eefdf08de8f489bee6f1d5b1bf4d2f3452d42757b4df580e76a40c3646acaf",
    "evaluation_protocol": os.environ["EVALUATION_PROTOCOL"],
    "evaluation_scope": os.environ["EVALUATION_SCOPE"],
    "evaluation_manifest_sha256": os.environ["EVALUATION_MANIFEST_SHA256"],
    "cuda_visible_devices": "0,1,2,3,4,5,6,7",
    "command_file": str(command_file),
    "command_sha256": hashlib.sha256(command_file.read_bytes()).hexdigest(),
    "environment_file": str(environment_file),
    "environment_sha256": hashlib.sha256(environment_file.read_bytes()).hexdigest(),
}
if payload["evaluation_protocol"] == "official_v4":
    payload["official_v4_manifest_sha256"] = payload["evaluation_manifest_sha256"]
Path(os.environ["RUNTIME_MANIFEST"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

build_command() {
  SERVER_COMMAND=(
    "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
    --served-model-name glm-5.2-fp8-pruned-reap-e154
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size 8
    --pipeline-parallel-size 1
    --distributed-executor-backend mp
    --trust-remote-code
    --safetensors-load-strategy lazy
    --attention-backend TRITON_MLA_SPARSE
    --kv-cache-dtype "${EXPECTED_KV_CACHE_DTYPE}"
    --gpu-memory-utilization 0.92
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs 16
    --no-enable-prefix-caching
    --enable-chunked-prefill
    --performance-mode interactivity
    --max-num-batched-tokens 2048
    --disable-custom-all-reduce
    --all2all-backend deepep_low_latency
    --ubatch-size 0
    --enforce-eager
    --seed 42
    --generation-config vllm
    --default-chat-template-kwargs '{"reasoning_effort":"high","enable_thinking":true}'
    --enable-force-include-usage
  )
  if [[ "${DISABLE_ASYNC_SCHEDULING}" == "1" ]]; then
    SERVER_COMMAND+=(--no-async-scheduling)
  fi
  if [[ -n "${PROFILER_CONFIG:-}" ]]; then
    SERVER_COMMAND+=(--profiler-config "${PROFILER_CONFIG}")
  fi
}

print_command() {
  local item
  for item in "${SERVER_COMMAND[@]}"; do
    printf '%q ' "${item}"
  done
  printf '\n'
}

validate_command_args() {
  (
    cd "${SOURCE_DIR}"
    "${PYTHON_BIN}" - \
      "${SERVER_COMMAND[@]:4}" <<'PY'
import json
import os
import sys

import torch
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = make_arg_parser(FlexibleArgumentParser())
args = parser.parse_args(sys.argv[1:])
validate_parsed_serve_args(args)
if args.enable_prefix_caching is not False:
    raise SystemExit("prefix caching must be explicitly disabled")
if args.speculative_config is not None:
    raise SystemExit("speculative decoding must be disabled")
if args.enforce_eager is not True:
    raise SystemExit("eager execution must be enabled")
if args.max_model_len != int(os.environ["MAX_MODEL_LEN"]):
    raise SystemExit(
        f"unexpected max model length: {args.max_model_len}; "
        f"expected {os.environ['MAX_MODEL_LEN']}"
    )
expected_kv_cache_dtype = os.environ["EXPECTED_KV_CACHE_DTYPE"]
if args.kv_cache_dtype != expected_kv_cache_dtype:
    raise SystemExit(
        f"unexpected KV cache dtype: {args.kv_cache_dtype}; "
        f"expected {expected_kv_cache_dtype}"
    )
if os.environ["DISABLE_ASYNC_SCHEDULING"] == "1" and (
    args.async_scheduling is not False
):
    raise SystemExit("asynchronous scheduling must be explicitly disabled")
print(json.dumps({
    "model": args.model_tag,
    "tensor_parallel_size": args.tensor_parallel_size,
    "pipeline_parallel_size": args.pipeline_parallel_size,
    "attention_backend": str(args.attention_backend),
    "kv_cache_dtype": args.kv_cache_dtype,
    "gpu_memory_utilization": args.gpu_memory_utilization,
    "max_model_len": args.max_model_len,
    "max_num_seqs": args.max_num_seqs,
    "max_num_batched_tokens": args.max_num_batched_tokens,
    "enable_chunked_prefill": args.enable_chunked_prefill,
    "enable_prefix_caching": args.enable_prefix_caching,
    "enforce_eager": args.enforce_eager,
    "speculative_config": args.speculative_config,
    "async_scheduling": args.async_scheduling,
    "seed": args.seed,
    "profiler": args.profiler_config.profiler,
    "torch_profiler_dir": args.profiler_config.torch_profiler_dir,
    "cuda_initialized": torch.cuda.is_initialized(),
}, sort_keys=True, default=str))
PY
  ) | tee "${RUN_DIR}/parsed_server_args.json"
}

monitor_server() {
  local server_pid="$1"
  local started ready=0 elapsed=0
  started="$(date +%s)"
  trap 'kill -TERM "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' INT TERM

  while kill -0 "${server_pid}" 2>/dev/null; do
    if [[ "${ready}" -eq 0 ]] && curl -fsS \
      "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      ready=1
      date -u +%FT%TZ | tee "${RUN_DIR}/ready_at_utc.txt"
      printf '%s\n' "${elapsed}" > "${RUN_DIR}/startup_seconds.txt"
      echo "${SERVICE_LABEL} server is ready."
    fi
    sleep 60
    elapsed="$(( $(date +%s) - started ))"
    if ((elapsed > 0 && elapsed % 600 < 60)); then
      {
        printf '%s server_pid=%s ready=%s elapsed_seconds=%s\n' \
          "$(date -u +%FT%TZ)" "${server_pid}" "${ready}" "${elapsed}"
        nvidia-smi \
          --query-gpu=index,memory.used,memory.total,utilization.gpu \
          --format=csv,noheader,nounits
      } | tee -a "${RUN_DIR}/progress_10min.log"
    fi
  done

  set +e
  wait "${server_pid}"
  local status=$?
  set -e
  printf '%s\n' "${status}" > "${RUN_DIR}/server_exit_status.txt"
  echo "${SERVICE_LABEL} server exited with status ${status}."
  return "${status}"
}

serve() {
  [[ "${FORMAL_RUN:-0}" == "1" ]] || {
    echo "ERROR: serve requires FORMAL_RUN=1" >&2
    return 1
  }
  run_static_preflight
  local main_commit source_commit
  main_commit="$(published_commit "${PROJECT_ROOT}" main "${EXPECTED_MAIN_BRANCH}")"
  source_commit="$(published_commit "${SOURCE_REPO}" source "${EXPECTED_SOURCE_BRANCH}")"
  [[ "${source_commit}" == "${EXPECTED_SOURCE_COMMIT}" ]] || {
    echo "ERROR: source repository HEAD is not the expected commit: ${source_commit}" >&2
    return 1
  }
  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "ERROR: an existing server is already healthy at ${HOST}:${PORT}" >&2
    return 1
  fi
  check_gpus_twice
  build_command
  validate_command_args
  print_command > "${RUN_DIR}/serve_command.txt"

  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
  env | LC_ALL=C sort | awk -F= '
    $1 == "CUDA_VISIBLE_DEVICES" ||
    $1 ~ /^EVALUATION_/ ||
    $1 == "FLASHINFER_DISABLE_VERSION_CHECK" ||
    $1 == "GLM52_CANDIDATE_ROOTFS" ||
    $1 == "HF_HOME" ||
    $1 == "HF_HUB_OFFLINE" ||
    $1 == "PROFILER_CONFIG" ||
    $1 == "PYTHONHOME" ||
    $1 == "PYTHONPATH" ||
    $1 == "PYTHONDONTWRITEBYTECODE" ||
    $1 == "TRANSFORMERS_CACHE" ||
    $1 == "VIRTUAL_ENV" ||
    $1 == "XDG_CACHE_HOME" ||
    $1 ~ /^VLLM_/ {print}
  ' > "${RUN_DIR}/runtime_environment.txt"
  write_runtime_manifest "${main_commit}" "${source_commit}" \
    "${RUN_DIR}/serve_command.txt" "${RUN_DIR}/runtime_environment.txt"

  cd "${SOURCE_DIR}"
  "${SERVER_COMMAND[@]}" > "${RUN_DIR}/server.log" 2>&1 &
  local server_pid=$!
  printf '%s\n' "${server_pid}" > "${RUN_DIR}/server.pid"
  monitor_server "${server_pid}"
}

formal_preflight() {
  [[ "${FORMAL_RUN:-0}" == "1" ]] || {
    echo "ERROR: formal-preflight requires FORMAL_RUN=1" >&2
    return 1
  }
  run_static_preflight
  local main_commit source_commit
  main_commit="$(published_commit "${PROJECT_ROOT}" main "${EXPECTED_MAIN_BRANCH}")"
  source_commit="$(published_commit "${SOURCE_REPO}" source "${EXPECTED_SOURCE_BRANCH}")"
  [[ "${source_commit}" == "${EXPECTED_SOURCE_COMMIT}" ]] || {
    echo "ERROR: source repository HEAD is not the expected commit: ${source_commit}" >&2
    return 1
  }
  check_gpus_twice
  printf 'main_commit=%s\nsource_repository_commit=%s\nruntime_source_commit=%s\n' \
    "${main_commit}" "${source_commit}" \
    "${RUNTIME_SOURCE_COMMIT}" \
    > "${RUN_DIR}/published_commits.txt"
}

mode="${1:-}"
case "${mode}" in
  preflight)
    run_static_preflight
    ;;
  dry-run)
    run_static_preflight
    build_command
    validate_command_args
    print_command
    ;;
  check-gpus)
    check_gpus_twice
    ;;
  formal-preflight)
    formal_preflight
    ;;
  serve)
    serve
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
