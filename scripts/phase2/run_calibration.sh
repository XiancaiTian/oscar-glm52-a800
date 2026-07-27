#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PHASE2_ROOT="${ARTIFACT_ROOT}/phase2"
CACHE_ROOT="${CACHE_ROOT:-${PHASE2_ROOT}/cache}"
FIT_CONFIG="${PROJECT_ROOT}/configs/phase2/calibration_fit_initial.json"
CAPTURE_PATH_VALIDATOR="${SCRIPT_DIR}/validate_calibration_capture_paths.py"
CALIBRATION_MANIFEST="${PROJECT_ROOT}/artifacts/phase2/calibration_manifest_final.jsonl"
SOURCE_REPO="${PROJECT_ROOT}/glm52_oscar_vllm"
CANDIDATE_ROOTFS="${PROJECT_ROOT}/artifacts/phase0-candidate-bundle/rootfs"
BASE_SOURCE="${CANDIDATE_ROOTFS}/opt/vllm_glm52_v1"
VENV_DIR="${CANDIDATE_ROOTFS}/opt/fp8_speed_up_v4_venv"
PYTHON_BIN="${CANDIDATE_ROOTFS}/usr/bin/python3.12"
VENV_SITE_PACKAGES="${VENV_DIR}/lib/python3.12/site-packages"
ROOTFS_LOCAL_SITE_PACKAGES="${CANDIDATE_ROOTFS}/usr/local/lib/python3.12/dist-packages"
ROOTFS_DIST_PACKAGES="${CANDIDATE_ROOTFS}/usr/lib/python3/dist-packages"
CANDIDATE_PYTHONPATH="${SOURCE_REPO}:${VENV_SITE_PACKAGES}:${ROOTFS_LOCAL_SITE_PACKAGES}:${ROOTFS_DIST_PACKAGES}"
MODEL_PATH="/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001"
NATIVE_LIB="${CANDIDATE_ROOTFS}/opt/glm52_speed_up_v1_stable/artifacts/native_ext/stage50_sparse_mla_m1_splitmerge_final_ops.so"
MODEL_NAME="glm-5.2-fp8-pruned-reap-e154"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"

export GLM52_CANDIDATE_ROOTFS="${CANDIDATE_ROOTFS}"
export PYTHONHOME="${CANDIDATE_ROOTFS}/usr"
export VIRTUAL_ENV="${VENV_DIR}"
export PYTHONPATH="${CANDIDATE_PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage:
  run_calibration.sh prepare-runtime
  run_calibration.sh preflight train|holdout
  FORMAL_RUN=1 run_calibration.sh serve train|holdout
  CALIBRATION_RUN_ID=<run-id> run_calibration.sh prompts train|holdout
  TRAIN_CALIBRATION_RUN_ID=<run-id> HOLDOUT_CALIBRATION_RUN_ID=<run-id> \
    run_calibration.sh fit

The formal train and holdout captures use separate server processes. Large
capture tensors, prompt responses, and rotation artifacts stay under
ARTIFACT_ROOT.
EOF
}

config_value() {
  local key="$1"
  "${PYTHON_BIN}" - "${FIT_CONFIG}" "${key}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
value = config[sys.argv[2]]
print(value)
PY
}

split_value() {
  local split="$1"
  local suffix="$2"
  config_value "${split}_${suffix}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || {
    echo "ERROR: required file is missing: ${path}" >&2
    return 1
  }
}

published_commit() {
  local repo="$1"
  local expected_branch="$2"
  local label="$3"
  local branch head remote_head remote_url

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
  remote_url="$(git -C "${repo}" remote get-url origin)"
  remote_head="$(
    git ls-remote --heads "${remote_url}" "refs/heads/${branch}" |
      awk 'NR == 1 {print $1}'
  )"
  [[ "${head}" == "${remote_head}" ]] || {
    echo "ERROR: ${label} HEAD ${head} is not published" >&2
    return 1
  }
  printf '%s\n' "${head}"
}

prepare_runtime() {
  local paths=(
    "vllm/_C.abi3.so"
    "vllm/_C_stable_libtorch.abi3.so"
    "vllm/_moe_C.abi3.so"
    "vllm/cumem_allocator.abi3.so"
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"
    "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so"
  )
  local relative source target
  for relative in "${paths[@]}"; do
    source="${BASE_SOURCE}/${relative}"
    target="${SOURCE_REPO}/${relative}"
    require_file "${source}"
    if [[ -L "${target}" ]]; then
      [[ "$(realpath "${target}")" == "$(realpath "${source}")" ]] || {
        echo "ERROR: native symlink has an unexpected target: ${target}" >&2
        return 1
      }
    elif [[ -e "${target}" ]]; then
      echo "ERROR: refusing to replace existing native file: ${target}" >&2
      return 1
    else
      ln -s "${source}" "${target}"
    fi
  done
  (
    cd "${SOURCE_REPO}"
    head -n 6 recovery/native_extensions.sha256 | sha256sum -c -
  )
  [[ "$(sha256sum "${NATIVE_LIB}" | awk '{print $1}')" == \
    "$(tail -n 1 "${SOURCE_REPO}/recovery/native_extensions.sha256" |
      awk '{print $1}')" ]] || {
    echo "ERROR: sparse MLA native extension SHA256 mismatch" >&2
    return 1
  }
  echo "Sparse MLA native extension: OK"
}

gpu_snapshot() {
  local output="$1"
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
      echo "ERROR: GPU is not idle: ${row}" >&2
      return 1
    }
  done
  if nvidia-smi \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "ERROR: a compute process is using a visible GPU" >&2
    return 1
  fi
}

check_gpus_twice() {
  local run_dir="$1"
  gpu_snapshot "${run_dir}/gpu_first.txt"
  echo "Waiting 60 seconds before the required second GPU check."
  sleep 60
  gpu_snapshot "${run_dir}/gpu_second.txt"
  echo "GPU checks passed: 8/8 idle twice."
}

static_preflight() {
  local split="$1"
  local run_dir="$2"
  [[ "${split}" == "train" || "${split}" == "holdout" ]] || {
    echo "ERROR: split must be train or holdout" >&2
    return 1
  }
  require_file "${FIT_CONFIG}"
  require_file "${CAPTURE_PATH_VALIDATOR}"
  require_file "${CALIBRATION_MANIFEST}"
  require_file "${MODEL_PATH}/config.json"
  require_file "${MODEL_PATH}/model.safetensors.index.json"
  require_file "${PYTHON_BIN}"
  require_file "${NATIVE_LIB}"
  require_file "${SOURCE_REPO}/tools/oscar_mla/run_calibration_prompts.py"
  require_file "${SOURCE_REPO}/tools/oscar_mla/fit_calibration.py"

  [[ "$(sha256sum "${CALIBRATION_MANIFEST}" | awk '{print $1}')" == \
    "$(config_value calibration_manifest_sha256)" ]] || {
    echo "ERROR: calibration manifest SHA256 mismatch" >&2
    return 1
  }
  [[ "$(sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')" == \
    "$(config_value model_config_sha256)" ]] || {
    echo "ERROR: model config SHA256 mismatch" >&2
    return 1
  }
  [[ "$(sha256sum "${MODEL_PATH}/model.safetensors.index.json" | awk '{print $1}')" == \
    "$(config_value checkpoint_manifest_sha256)" ]] || {
    echo "ERROR: checkpoint manifest SHA256 mismatch" >&2
    return 1
  }
  [[ "$(
    rg -o 'experts\.[0-9]+' "${MODEL_PATH}/model.safetensors.index.json" |
      LC_ALL=C sort -u | sha256sum | awk '{print $1}'
  )" == "$(config_value expert_mapping_sha256)" ]] || {
    echo "ERROR: checkpoint expert mapping SHA256 mismatch" >&2
    return 1
  }

  mkdir -p "${run_dir}"
  prepare_runtime
  (
    cd "${SOURCE_REPO}"
    "${PYTHON_BIN}" - "${SOURCE_REPO}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

import torch
import vllm
import vllm._C

source = Path(sys.argv[1]).resolve()
if source not in Path(vllm.__file__).resolve().parents:
    raise SystemExit(f"vLLM was not imported from calibration source: {vllm.__file__}")
if source not in Path(vllm._C.__file__).absolute().parents:
    raise SystemExit(f"vllm._C did not resolve through calibration source: {vllm._C.__file__}")
for optional_package in ("flash_attn", "triton_kernels"):
    if importlib.util.find_spec(optional_package) is not None:
        raise SystemExit(f"host package leaked into runtime: {optional_package}")
print(json.dumps({
    "vllm_source": vllm.__file__,
    "vllm_C": vllm._C.__file__,
    "cuda_initialized": torch.cuda.is_initialized(),
}, sort_keys=True))
PY
  ) > "${run_dir}/runtime_import.json"
  local main_commit source_commit expected_source pointer
  main_commit="$(
    published_commit "${PROJECT_ROOT}" feat/glm52-model-load main
  )"
  source_commit="$(
    published_commit \
      "${SOURCE_REPO}" feat/glm52-oscar-integration source
  )"
  expected_source="$(config_value source_commit)"
  [[ "${source_commit}" == "${expected_source}" ]] || {
    echo "ERROR: source commit ${source_commit} != ${expected_source}" >&2
    return 1
  }
  pointer="$(git -C "${PROJECT_ROOT}" ls-tree HEAD glm52_oscar_vllm | awk '{print $3}')"
  [[ "${pointer}" == "${source_commit}" ]] || {
    echo "ERROR: root submodule pointer ${pointer} != source ${source_commit}" >&2
    return 1
  }
  printf 'main_commit=%s\nsource_commit=%s\nsplit=%s\n' \
    "${main_commit}" "${source_commit}" "${split}" \
    > "${run_dir}/published_commits.txt"
}

write_capture_config() {
  local split="$1"
  local run_dir="$2"
  local token_budget reservoir_rows dsa_rows
  token_budget="$(split_value "${split}" token_budget)"
  reservoir_rows="$(split_value "${split}" reservoir_rows)"
  if [[ "${split}" == "holdout" ]]; then
    dsa_rows="$(split_value "${split}" dsa_sample_rows)"
  else
    dsa_rows=0
  fi
  OUTPUT_PATH="${run_dir}/capture_config.json" \
    CAPTURE_OUTPUT="${run_dir}/capture" \
    TOKEN_BUDGET="${token_budget}" \
    RESERVOIR_ROWS="${reservoir_rows}" \
    DSA_ROWS="${dsa_rows}" \
    CAPTURE_SPLIT="${split}" \
    CAPTURE_SEED="$(config_value seed)" \
    LATENT_RANK="$(config_value latent_rank)" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "output_dir": os.environ["CAPTURE_OUTPUT"],
    "token_budget": int(os.environ["TOKEN_BUDGET"]),
    "latent_rank": int(os.environ["LATENT_RANK"]),
    "seed": int(os.environ["CAPTURE_SEED"]),
    "split": os.environ["CAPTURE_SPLIT"],
    "flush_interval_rows": 65536,
    "reservoir_rows": int(os.environ["RESERVOIR_ROWS"]),
    "dsa_sample_rows": int(os.environ["DSA_ROWS"]),
}
Path(os.environ["OUTPUT_PATH"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

build_command() {
  SERVER_COMMAND=(
    "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size 8
    --pipeline-parallel-size 1
    --distributed-executor-backend mp
    --trust-remote-code
    --safetensors-load-strategy lazy
    --attention-backend TRITON_MLA_SPARSE
    --kv-cache-dtype auto
    --gpu-memory-utilization 0.92
    --max-model-len 32768
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
}

validate_command_args() {
  (
    cd "${SOURCE_REPO}"
    "${PYTHON_BIN}" - "${SERVER_COMMAND[@]:4}" <<'PY'
import json
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
    raise SystemExit("prefix caching must be disabled")
if args.speculative_config is not None:
    raise SystemExit("speculative decoding must be disabled")
if args.enforce_eager is not True:
    raise SystemExit("eager execution must be enabled")
print(json.dumps({
    "tensor_parallel_size": args.tensor_parallel_size,
    "attention_backend": str(args.attention_backend),
    "kv_cache_dtype": args.kv_cache_dtype,
    "max_model_len": args.max_model_len,
    "enable_prefix_caching": args.enable_prefix_caching,
    "enforce_eager": args.enforce_eager,
    "cuda_initialized": torch.cuda.is_initialized(),
}, sort_keys=True, default=str))
PY
  )
}

export_runtime_environment() {
  local capture_config="$1"
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
  export VLLM_OSCAR_MLA_CAPTURE_CONFIG="${capture_config}"
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
}

monitor_server() {
  local server_pid="$1"
  local run_dir="$2"
  local started ready elapsed next_progress
  started="$(date +%s)"
  ready=0
  next_progress=600
  trap 'kill -TERM "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' INT TERM
  while kill -0 "${server_pid}" 2>/dev/null; do
    if [[ "${ready}" -eq 0 ]] && curl -fsS \
      "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      ready=1
      date -u +%FT%TZ | tee "${run_dir}/ready_at_utc.txt"
    fi
    sleep 60
    elapsed="$(( $(date +%s) - started ))"
    if ((elapsed >= next_progress)); then
      {
        printf '%s server_pid=%s ready=%s elapsed_seconds=%s\n' \
          "$(date -u +%FT%TZ)" "${server_pid}" "${ready}" "${elapsed}"
        nvidia-smi \
          --query-gpu=index,memory.used,memory.total,utilization.gpu \
          --format=csv,noheader,nounits
      } | tee -a "${run_dir}/progress_10min.log"
      next_progress="$((next_progress + 600))"
    fi
  done
  set +e
  wait "${server_pid}"
  local status=$?
  set -e
  printf '%s\n' "${status}" > "${run_dir}/server_exit_status.txt"
  return "${status}"
}

serve_split() {
  local split="$1"
  [[ "${FORMAL_RUN:-0}" == "1" ]] || {
    echo "ERROR: serve requires FORMAL_RUN=1" >&2
    return 1
  }
  mkdir -p "${PHASE2_ROOT}"
  local serve_lock="${PHASE2_ROOT}/.serve-${HOST}-${PORT}.lock"
  local serve_lock_fd
  exec {serve_lock_fd}> "${serve_lock}"
  flock -n "${serve_lock_fd}" || {
    echo "ERROR: another calibration server owns ${serve_lock}" >&2
    return 1
  }
  local run_id run_dir
  run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_calibration_${split}_tp8}"
  run_dir="${PHASE2_ROOT}/${run_id}"
  [[ ! -e "${run_dir}" ]] || {
    echo "ERROR: run directory already exists: ${run_dir}" >&2
    return 1
  }
  mkdir -p "${run_dir}"
  static_preflight "${split}" "${run_dir}"
  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "ERROR: an existing server is healthy at ${HOST}:${PORT}" >&2
    return 1
  fi
  build_command
  validate_command_args > "${run_dir}/parsed_server_args.json"
  check_gpus_twice "${run_dir}"
  write_capture_config "${split}" "${run_dir}"
  export_runtime_environment "${run_dir}/capture_config.json"
  printf '%q ' "${SERVER_COMMAND[@]}" > "${run_dir}/serve_command.txt"
  printf '\n' >> "${run_dir}/serve_command.txt"
  env | LC_ALL=C sort | awk -F= '
    $1 == "CUDA_VISIBLE_DEVICES" ||
    $1 == "HF_HOME" ||
    $1 == "HF_HUB_OFFLINE" ||
    $1 == "PYTHONHOME" ||
    $1 == "PYTHONPATH" ||
    $1 == "PYTHONDONTWRITEBYTECODE" ||
    $1 == "TRANSFORMERS_CACHE" ||
    $1 == "VIRTUAL_ENV" ||
    $1 == "XDG_CACHE_HOME" ||
    $1 ~ /^VLLM_/ {print}
  ' > "${run_dir}/runtime_environment.txt"

  (
    cd "${SOURCE_REPO}"
    "${SERVER_COMMAND[@]}" > "${run_dir}/server.log" 2>&1 &
    server_pid=$!
    printf '%s\n' "${server_pid}" > "${run_dir}/server.pid"
    monitor_server "${server_pid}" "${run_dir}"
  )
}

run_prompts() {
  local split="$1"
  local run_id="${CALIBRATION_RUN_ID:?set CALIBRATION_RUN_ID}"
  local run_dir="${PHASE2_ROOT}/${run_id}"
  require_file "${run_dir}/server.pid"
  require_file "${run_dir}/capture_config.json"
  [[ "$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["split"])' \
    "${run_dir}/capture_config.json")" == "${split}" ]] || {
    echo "ERROR: capture config split does not match ${split}" >&2
    return 1
  }
  local server_pid
  server_pid="$(<"${run_dir}/server.pid")"
  kill -0 "${server_pid}" 2>/dev/null || {
    echo "ERROR: server PID ${server_pid} is not running" >&2
    return 1
  }
  tr '\0' '\n' < "/proc/${server_pid}/environ" |
    grep -Fx \
      "VLLM_OSCAR_MLA_CAPTURE_CONFIG=${run_dir}/capture_config.json" \
      >/dev/null || {
    echo "ERROR: server PID ${server_pid} is not using this capture config" >&2
    return 1
  }
  curl -fsS "http://${HOST}:${PORT}/health" >/dev/null
  (
    cd "${SOURCE_REPO}"
    "${PYTHON_BIN}" tools/oscar_mla/run_calibration_prompts.py \
      --manifest "${CALIBRATION_MANIFEST}" \
      --split "${split}" \
      --expected-tokens "$(split_value "${split}" token_budget)" \
      --base-url "http://${HOST}:${PORT}/v1" \
      --model "${MODEL_NAME}" \
      --output-dir "${run_dir}/prompt_run" \
      --timeout-seconds 1800
  ) | tee "${run_dir}/prompt_runner.log"

  local expected_layers expected_files actual_files
  expected_layers="$(config_value num_layers)"
  expected_files="$(( $(config_value tp_size) * expected_layers ))"
  actual_files="$(find "${run_dir}/capture" -type f -name '*.pt' | wc -l)"
  [[ "${actual_files}" -eq "${expected_files}" ]] || {
    echo "ERROR: capture files ${actual_files} != ${expected_files}" >&2
    return 1
  }
  printf 'capture_files=%s\nexpected_layers=%s\n' \
    "${actual_files}" "${expected_layers}" \
    > "${run_dir}/capture_validation.txt"
}

fit_artifact() {
  local train_id="${TRAIN_CALIBRATION_RUN_ID:?set TRAIN_CALIBRATION_RUN_ID}"
  local holdout_id="${HOLDOUT_CALIBRATION_RUN_ID:?set HOLDOUT_CALIBRATION_RUN_ID}"
  local output_id="${FIT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_rotation_fit}"
  local output_dir="${PHASE2_ROOT}/${output_id}"
  [[ ! -e "${output_dir}" ]] || {
    echo "ERROR: fit output already exists: ${output_dir}" >&2
    return 1
  }
  local source_commit
  source_commit="$(
    published_commit \
      "${SOURCE_REPO}" feat/glm52-oscar-integration source
  )"
  [[ "${source_commit}" == "$(config_value source_commit)" ]] || {
    echo "ERROR: fit source commit does not match config" >&2
    return 1
  }
  "${PYTHON_BIN}" "${CAPTURE_PATH_VALIDATOR}" \
    --config "${FIT_CONFIG}" \
    --train-capture-dir \
      "${PHASE2_ROOT}/${train_id}/capture" \
    --holdout-capture-dir \
      "${PHASE2_ROOT}/${holdout_id}/capture"
  (
    cd "${SOURCE_REPO}"
    "${PYTHON_BIN}" tools/oscar_mla/fit_calibration.py \
      --config "${FIT_CONFIG}" \
      --train-capture-dir \
        "${PHASE2_ROOT}/${train_id}/capture" \
      --holdout-capture-dir \
        "${PHASE2_ROOT}/${holdout_id}/capture" \
      --calibration-code-commit "${source_commit}" \
      --output-dir "${output_dir}"
  ) | tee "${output_dir}.log"
}

mode="${1:-}"
split="${2:-}"
case "${mode}" in
  prepare-runtime)
    prepare_runtime
    ;;
  preflight)
    run_id="${RUN_ID:-preflight_calibration_${split}}"
    static_preflight "${split}" "${PHASE2_ROOT}/${run_id}"
    ;;
  serve)
    serve_split "${split}"
    ;;
  prompts)
    run_prompts "${split}"
    ;;
  fit)
    fit_artifact
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
