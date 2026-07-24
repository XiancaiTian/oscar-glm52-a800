#!/usr/bin/env bash
set -euo pipefail
ROLE="${1:?role prefill|decode|proxy|nonpd}"
NODE_RANK="${2:-0}"
SCHEME="${3:-baseline_v7_stage80_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_20260601}"
TS="${4:?timestamp}"

if [[ -n "${TASK_ROOT_IN_CONTAINER:-}" && -d "${TASK_ROOT_IN_CONTAINER}" ]]; then
  TASK_ROOT="${TASK_ROOT_IN_CONTAINER}"
  BASE="$(dirname "${TASK_ROOT}")"
elif [[ -n "${TASK_ROOT:-}" && -d "${TASK_ROOT}" ]]; then
  BASE="$(dirname "${TASK_ROOT}")"
else
  BASE="/nfs/AE/zhanghong/workflow/vllm_a"
  TASK_ROOT="${BASE}/glm52_speed_up_v2_stable_8th"
fi
SOURCE_DIR="/opt/vllm_glm52_v1"
VENV="${VENV:-/opt/fp8_speed_up_v4_venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV}/bin/python}"
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:${PATH}"
DEFAULT_NSYS_BIN="${BASE}/bench_serve_test_v2_431bdcc_workcode_del/bench_serve_test_v7_nsys_task/tools/nsight-systems-cli-2026.2.1/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys"
VLLM_STAGE_NSYS_LAUNCH="${VLLM_STAGE_NSYS_LAUNCH:-0}"
VLLM_STAGE_NSYS_ROLES="${VLLM_STAGE_NSYS_ROLES:-decode}"
VLLM_STAGE_NSYS_SESSION_PREFIX="${VLLM_STAGE_NSYS_SESSION_PREFIX:-stage50nsys}"
VLLM_STAGE_NSYS_BIN="${VLLM_STAGE_NSYS_BIN:-${DEFAULT_NSYS_BIN}}"
VLLM_STAGE_NSYS_TRACE="${VLLM_STAGE_NSYS_TRACE:-cuda,nvtx,osrt,cublas,nccl}"
VLLM_STAGE_NSYS_CUDA_MEMORY="${VLLM_STAGE_NSYS_CUDA_MEMORY:-false}"
VLLM_STAGE_NSYS_OUT_DIR="${VLLM_STAGE_NSYS_OUT_DIR:-${TASK_ROOT}/logs/profile/nsys}"
VLLM_STAGE_NSYS_LAUNCH_EXTRA_ARGS="${VLLM_STAGE_NSYS_LAUNCH_EXTRA_ARGS:-}"
MODEL="${MODEL_PATH:-/nfs/AIED/models/GLM-5.2-FP8}"
MODEL_NAME="${MODEL_ID:-GLM-5.2-FP8}"
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-lazy}"
LOG_DIR="${TASK_ROOT}/logs/deploy/${SCHEME}"
XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg"
case "${SCHEME}" in
  stage_70ms_*|stage_50ms_*)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_cachealias_xdgold_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_mqabn512_ncclsimple_empty_logits_skip_decode_clear_sparse_singlepass_20260603"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_hca1_ppsampleonly_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_pairp2p_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_firstp2p_middleskip_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_chunk262k_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_ppfastpath_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_replay_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_pairp2p_skipfinal_xdgfresh_20260604|stage50_mqabn512_ncclsimple_replay_xdgfresh_20260604|stage50_mqabn512_ncclsimple_mbt2048_xdgfresh_20260604|stage50_mqabn512_ncclsimple_maxseq8_xdgfresh_20260604|stage50_mqabn512_nccl_ll_xdgfresh_20260604|stage50_mqabn512_ncclsimple_chunk262k_20260603|stage50_mqabn512_ncclsimple_metacache_20260603|stage50_mqabn512_ncclsimple_tree_20260603|stage50_mqabn512_nccl_ll128_20260603|stage50_mqabn512_nccl_ll128_xdgfresh_20260603|stage50_mqabn512_ncclsimple_ch1_xdgfresh_20260603|stage50_mqabn512_ncclsimple_ch16_xdgfresh_20260603|stage50_mqabn512_ncclsimple_ch4_xdgfresh_20260603|stage50_mqabn512_ncclsimple_skipfinal_xdgfresh_20260603|stage50_mqabn512_ncclsimple_cpubcast_xdgfresh_20260603|stage50_mqabn512_ncclsimple_cpup2p_xdgfresh_20260604|stage50_mqabn512_ncclsimple_pairp2p_xdgfresh_20260604|stage50_mqabn512_ncclsimple_recvbuf_xdgfresh_20260603|stage50_mqabn512_ncclsimple_firstp2p_xdgfresh_20260603|stage50_mqabn512_ncclsimple_hca1_xdgfresh_20260603|stage50_mqabn512_ncclsimple_noasync_xdgfresh_20260603|stage50_mqabn512_ncclsimple_decodeswap_xdgfresh_20260603|stage50_mqabn512_ncclsimple_asyncbcast_xdgfresh_20260603|stage50_mqabn512_ncclsimple_middleskip_xdgfresh_20260603|stage50_pf4pp4_dtp4pp3_mqabn512_ncclsimple_20260604|stage50_pf4pp4_dtp4pp2_mqabn512_ncclsimple_20260604|stage50_pf4pp4_dtp8pp2_mqabn512_ncclsimple_20260603|stage50_stage70reqbn256_ncclsimple_xdgfresh_20260603)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_hca1_ppbatchp2p_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_noskip_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_prefixmask_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppsendwaitdefer_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_decodem1final64_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_hca1_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260604|stage50_mqabn512_ncclsimple_skipfinal_noallgather_tree_xdgfresh_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_cachefresh_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_noautobench_20260605|stage50_mqabn512_ncclsimple_persistenttopk_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_fulldecode_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_fulldecode_ppclone_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_fulldecode_ppcopybuf_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_inline_sample_broadcast_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_ppsendwaitdefer_xdgfresh_20260605|stage50_mqabn512_ncclsimple_dvtile128bh8_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_decodeprepfast_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_deferrecv_skipfinal_noallgather_xdgfresh_20260606|stage50_ppqueue6_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260606|stage50_ppqueue6_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_recoveryfresh_20260606|stage50_pf4pp4_dtp4pp3_mqfix_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260605)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_fulldecode_strongout_skipfinal_noallgather_xdgfresh_20260608)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_splitkv16_skipfinal_noallgather_xdgfresh_20260608|stage50_mqabn512_ncclsimple_splitkv16_skipfinal_noallgather_xdgfresh_formalfresh_20260608|stage50_mqabn512_ncclsimple_splitkv16_noautobench_skipfinal_noallgather_xdgfresh_20260608|stage50_mqabn512_ncclsimple_splitkv4_skipfinal_noallgather_xdgfresh_20260608|stage50_mqabn512_ncclsimple_splitkv2_skipfinal_noallgather_xdgfresh_20260608)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_inductorpart_splitops_skipfinal_noallgather_xdgfresh_20260608|stage50_mqabn512_ncclsimple_inductorpart_fxsplit_skipfinal_noallgather_xdgfresh_20260608)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_pf4pp4_dtp4pp3_mqfix_highmem093_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260605)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_tp4pp3_mqabn512_ncclsimple_homopp_highmem093_skipfinal_noallgather_xdgfresh_20260606)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_pf4pp4_dtp16pp1_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260606|stage50_pf4pp4_dtp16pp1_highmem093_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260606)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_tp16pp1_mqabn512_ncclsimple_homopp_highmem093_skipfinal_noallgather_xdgfresh_20260606|stage50_tp16pp1_mqabn512_ncclsimple_homopp_highmem0925_skipfinal_noallgather_xdgfresh_20260606|stage50_tp16pp1_mqabn512_ncclsimple_homopp_highmem0925_formalfresh_skipfinal_noallgather_xdgfresh_20260606)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh2_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh3_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh4_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh5_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh6_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh7_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh8_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh9_20260606)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh10_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh11_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh12_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh13_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh14_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh15_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh16_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh17_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh20_combobenchoff_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh21_after_coreopxid_20260610|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_fp8lut_20260610|stage50_mqabn512_ncclsimple_coopfinal_graphsafe_skipfinal_noallgather_xdgfresh_20260609|stage50_mqabn512_ncclsimple_splitmergefinal_decodeonly_skipfinal_noallgather_xdgfresh_20260609)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_fp8lutfresh1_20260610)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_fp8lutfresh1_splitmerge_persistent_bp2w4s2_20260610|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_fp8lutfresh1_splitmerge_warpreduce_persistent_bp2w4s2_20260610|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_fp8lutfresh1_splitmerge_coreop_persistent_bp2w4s2_20260610)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_readyfirst_skipfinal_noallgather_xdgfresh_20260609)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_ppqueue2exact_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260609)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_asyncbcast_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppbatchp2p_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_part20201919_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_part19202019_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_earlybcast_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_decodemeta_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_noskip_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P="1"
    export VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN="0"
    export VLLM_PP_DECODE_TENSOR_DICT_METADATA_SKIP="0"
    export VLLM_PP_BATCH_P2P_TENSOR_DICT="0"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_early_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_decodemeta_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_decodemeta_ppbatchp2p_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_metacache_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppsampleonly_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppsampleonly_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_reqbn512_envcache_replay_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_nccl_ll_pairp2p_xdgfresh_20260604|stage50_mqabn512_nccl_ll_skipfinal_xdgfresh_20260604|stage50_mqabn512_nccl_ll_skipfinal_noallgather_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_predeqq_notrim_xdgfresh_20260604)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_${SCHEME}"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_finaldyn64_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_final64_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601"
    ;;
esac
if [[ -n "${VLLM_NORM_CAPTURE_MODE:-}" || -n "${VLLM_NORM_CAPTURE_HOOK:-}" || -n "${VLLM_LAYER_CAPTURE_DIR:-}" ]]; then
  XDG_CACHE_ROOT="${TASK_ROOT}/cache/xdg_capture_${SCHEME}_${TS}"
fi
if [[ -n "${XDG_CACHE_HOME_OVERRIDE:-}" ]]; then
  XDG_CACHE_ROOT="${XDG_CACHE_HOME_OVERRIDE}"
fi
mkdir -p "${LOG_DIR}" "${TASK_ROOT}/cache/hf" "${XDG_CACHE_ROOT}"
cd "${BASE}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SOURCE_DIR}:${PYTHONPATH:-}"
export VLLM_1P1D_ROLE="${ROLE}"
export HF_HOME="${TASK_ROOT}/cache/hf"
export TRANSFORMERS_CACHE="${TASK_ROOT}/cache/hf/transformers"
export XDG_CACHE_HOME="${XDG_CACHE_ROOT}"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OPENAI_API_KEY="EMPTY"
export HF_HUB_OFFLINE="1"
export FLASHINFER_DISABLE_VERSION_CHECK="1"
export VLLM_KV_CACHE_LAYOUT="HND"
export VLLM_ENABLE_V1_MULTIPROCESSING="1"
export VLLM_ALLREDUCE_USE_SYMM_MEM="0"
export VLLM_SPARSE_INDEXER_MQA_LOGITS_BACKEND="${VLLM_SPARSE_INDEXER_MQA_LOGITS_BACKEND:-cuda_v7}"
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-174}"
export VLLM_MQA_CUDA_V7_GROUP32_TILED="1"
export VLLM_MQA_CUDA_V7_GROUP32_TILE_N="26112"
export VLLM_MQA_CUDA_V7_GROUP16_MAX_ELEMENTS="67108864"
export VLLM_MQA_CUDA_V7_GROUP32_MAX_ELEMENTS="0"
export VLLM_MQA_CUDA_V7_SKIP_LATER_GROUP_MASK="0"
export VLLM_MQA_CUDA_V7_FLAT_GROUP_GEMM="0"
export VLLM_MQA_CUDA_V7_FAST_BF16_GEMM="0"
export VLLM_MQA_CUDA_V7_PAD_N="1"
export VLLM_MQA_CUDA_V7_PREDEQUANT_K="1"
export VLLM_MQA_CUDA_V7_PREDEQUANT_Q="1"
export VLLM_MQA_CUDA_V7_FUSED_TRITON="${VLLM_MQA_CUDA_V7_FUSED_TRITON:-1}"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_PREFILL_CANONICAL_M="${VLLM_MQA_CUDA_V7_FUSED_TRITON_PREFILL_CANONICAL_M:-512}"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_MIN_N="4096"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_MAX_N="65536"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_BLOCK_M="${VLLM_MQA_CUDA_V7_FUSED_TRITON_BLOCK_M:-16}"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_BLOCK_N="${VLLM_MQA_CUDA_V7_FUSED_TRITON_BLOCK_N:-128}"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_ROW_START_ZERO="1"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_ROW_START_MIN_M="0"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_ROW_END_CONTIGUOUS="1"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_ROW_END_MIN_M="0"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_FAST_INVALID_TILE="1"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_SKIP_INVALID_STORE="1"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_FAST_FULL_TILE="1"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_REUSE_K_TILE="1"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_NUM_WARPS="${VLLM_MQA_CUDA_V7_FUSED_TRITON_NUM_WARPS:-4}"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_NUM_STAGES="${VLLM_MQA_CUDA_V7_FUSED_TRITON_NUM_STAGES:-3}"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_M_MAX="${VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_M_MAX:-}"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_M="${VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_M:-}"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="${VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N:-}"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_NUM_WARPS="${VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_NUM_WARPS:-}"
export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_NUM_STAGES="${VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_NUM_STAGES:-}"
if [[ "${ROLE}" == "prefill" ]]; then
  unset VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_M_MAX
  unset VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_M
  unset VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N
  unset VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_NUM_WARPS
  unset VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_NUM_STAGES
fi
export VLLM_SPARSE_INDEXER_SKIP_PREFILL_TOPK_CLEAR="1"
export VLLM_SPARSE_INDEXER_PREFILL_DECODE_TOPK="1"
export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="${VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS:-0}"
export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="${VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE:-8192}"
export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="${VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR:-0}"
export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="${VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS:-0}"
export VLLM_SPARSE_INDEXER_DECODE_FP8_LUT="${VLLM_SPARSE_INDEXER_DECODE_FP8_LUT:-0}"
export VLLM_SPARSE_INDEXER_DECODE_PREDEQUANT_Q="${VLLM_SPARSE_INDEXER_DECODE_PREDEQUANT_Q:-0}"
export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="${VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE:-0}"
export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="${VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE:-0}"
export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="${VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND:-persistent}"
export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="${VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES:-1}"
export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="${VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS:-4}"
export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="${VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES:-3}"
export VLLM_SPARSE_INDEXER_DECODE_TOPK_PAD_LOGITS_LEN="${VLLM_SPARSE_INDEXER_DECODE_TOPK_PAD_LOGITS_LEN:-0}"
export VLLM_SPARSE_INDEXER_DECODE_TOPK_HIST_FUSION="${VLLM_SPARSE_INDEXER_DECODE_TOPK_HIST_FUSION:-0}"
export VLLM_SPARSE_INDEXER_DECODE_TOPK_BIN_FUSION="${VLLM_SPARSE_INDEXER_DECODE_TOPK_BIN_FUSION:-0}"
export VLLM_SPARSE_INDEXER_DECODE_TOPK_TILE_SELECT="${VLLM_SPARSE_INDEXER_DECODE_TOPK_TILE_SELECT:-0}"
export VLLM_SPARSE_INDEXER_DECODE_TOPK_TILE_SELECT_CANDIDATES="${VLLM_SPARSE_INDEXER_DECODE_TOPK_TILE_SELECT_CANDIDATES:-16}"
export VLLM_TOPK_ENV_CACHE="${VLLM_TOPK_ENV_CACHE:-0}"
export VLLM_TOPK_DECODE_SPLIT_THRESHOLD="${VLLM_TOPK_DECODE_SPLIT_THRESHOLD:-}"
export VLLM_TOPK_DECODE_SPLIT_BLOCKS="${VLLM_TOPK_DECODE_SPLIT_BLOCKS:-}"
export VLLM_SPARSE_MLA_BLOCK_H="${VLLM_SPARSE_MLA_BLOCK_H:-16}"
export VLLM_SPARSE_MLA_FINAL_CONFIG="${VLLM_SPARSE_MLA_FINAL_CONFIG:-16,2,1}"
export VLLM_SPARSE_MLA_FINAL_DYNAMIC_CONFIG="${VLLM_SPARSE_MLA_FINAL_DYNAMIC_CONFIG:-0}"
export VLLM_SPARSE_MLA_FINAL_DYNAMIC_SMALL_M="${VLLM_SPARSE_MLA_FINAL_DYNAMIC_SMALL_M:-1024}"
export VLLM_SPARSE_MLA_FINAL_DYNAMIC_LARGE_M="${VLLM_SPARSE_MLA_FINAL_DYNAMIC_LARGE_M:-4096}"
export VLLM_SPARSE_MLA_FINAL_DYNAMIC_SMALL_CONFIG="${VLLM_SPARSE_MLA_FINAL_DYNAMIC_SMALL_CONFIG:-16,2,4}"
export VLLM_SPARSE_MLA_FINAL_DYNAMIC_MID_CONFIG="${VLLM_SPARSE_MLA_FINAL_DYNAMIC_MID_CONFIG:-16,2,1}"
export VLLM_SPARSE_MLA_FINAL_DYNAMIC_LARGE_CONFIG="${VLLM_SPARSE_MLA_FINAL_DYNAMIC_LARGE_CONFIG:-16,2,2}"
export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="${VLLM_SPARSE_MLA_DECODE_M1_FINAL64:-0}"
export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_FINAL="${VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_FINAL:-0}"
export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE="${VLLM_SPARSE_MLA_DECODE_M1_DV_TILE:-128}"
export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_BLOCK_H="${VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_BLOCK_H:-8}"
export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_BLOCK_N="${VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_BLOCK_N:-32}"
export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_NUM_WARPS="${VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_NUM_WARPS:-4}"
export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_NUM_STAGES="${VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_NUM_STAGES:-1}"
if [[ "${ROLE}" != "decode" ]]; then
  export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
  export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_FINAL="0"
fi
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_DECODE_ONLY="${VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_DECODE_ONLY:-0}"
if [[ "${VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_DECODE_ONLY}" == "1" && "${ROLE}" != "decode" ]]; then
  export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL="0"
  export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_UNSAFE_ENABLE="0"
fi
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL="${VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL:-0}"
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_UNSAFE_ENABLE="${VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_UNSAFE_ENABLE:-0}"
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_NUM_SPLITS="${VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_NUM_SPLITS:-32}"
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_LIB="${VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_LIB:-}"
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_DEBUG_LOGS="${VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_DEBUG_LOGS:-0}"
export VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_SYNC_DEBUG="${VLLM_SPARSE_MLA_M1_SPLITMERGE_FINAL_SYNC_DEBUG:-0}"
export VLLM_SPARSE_MLA_REUSE_K_AS_V="1"
export VLLM_SPARSE_MLA_FULL_BLOCK_H="1"
export VLLM_SPARSE_MLA_ASSUME_VALID_NOMASK="${VLLM_SPARSE_MLA_ASSUME_VALID_NOMASK:-0}"
export VLLM_SPARSE_MLA_ASSUME_VALID_AFTER_TOPK_NOMASK="1"
export VLLM_SPARSE_MLA_ASSUME_PREFIX_LEN_MASK="1"
export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="${VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE:-0}"
export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE_MAX_TOKENS="${VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE_MAX_TOKENS:-4}"
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_moebm64_nofp32reduce_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_MARLIN_MOE_BLOCK_SIZE_M="${VLLM_MARLIN_MOE_BLOCK_SIZE_M:-64}"
    export VLLM_MARLIN_MOE_USE_FP32_REDUCE_UNSET="${VLLM_MARLIN_MOE_USE_FP32_REDUCE_UNSET:-1}"
    ;;
esac
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD="4096"
export VLLM_MARLIN_MOE_BLOCK_SIZE_M="${VLLM_MARLIN_MOE_BLOCK_SIZE_M:-48}"
export VLLM_MARLIN_USE_FP32_REDUCE="${VLLM_MARLIN_USE_FP32_REDUCE:-0}"
if [[ "${VLLM_MARLIN_MOE_USE_FP32_REDUCE_UNSET:-0}" == "1" ]]; then
  unset VLLM_MARLIN_MOE_USE_FP32_REDUCE
else
  export VLLM_MARLIN_MOE_USE_FP32_REDUCE="${VLLM_MARLIN_MOE_USE_FP32_REDUCE:-1}"
fi
export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="${VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST:-0}"
export VLLM_PP_CPU_SAMPLED_TOKEN_BROADCAST="${VLLM_PP_CPU_SAMPLED_TOKEN_BROADCAST:-0}"
export VLLM_PP_CPU_FIRST_RANK_SAMPLED_TOKEN_P2P="${VLLM_PP_CPU_FIRST_RANK_SAMPLED_TOKEN_P2P:-0}"
export VLLM_PP_SAMPLED_TOKEN_RECV_BUFFER="${VLLM_PP_SAMPLED_TOKEN_RECV_BUFFER:-0}"
export VLLM_PP_SAMPLED_TOKEN_PAIR_P2P="${VLLM_PP_SAMPLED_TOKEN_PAIR_P2P:-0}"
export VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P="${VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P:-0}"
export VLLM_PP_ASYNC_SAMPLED_TOKEN_BROADCAST="${VLLM_PP_ASYNC_SAMPLED_TOKEN_BROADCAST:-0}"
export VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN="${VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN:-0}"
export VLLM_PP_BROADCAST_SAMPLED_TOKEN_ONLY="${VLLM_PP_BROADCAST_SAMPLED_TOKEN_ONLY:-0}"
export VLLM_PP_DIRECT_RECV_INTERMEDIATE="${VLLM_PP_DIRECT_RECV_INTERMEDIATE:-0}"
export VLLM_PP_BATCH_P2P_TENSOR_DICT="${VLLM_PP_BATCH_P2P_TENSOR_DICT:-0}"
export VLLM_PP_TENSOR_DICT_METADATA_CACHE="${VLLM_PP_TENSOR_DICT_METADATA_CACHE:-0}"
export VLLM_PP_DECODE_TENSOR_DICT_METADATA_SKIP="${VLLM_PP_DECODE_TENSOR_DICT_METADATA_SKIP:-0}"
export VLLM_PP_EARLY_SAMPLED_TOKEN_BROADCAST="${VLLM_PP_EARLY_SAMPLED_TOKEN_BROADCAST:-0}"
export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="${VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER:-0}"
export VLLM_PP_DEFER_INTERMEDIATE_RECV="${VLLM_PP_DEFER_INTERMEDIATE_RECV:-0}"
export VLLM_PP_DEFER_SEND_WAIT="${VLLM_PP_DEFER_SEND_WAIT:-0}"
export VLLM_PP_CLONE_CUDAGRAPH_OUTPUT_BEFORE_SEND="${VLLM_PP_CLONE_CUDAGRAPH_OUTPUT_BEFORE_SEND:-0}"
export VLLM_PP_COPY_CUDAGRAPH_OUTPUT_BEFORE_SEND="${VLLM_PP_COPY_CUDAGRAPH_OUTPUT_BEFORE_SEND:-0}"
export VLLM_FULL_CUDAGRAPH_STRONG_OUTPUT="${VLLM_FULL_CUDAGRAPH_STRONG_OUTPUT:-0}"
export VLLM_STAGE50_DECODE_PREP_FASTPATH="${VLLM_STAGE50_DECODE_PREP_FASTPATH:-0}"
export VLLM_STAGE50_PP3_GREEDY_LOCAL_ARGMAX="${VLLM_STAGE50_PP3_GREEDY_LOCAL_ARGMAX:-0}"
export VLLM_STAGE50_PP_READY_FIRST_DRAIN="${VLLM_STAGE50_PP_READY_FIRST_DRAIN:-0}"
export VLLM_SKIP_CHUNKED_PREFILL_LOGITS="0"
export VLLM_FP8_SHARED_EXPERTS_DEQUANT_BF16="1"
export VLLM_FP8_ATTN_LINEAR_DEQUANT_BF16="1"
if [[ -n "${COMPILATION_CONFIG_B64:-}" ]]; then
  COMPILATION_CONFIG_JSON="$(printf '%s' "${COMPILATION_CONFIG_B64}" | base64 -d)"
  export COMPILATION_CONFIG_JSON
fi
case "${SCHEME}" in
  accv2_tp4pp2|stage_100ms_tp4pp2_sparse_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="38,40"
    TP_SIZE="4"
    PP_SIZE="2"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="10208"
    ;;
  stage_100ms_tp4pp3_sparse_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="26,26,26"
    TP_SIZE="4"
    PP_SIZE="3"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp2pp6_sparse_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="13,13,13,13,13,13"
    TP_SIZE="2"
    PP_SIZE="6"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_asym_prefill_tp2pp8_decode_tp4pp4_directrecv_sparse_singlepass_20260529)
    if [[ "${ROLE}" == "prefill" ]]; then
      export VLLM_PP_LAYER_PARTITION="10,9,9,10,10,10,10,10"
      TP_SIZE="2"
      PP_SIZE="8"
    else
      export VLLM_PP_LAYER_PARTITION="19,20,19,20"
      TP_SIZE="4"
      PP_SIZE="4"
    fi
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp4pp4_sparse_singlepass_20260528|stage_100ms_tp4pp4_mbt4096_sparse_singlepass_20260528|stage_100ms_tp4pp4_splitkv_20260528|stage_100ms_tp4pp4_final16_sparse_singlepass_20260528|stage_100ms_tp4pp4_mqa_bn256_sparse_singlepass_20260528|stage_100ms_tp4pp4_mqa_cuda_sparse_singlepass_20260528|stage_100ms_tp4pp4_mqa_decode_bn256_sparse_singlepass_20260528|stage_100ms_tp4pp4_directrecv_moe_bm64_sparse_singlepass_20260528|stage_100ms_tp4pp4_directrecv_moe_bm16_sparse_singlepass_20260529|stage_100ms_tp4pp4_noeager_sparse_singlepass_20260528|stage_100ms_tp4pp4_noeager_skipfinal_sparse_singlepass_20260529|stage_100ms_tp4pp4_noeager_sync_sparse_singlepass_20260529|stage_100ms_tp4pp4_custom_ar_sparse_singlepass_20260528|profile_tp4pp4_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp4pp4_part20201820_sparse_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="20,20,18,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp4pp4_part18202020_directrecv_sparse_singlepass_20260529)
    export VLLM_PP_LAYER_PARTITION="18,20,20,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_mqaw8_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_bh32_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_topkpad200k_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  baseline_v7_stage80_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_20260601|stage_100ms_tp4pp4_directrecv_sparse_singlepass_20260528|stage_100ms_tp4pp4_directrecv_compile_piecewise_sparse_singlepass_20260529|stage_100ms_tp4pp4_directrecv_compile_nocg_sparse_singlepass_20260529|stage_100ms_tp4pp4_directrecv_compile_piecewise_copyinputs_sparse_singlepass_20260529|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_decodem1_sparse_singlepass_20260529|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_decodem1bn256_sparse_singlepass_20260529|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_mqa256w8_final16_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_decodem1bn512_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_decode_trim8192_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_noasync_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_blockgrid_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim4096_legacytopk_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_persistenttopk_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_pagedblk2_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_combo_bench_off_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trimexact_legacytopk_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_moebm8_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_splitkvauto_empty_logits_skip_decode_clear_sparse_dsa_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_fusedreq_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_finaldyn64_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_topksplit4_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_100ms_tp4pp4_directrecv_mlp_bf16_sparse_singlepass_20260529|stage_100ms_tp4pp4_directrecv_moe_deepgemm_sparse_singlepass_20260529|stage_100ms_tp4pp4_directrecv_moe_cutlass_sparse_singlepass_20260529|stage_100ms_tp4pp4_directrecv_moe_flashinfer_cutlass_sparse_singlepass_20260529|stage_100ms_tp4pp4_directrecv_moe_flashinfer_trtllm_sparse_singlepass_20260529)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp4pp4_directrecv_finaldyn_decode32nw2_sparse_singlepass_20260528|stage_100ms_tp4pp4_directrecv_finaldyn_decode16w4_sparse_singlepass_20260529)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp4pp4_directrecv_mqa_m1_sparse_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp4pp4_noeager_batchinv_sparse_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp4pp4_directrecv_nccl_ll128_sparse_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp4pp4_directrecv_moe_triton_sparse_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp4pp4_moe_tn128_sparse_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp4pp4_sparse_bh32_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_100ms_tp8pp2_sparse_singlepass_20260528)
    export VLLM_PP_LAYER_PARTITION="38,40"
    TP_SIZE="8"
    PP_SIZE="2"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_final64_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_50ms_tp4pp4_part20202018_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_clean_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_PP_LAYER_PARTITION="20,20,20,18"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_50ms_pf4pp4_d2pp8_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_PP_LAYER_PARTITION="10,9,9,10,10,10,10,10"
      TP_SIZE="2"
      PP_SIZE="8"
    else
      export VLLM_PP_LAYER_PARTITION="19,20,19,20"
      TP_SIZE="4"
      PP_SIZE="4"
    fi
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_50ms_pf4pp4_d2pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    if [[ "${ROLE}" == "decode" ]]; then
      TP_SIZE="2"
    else
      TP_SIZE="4"
    fi
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage50_pf4pp4_dtp8pp2_mqabn512_ncclsimple_20260603)
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_PP_LAYER_PARTITION="38,40"
      TP_SIZE="8"
      PP_SIZE="2"
    else
      export VLLM_PP_LAYER_PARTITION="19,20,19,20"
      TP_SIZE="4"
      PP_SIZE="4"
    fi
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage50_pf4pp4_dtp4pp2_mqabn512_ncclsimple_20260604)
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_PP_LAYER_PARTITION="38,40"
      TP_SIZE="4"
      PP_SIZE="2"
    else
      export VLLM_PP_LAYER_PARTITION="19,20,19,20"
      TP_SIZE="4"
      PP_SIZE="4"
    fi
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage50_pf4pp4_dtp4pp3_mqabn512_ncclsimple_20260604)
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_PP_LAYER_PARTITION="26,26,26"
      TP_SIZE="4"
      PP_SIZE="3"
    else
      export VLLM_PP_LAYER_PARTITION="19,20,19,20"
      TP_SIZE="4"
      PP_SIZE="4"
    fi
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_splitkv16_empty_logits_skip_decode_clear_sparse_dsa_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_mqabn128_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk4w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache4_xdgfresh_pagedblk2w8s2_topkbranchcache_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_restore_stage70_source_trim8192_legacytopk_logits_workspace_reqbn256_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_clean_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache_xdgfresh_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache2_xdgfresh_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache_decodem1final64_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache_cpp_topk_envcache_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  *)
    export VLLM_PP_LAYER_PARTITION="10,9,9,10,10,10,10,10"
    TP_SIZE="2"
    PP_SIZE="8"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_cachealias_xdgold_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_decodefinal64role_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256p512d_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_moebm64_nofp32reduce_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_persistenttopk_trimblock_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_skipfinal_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_fullandpiecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_ppbatchp2p_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_dcp2_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_dcp2a2a_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn384_envcache5_xdgfresh_pagedblk2w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_mqabn512_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_mqabn512_split8_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_metacache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_moetn64x128_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk8w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_tp4pp4_mqabn512_predeqq_tb_20260603)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage50_tp4pp4_req512_mqabn512_tb_20260603)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache5_xdgfresh_pagedblk2w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache5_xdgfresh_pagedblk2w8s2_topkcache_histfusion_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_pppairp2p_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_mqabn512_mqastage2_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_hca1_ppsampleonly_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_pairp2p_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_firstp2p_middleskip_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_chunk262k_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_ppfastpath_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_replay_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_pairp2p_skipfinal_xdgfresh_20260604|stage50_mqabn512_ncclsimple_replay_xdgfresh_20260604|stage50_mqabn512_ncclsimple_mbt2048_xdgfresh_20260604|stage50_mqabn512_ncclsimple_maxseq8_xdgfresh_20260604|stage50_mqabn512_nccl_ll_xdgfresh_20260604|stage50_mqabn512_ncclsimple_chunk262k_20260603|stage50_mqabn512_ncclsimple_metacache_20260603|stage50_mqabn512_ncclsimple_tree_20260603|stage50_mqabn512_nccl_ll128_20260603|stage50_mqabn512_nccl_ll128_xdgfresh_20260603|stage50_mqabn512_ncclsimple_ch1_xdgfresh_20260603|stage50_mqabn512_ncclsimple_ch16_xdgfresh_20260603|stage50_mqabn512_ncclsimple_ch4_xdgfresh_20260603|stage50_mqabn512_ncclsimple_skipfinal_xdgfresh_20260603|stage50_mqabn512_ncclsimple_cpubcast_xdgfresh_20260603|stage50_mqabn512_ncclsimple_cpup2p_xdgfresh_20260604|stage50_mqabn512_ncclsimple_pairp2p_xdgfresh_20260604|stage50_mqabn512_ncclsimple_recvbuf_xdgfresh_20260603|stage50_mqabn512_ncclsimple_firstp2p_xdgfresh_20260603|stage50_mqabn512_ncclsimple_hca1_xdgfresh_20260603|stage50_mqabn512_ncclsimple_noasync_xdgfresh_20260603|stage50_mqabn512_ncclsimple_decodeswap_xdgfresh_20260603|stage50_mqabn512_ncclsimple_asyncbcast_xdgfresh_20260603|stage50_mqabn512_ncclsimple_middleskip_xdgfresh_20260603|stage50_stage70reqbn256_ncclsimple_xdgfresh_20260603)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_hca1_ppbatchp2p_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_prefixmask_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppsendwaitdefer_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_decodem1final64_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_earlybcast_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_asyncbcast_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_part20201919_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="20,20,19,19"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_part19202019_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,20,19"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_hca1_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppsampleonly_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppsampleonly_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_early_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppsampleonly_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_PP_BROADCAST_SAMPLED_TOKEN_ONLY="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppsampleonly_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_BROADCAST_SAMPLED_TOKEN_ONLY="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_reqbn512_envcache_replay_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_nccl_ll_pairp2p_xdgfresh_20260604|stage50_mqabn512_nccl_ll_skipfinal_xdgfresh_20260604|stage50_mqabn512_nccl_ll_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_predeqq_notrim_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac

case "${SCHEME}" in
  stage50_stage70reqbn256_ncclsimple_xdgfresh_20260603)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="256"
    fi
    ;;
esac

case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_decode_trim8192_sparse_singlepass_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    ;;
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_noasync_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_blockgrid_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="0"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_decodefinal64role_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_pf4pp4_d2pp8_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_pf4pp4_d2pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_persistenttopk_trimblock_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="persistent"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_skipfinal_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_blockgrid_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_decodefinal64role_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="1"
    else
      export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    fi
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256p512d_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="512"
    else
      export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    fi
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_moebm64_nofp32reduce_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_MARLIN_MOE_BLOCK_SIZE_M="64"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    ;;
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trimexact_legacytopk_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="0"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    ;;
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_persistenttopk_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="persistent"
    ;;
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim4096_legacytopk_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="4096"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    ;;
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_pagedblk2_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_combo_bench_off_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_moebm8_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_bh32_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_topkpad200k_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_finaldyn64_empty_logits_skip_decode_clear_sparse_singlepass_20260601|stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_topksplit4_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    ;;
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_final64_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    ;;
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_splitkvauto_empty_logits_skip_decode_clear_sparse_dsa_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_MLA_FORCE_KV_SPLITS=""
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_clean_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache_xdgfresh_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache2_xdgfresh_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_splitkv16_empty_logits_skip_decode_clear_sparse_dsa_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_mqabn128_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_fullandpiecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk4w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache4_xdgfresh_pagedblk2w8s2_topkbranchcache_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache_decodem1final64_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache_cpp_topk_envcache_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="512"
    ;;
esac
case "${SCHEME}" in
  stage50_reqbn512_envcache_replay_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS="1"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE="8192"
    export VLLM_SPARSE_INDEXER_DECODE_EMPTY_LOGITS="1"
    export VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE="1"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND="legacy"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="512"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_mqabn128_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="128"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_pf4pp4_d2pp8_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_pf4pp4_d2pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_splitkv16_empty_logits_skip_decode_clear_sparse_dsa_20260602)
    export VLLM_SPARSE_MLA_FORCE_KV_SPLITS="16"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602|stage_50ms_tp4pp4_directrecv_compile_fullandpiecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk2w8s2_skipfinal_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache3_xdgfresh_pagedblk4w8s2_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="4"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache4_xdgfresh_pagedblk2w8s2_topkbranchcache_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_ppbatchp2p_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_dcp2_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_dcp2a2a_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn384_envcache5_xdgfresh_pagedblk2w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_mqabn512_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_mqabn512_tileselect_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_metacache_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_moetn64x128_empty_logits_skip_decode_clear_sparse_singlepass_20260603|stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk8w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    if [[ "${SCHEME}" == *"_pagedblk8w8s2_"* ]]; then
      export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="8"
    fi
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    if [[ "${SCHEME}" == *"_reqbn384_"* ]]; then
      export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="384"
    fi
    if [[ "${SCHEME}" == *"_mqabn512_"* && "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    if [[ "${SCHEME}" == *"_tileselect_"* ]]; then
      export VLLM_SPARSE_INDEXER_DECODE_TOPK_TILE_SELECT="1"
      export VLLM_SPARSE_INDEXER_DECODE_TOPK_TILE_SELECT_CANDIDATES="16"
    fi
    if [[ "${SCHEME}" == *"_ppbatchp2p_"* ]]; then
      export VLLM_PP_BATCH_P2P_TENSOR_DICT="1"
    fi
    if [[ "${SCHEME}" == *"_metacache_"* ]]; then
      export VLLM_PP_TENSOR_DICT_METADATA_CACHE="1"
    fi
    if [[ "${SCHEME}" == *"_moetn64x128_"* ]]; then
      export VLLM_MARLIN_MOE_BLOCK_SIZE_M="48"
      export VLLM_MARLIN_USE_FP32_REDUCE="0"
      export VLLM_MARLIN_MOE_USE_FP32_REDUCE="1"
      export VLLM_MARLIN_MOE_THREAD_K="64"
      export VLLM_MARLIN_MOE_THREAD_N="128"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_tp4pp4_mqabn512_predeqq_tb_20260603)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    export VLLM_SPARSE_INDEXER_DECODE_PREDEQUANT_Q="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="1"
    ;;
  stage50_tp4pp4_req512_mqabn512_tb_20260603)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    export VLLM_SPARSE_INDEXER_DECODE_PREDEQUANT_Q="0"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="512"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_hca1_ppsampleonly_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_pairp2p_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_firstp2p_middleskip_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_chunk262k_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_ppfastpath_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_replay_xdgfresh_20260604|stage50_mqabn512_ncclsimple_hca1_pairp2p_skipfinal_xdgfresh_20260604|stage50_mqabn512_ncclsimple_replay_xdgfresh_20260604|stage50_mqabn512_ncclsimple_mbt2048_xdgfresh_20260604|stage50_mqabn512_ncclsimple_maxseq8_xdgfresh_20260604|stage50_mqabn512_nccl_ll_xdgfresh_20260604|stage50_mqabn512_ncclsimple_chunk262k_20260603|stage50_mqabn512_ncclsimple_metacache_20260603|stage50_mqabn512_ncclsimple_tree_20260603|stage50_mqabn512_nccl_ll128_20260603|stage50_mqabn512_nccl_ll128_xdgfresh_20260603|stage50_mqabn512_ncclsimple_ch1_xdgfresh_20260603|stage50_mqabn512_ncclsimple_ch16_xdgfresh_20260603|stage50_mqabn512_ncclsimple_ch4_xdgfresh_20260603|stage50_mqabn512_ncclsimple_skipfinal_xdgfresh_20260603|stage50_mqabn512_ncclsimple_cpubcast_xdgfresh_20260603|stage50_mqabn512_ncclsimple_cpup2p_xdgfresh_20260604|stage50_mqabn512_ncclsimple_pairp2p_xdgfresh_20260604|stage50_mqabn512_ncclsimple_recvbuf_xdgfresh_20260603|stage50_mqabn512_ncclsimple_firstp2p_xdgfresh_20260603|stage50_mqabn512_ncclsimple_hca1_xdgfresh_20260603|stage50_mqabn512_ncclsimple_noasync_xdgfresh_20260603|stage50_mqabn512_ncclsimple_decodeswap_xdgfresh_20260603|stage50_mqabn512_ncclsimple_asyncbcast_xdgfresh_20260603|stage50_mqabn512_ncclsimple_middleskip_xdgfresh_20260603|stage50_pf4pp4_dtp4pp3_mqabn512_ncclsimple_20260604|stage50_pf4pp4_dtp4pp2_mqabn512_ncclsimple_20260604|stage50_pf4pp4_dtp8pp2_mqabn512_ncclsimple_20260603)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_metacache_20260603" ]]; then
      export VLLM_PP_BATCH_P2P_TENSOR_DICT="0"
      export VLLM_PP_TENSOR_DICT_METADATA_CACHE="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_skipfinal_xdgfresh_20260603" ]]; then
      export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_cpubcast_xdgfresh_20260603" ]]; then
      export VLLM_PP_CPU_SAMPLED_TOKEN_BROADCAST="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_cpup2p_xdgfresh_20260604" ]]; then
      export VLLM_PP_CPU_FIRST_RANK_SAMPLED_TOKEN_P2P="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_pairp2p_xdgfresh_20260604" ]]; then
      export VLLM_PP_SAMPLED_TOKEN_PAIR_P2P="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_hca1_pairp2p_skipfinal_xdgfresh_20260604" ]]; then
      export VLLM_PP_SAMPLED_TOKEN_PAIR_P2P="1"
      export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_hca1_pairp2p_xdgfresh_20260604" ]]; then
      export VLLM_PP_SAMPLED_TOKEN_PAIR_P2P="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_hca1_ppsampleonly_xdgfresh_20260604" ]]; then
      export VLLM_PP_BROADCAST_SAMPLED_TOKEN_ONLY="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_hca1_ppbatchp2p_xdgfresh_20260604" ]]; then
      export VLLM_PP_BATCH_P2P_TENSOR_DICT="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_recvbuf_xdgfresh_20260603" ]]; then
      export VLLM_PP_SAMPLED_TOKEN_RECV_BUFFER="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_firstp2p_xdgfresh_20260603" ]]; then
      export VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_asyncbcast_xdgfresh_20260603" ]]; then
      export VLLM_PP_ASYNC_SAMPLED_TOKEN_BROADCAST="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_middleskip_xdgfresh_20260603" ]]; then
      export VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P="1"
      export VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_hca1_firstp2p_middleskip_xdgfresh_20260604" ]]; then
      export VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P="1"
      export VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN="1"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_cachealias_xdgold_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_prefixmask_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="1"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE_MAX_TOKENS="4"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppsendwaitdefer_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_PP_DEFER_SEND_WAIT="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_decodem1final64_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
      export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="1"
    else
      export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="0"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_hca1_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="0"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260604|stage50_mqabn512_ncclsimple_skipfinal_noallgather_tree_xdgfresh_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_cachefresh_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_noautobench_20260605|stage50_mqabn512_ncclsimple_persistenttopk_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_fulldecode_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_fulldecode_ppclone_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_fulldecode_ppcopybuf_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_inline_sample_broadcast_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_ppsendwaitdefer_xdgfresh_20260605|stage50_mqabn512_ncclsimple_dvtile128bh8_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_decodeprepfast_skipfinal_noallgather_xdgfresh_20260605|stage50_mqabn512_ncclsimple_deferrecv_skipfinal_noallgather_xdgfresh_20260606|stage50_ppqueue6_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260606|stage50_ppqueue6_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_recoveryfresh_20260606|stage50_pf4pp4_dtp4pp3_mqfix_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260605)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_skipfinal_noallgather_ppsendwaitdefer_xdgfresh_20260605" ]]; then
      export VLLM_PP_DEFER_SEND_WAIT="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_fulldecode_ppclone_skipfinal_noallgather_xdgfresh_20260605" ]]; then
      export VLLM_PP_CLONE_CUDAGRAPH_OUTPUT_BEFORE_SEND="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_decodeprepfast_skipfinal_noallgather_xdgfresh_20260605" ]]; then
      export VLLM_STAGE50_DECODE_PREP_FASTPATH="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_deferrecv_skipfinal_noallgather_xdgfresh_20260606" ]]; then
      export VLLM_PP_DEFER_INTERMEDIATE_RECV="1"
    fi
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_dvtile128bh8_skipfinal_noallgather_xdgfresh_20260605" && "${ROLE}" == "decode" ]]; then
      export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_FINAL="1"
      export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE="128"
      export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_BLOCK_H="8"
      export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_BLOCK_N="32"
      export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_NUM_WARPS="4"
      export VLLM_SPARSE_MLA_DECODE_M1_DV_TILE_NUM_STAGES="1"
    fi
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh2_20260605|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh3_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh4_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh5_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh6_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh7_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh8_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh9_20260606)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh10_20260606|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh11_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh12_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh13_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh14_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh15_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh16_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh17_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh20_combobenchoff_20260609|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_recoveryfresh21_after_coreopxid_20260610|stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_fp8lut_20260610|stage50_mqabn512_ncclsimple_coopfinal_graphsafe_skipfinal_noallgather_xdgfresh_20260609|stage50_mqabn512_ncclsimple_splitmergefinal_decodeonly_skipfinal_noallgather_xdgfresh_20260609)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_fp8lut_20260610)
    export VLLM_SPARSE_INDEXER_DECODE_FP8_LUT="1"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_restore2_fp8lutfresh1_20260610)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_SPARSE_INDEXER_DECODE_FP8_LUT="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_pf4pp4_dtp4pp3_mqfix_highmem093_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260605)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_pf4pp4_dtp4pp3_mqfix_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260605|stage50_pf4pp4_dtp4pp3_mqfix_highmem093_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260605)
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_PP_LAYER_PARTITION="26,26,26"
      TP_SIZE="4"
      PP_SIZE="3"
    else
      export VLLM_PP_LAYER_PARTITION="19,20,19,20"
      TP_SIZE="4"
      PP_SIZE="4"
    fi
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    ;;
esac
case "${SCHEME}" in
  stage50_tp4pp3_mqabn512_ncclsimple_homopp_highmem093_skipfinal_noallgather_xdgfresh_20260606)
    export VLLM_PP_LAYER_PARTITION="26,26,26"
    TP_SIZE="4"
    PP_SIZE="3"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_pf4pp4_dtp16pp1_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260606|stage50_pf4pp4_dtp16pp1_highmem093_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260606)
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_PP_LAYER_PARTITION="78"
      TP_SIZE="16"
      PP_SIZE="1"
    else
      export VLLM_PP_LAYER_PARTITION="19,20,19,20"
      TP_SIZE="4"
      PP_SIZE="4"
    fi
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_fulldecode_strongout_skipfinal_noallgather_xdgfresh_20260608)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_FULL_CUDAGRAPH_STRONG_OUTPUT="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_splitkv2_skipfinal_noallgather_xdgfresh_20260608)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_SPARSE_MLA_FORCE_KV_SPLITS="2"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_splitkv4_skipfinal_noallgather_xdgfresh_20260608)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_SPARSE_MLA_FORCE_KV_SPLITS="4"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_splitkv16_skipfinal_noallgather_xdgfresh_20260608|stage50_mqabn512_ncclsimple_splitkv16_skipfinal_noallgather_xdgfresh_formalfresh_20260608|stage50_mqabn512_ncclsimple_splitkv16_noautobench_skipfinal_noallgather_xdgfresh_20260608)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_SPARSE_MLA_FORCE_KV_SPLITS="16"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_inductorpart_splitops_skipfinal_noallgather_xdgfresh_20260608|stage50_mqabn512_ncclsimple_inductorpart_fxsplit_skipfinal_noallgather_xdgfresh_20260608)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_SPARSE_MLA_FORCE_KV_SPLITS="1"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${SCHEME}" == "stage50_mqabn512_ncclsimple_inductorpart_fxsplit_skipfinal_noallgather_xdgfresh_20260608" ]]; then
      export VLLM_INDUCTOR_PARTITION_KEEP_FX_SPLIT_OPS="1"
    fi
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_readyfirst_skipfinal_noallgather_xdgfresh_20260609)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_SPARSE_MLA_FORCE_KV_SPLITS="1"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_STAGE50_PP_READY_FIRST_DRAIN="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_ppqueue2exact_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260609)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_tp16pp1_mqabn512_ncclsimple_homopp_highmem093_skipfinal_noallgather_xdgfresh_20260606|stage50_tp16pp1_mqabn512_ncclsimple_homopp_highmem0925_skipfinal_noallgather_xdgfresh_20260606|stage50_tp16pp1_mqabn512_ncclsimple_homopp_highmem0925_formalfresh_skipfinal_noallgather_xdgfresh_20260606)
    export VLLM_PP_LAYER_PARTITION="78"
    TP_SIZE="16"
    PP_SIZE="1"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_ppbatchp2p_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_BATCH_P2P_TENSOR_DICT="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_early_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P="1"
    export VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN="1"
    export VLLM_PP_EARLY_SAMPLED_TOKEN_BROADCAST="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_earlybcast_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_EARLY_SAMPLED_TOKEN_BROADCAST="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_part20201919_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_decodemeta_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_DECODE_TENSOR_DICT_METADATA_SKIP="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P="1"
    export VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_decodemeta_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P="1"
    export VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN="1"
    export VLLM_PP_DECODE_TENSOR_DICT_METADATA_SKIP="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_decodemeta_ppbatchp2p_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P="1"
    export VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN="1"
    export VLLM_PP_DECODE_TENSOR_DICT_METADATA_SKIP="1"
    export VLLM_PP_BATCH_P2P_TENSOR_DICT="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_pairp2p_middleskip_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_SAMPLED_TOKEN_PAIR_P2P="1"
    export VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp2pp8_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260604|stage_50ms_tp2pp8_mqabn512_ncclsimple_firstp2p_middleskip_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="10,9,9,10,10,10,10,10"
    TP_SIZE="2"
    PP_SIZE="8"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
  stage_50ms_tp2pp8_part10101010101099_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="10,10,10,10,10,10,9,9"
    TP_SIZE="2"
    PP_SIZE="8"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
  stage_50ms_tp2pp8_part101010101010108_mqabn512_ncclsimple_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="10,10,10,10,10,10,10,8"
    TP_SIZE="2"
    PP_SIZE="8"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_part19202019_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_asyncbcast_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_ASYNC_SAMPLED_TOKEN_BROADCAST="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_ncclsimple_firstp2p_middleskip_metacache_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_PP_LAYER_PARTITION="19,20,19,20"
    TP_SIZE="4"
    PP_SIZE="4"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="4096"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE="0"
    export VLLM_SPARSE_MLA_DECODE_M1_FINAL64="0"
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    export VLLM_PP_FIRST_RANK_ONLY_SAMPLED_TOKEN_P2P="1"
    export VLLM_PP_MIDDLE_RANK_SKIP_SAMPLED_TOKEN="1"
    export VLLM_PP_TENSOR_DICT_METADATA_CACHE="1"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_nccl_ll_pairp2p_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    export VLLM_PP_SAMPLED_TOKEN_PAIR_P2P="1"
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_nccl_ll_skipfinal_xdgfresh_20260604|stage50_mqabn512_nccl_ll_skipfinal_noallgather_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    export VLLM_PP_SKIP_FINAL_MAX_TOKENS_BROADCAST="1"
    if [[ "${SCHEME}" == "stage50_mqabn512_nccl_ll_skipfinal_noallgather_xdgfresh_20260604" ]]; then
      export VLLM_PP_DISABLE_INTERMEDIATE_ALLGATHER="1"
    fi
    ;;
esac
case "${SCHEME}" in
  stage50_mqabn512_predeqq_notrim_xdgfresh_20260604)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    if [[ "${ROLE}" == "decode" ]]; then
      export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="512"
    fi
    export VLLM_SPARSE_INDEXER_DECODE_PREDEQUANT_Q="1"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn256_envcache5_xdgfresh_pagedblk2w8s2_topkcache_pppairp2p_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="256"
    export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="256"
    export VLLM_PP_BATCH_P2P_TENSOR_DICT="1"
    export VLLM_PP_TENSOR_DICT_METADATA_CACHE="0"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_topksplit4_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_TOPK_DECODE_SPLIT_THRESHOLD="32768"
    export VLLM_TOPK_DECODE_SPLIT_BLOCKS="4"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="128"
    ;;
esac
case "${SCHEME}" in
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache5_xdgfresh_pagedblk2w8s2_topkcache_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="256"
    export VLLM_SPARSE_INDEXER_DECODE_PREDEQUANT_Q="0"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="512"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache5_xdgfresh_pagedblk2w8s2_topkcache_histfusion_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="256"
    export VLLM_SPARSE_INDEXER_DECODE_PREDEQUANT_Q="0"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_HIST_FUSION="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="512"
    ;;
  stage_50ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_reqbn512_envcache5_xdgfresh_pagedblk2w8s2_topkcache_binfusion_empty_logits_skip_decode_clear_sparse_singlepass_20260603)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_WARPS="8"
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES_STAGES="2"
    export VLLM_TOPK_ENV_CACHE="1"
    export VLLM_MQA_CUDA_V7_FUSED_TRITON_DECODE_BLOCK_N="256"
    export VLLM_SPARSE_INDEXER_DECODE_PREDEQUANT_Q="0"
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="0"
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_BIN_FUSION="1"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="512"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_finaldyn64_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_MLA_FINAL_STATIC_BY_TOKENS="0"
    export VLLM_SPARSE_MLA_FINAL_DYNAMIC_CONFIG="1"
    export VLLM_SPARSE_MLA_FINAL_DYNAMIC_SMALL_CONFIG="64,4,1"
    export VLLM_SPARSE_MLA_FINAL_DYNAMIC_MID_CONFIG="16,2,1"
    export VLLM_SPARSE_MLA_FINAL_DYNAMIC_LARGE_CONFIG="16,2,2"
    ;;
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_final64_empty_logits_skip_decode_clear_sparse_singlepass_20260602)
    export VLLM_SPARSE_MLA_FINAL_STATIC_BY_TOKENS="0"
    export VLLM_SPARSE_MLA_FINAL_CONFIG="64,4,1"
    export VLLM_SPARSE_MLA_FINAL_DYNAMIC_CONFIG="0"
    export VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N="128"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_bh32_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_MLA_BLOCK_H="32"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_topkpad200k_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_TOPK_PAD_LOGITS_LEN="200000"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_pagedblk2_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES="2"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_trim8192_legacytopk_logits_workspace_moebm8_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_MARLIN_MOE_BLOCK_SIZE_M="8"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_blockgrid_empty_logits_skip_decode_clear_sparse_singlepass_20260601)
    export VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE="1"
    ;;
esac
case "${SCHEME}" in
  stage_70ms_tp4pp4_directrecv_compile_piecewise_copyinputs_p16d16_noinductorautotune_noasync_skip_decode_clear_sparse_singlepass_20260601|stage50_mqabn512_ncclsimple_noasync_xdgfresh_20260603)
    if [[ " ${EXTRA_SERVE_ARGS:-} " != *" --no-async-scheduling "* ]]; then
      EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-} --no-async-scheduling"
      export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS# }"
    fi
    ;;
esac
export VLLM_ENGINE_READY_TIMEOUT_S="14400"
export NCCL_DEBUG="INFO"
export NCCL_IB_DISABLE="0"
export GLOO_SOCKET_IFNAME="${IFACE:-ens22f0}"
export NCCL_SOCKET_IFNAME="${IFACE:-ens22f0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2}"
if [[ -n "${NCCL_P2P_DISABLE:-}" ]]; then export NCCL_P2P_DISABLE; fi
if [[ -n "${NCCL_CUMEM_ENABLE:-}" ]]; then export NCCL_CUMEM_ENABLE; fi
if [[ -n "${NCCL_CUMEM_HOST_ENABLE:-}" ]]; then export NCCL_CUMEM_HOST_ENABLE; fi
if [[ -n "${NCCL_NVLS_ENABLE:-}" ]]; then export NCCL_NVLS_ENABLE; fi
if [[ -n "${NCCL_NET_GDR_LEVEL:-}" ]]; then export NCCL_NET_GDR_LEVEL; fi
if [[ -n "${NCCL_NET_GDR_READ:-}" ]]; then export NCCL_NET_GDR_READ; fi
if [[ -n "${NCCL_DMABUF_ENABLE:-}" ]]; then export NCCL_DMABUF_ENABLE; fi
if [[ -n "${NCCL_PXN_DISABLE:-}" ]]; then export NCCL_PXN_DISABLE; fi
export UCX_TLS="${UCX_TLS:-tcp}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-${UCX_DEVICES:-ens22f0}}"
if [[ -n "${UCX_MAX_RNDV_RAILS:-}" ]]; then export UCX_MAX_RNDV_RAILS; fi
if [[ -n "${UCX_MAX_EAGER_RAILS:-}" ]]; then export UCX_MAX_EAGER_RAILS; fi
if [[ -n "${UCX_CM_USE_ALL_DEVICES:-}" ]]; then export UCX_CM_USE_ALL_DEVICES; fi

unset VLLM_FP8_MOE_DEQUANT_BF16
unset VLLM_FP8_MLP_LINEAR_DEQUANT_BF16
unset VLLM_FP8_INDEXER_LINEAR_DEQUANT_BF16
unset VLLM_FP8_LM_HEAD_DEQUANT_BF16
unset VLLM_NVFP4_GEMM_BACKEND
if [[ "${VLLM_FP8_MLP_LINEAR_DEQUANT_BF16_ENABLE:-0}" == "1" ]]; then export VLLM_FP8_MLP_LINEAR_DEQUANT_BF16=1; fi
if [[ "${VLLM_FP8_INDEXER_LINEAR_DEQUANT_BF16_ENABLE:-0}" == "1" ]]; then export VLLM_FP8_INDEXER_LINEAR_DEQUANT_BF16=1; fi
if [[ "${VLLM_FP8_LM_HEAD_DEQUANT_BF16_ENABLE:-0}" == "1" ]]; then export VLLM_FP8_LM_HEAD_DEQUANT_BF16=1; fi

PREFILL_HEAD="${PREFILL_HEAD:-${PREFILL0_HEAD:-127.0.0.1}}"
PREFILL_WORKER="${PREFILL_WORKER:-${PREFILL0_WORKER:-127.0.0.1}}"
DECODE_HEAD="${DECODE_HEAD:-127.0.0.1}"
DECODE_WORKER="${DECODE_WORKER:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-19182}"
DECODE_PORT="${DECODE_PORT:-19183}"
PROXY_PORT="${PROXY_PORT:-19181}"
PREFILL_MASTER_PORT="${PREFILL_MASTER_PORT:-32366}"
DECODE_MASTER_PORT="${DECODE_MASTER_PORT:-32354}"
SIDE_PORT="${SIDE_PORT:-25700}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-${MAX_NUM_SEQS}}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-${MAX_NUM_SEQS}}"
SERVER_SEED="${SERVER_SEED:-42}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${DEFAULT_MAX_NUM_BATCHED_TOKENS}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-${GPU_MEMORY_UTILIZATION}}"
DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-${GPU_MEMORY_UTILIZATION}}"
if [[ -n "${ROLE_TP_SIZE:-}" ]]; then TP_SIZE="${ROLE_TP_SIZE}"; fi
if [[ -n "${ROLE_PP_SIZE:-}" ]]; then PP_SIZE="${ROLE_PP_SIZE}"; fi
if [[ -n "${ROLE_LAYER_PARTITION:-}" ]]; then export VLLM_PP_LAYER_PARTITION="${ROLE_LAYER_PARTITION}"; fi
UBATCH_SIZE="${UBATCH_SIZE:-4}"
KV_BUFFER_DEVICE="${KV_BUFFER_DEVICE:-cpu}"
HYBRID_KV_ARG="${HYBRID_KV_ARG:-}"
NIXL_ENABLE_CROSS_LAYER_BLOCKS="${NIXL_ENABLE_CROSS_LAYER_BLOCKS:-False}"
NIXL_NUM_THREADS="${NIXL_NUM_THREADS:-4}"
NIXL_SIMPLE_EXTRA_CONFIG="${NIXL_SIMPLE_EXTRA_CONFIG:-False}"
COMPILATION_CONFIG_JSON="${COMPILATION_CONFIG_JSON:-}"
PROFILER_CONFIG_JSON="${PROFILER_CONFIG_JSON:-}"
if [[ -z "${PROFILER_CONFIG_JSON}" && -n "${PROFILER_CONFIG_B64:-}" ]]; then
  PROFILER_CONFIG_JSON="$(printf '%s' "${PROFILER_CONFIG_B64}" | base64 -d)"
fi
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
PREFILL_EXTRA_SERVE_ARGS="${PREFILL_EXTRA_SERVE_ARGS:-}"
DECODE_EXTRA_SERVE_ARGS="${DECODE_EXTRA_SERVE_ARGS:-}"
ROLE_LOG_SUFFIX="${ROLE_LOG_SUFFIX:-}"
DCP_SIZE="${DCP_SIZE:-1}"
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-}"
DCP_KV_CACHE_INTERLEAVE_SIZE="${DCP_KV_CACHE_INTERLEAVE_SIZE:-}"
MOE_BACKEND="${MOE_BACKEND:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"

KV_CONNECTOR="${KV_CONNECTOR:-NixlConnector}"
KV_LOAD_FAILURE_POLICY="${KV_LOAD_FAILURE_POLICY:-recompute}"
P2P_PROXY_IP="${P2P_PROXY_IP:-${DECODE_HEAD}}"
P2P_PROXY_PORT="${P2P_PROXY_PORT:-30001}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-21001}"
DECODE_KV_PORT="${DECODE_KV_PORT:-22001}"
P2P_SEND_TYPE="${P2P_SEND_TYPE:-PUT_ASYNC}"
P2P_NCCL_NUM_CHANNELS="${P2P_NCCL_NUM_CHANNELS:-16}"
P2P_PREFILL_BUFFER_SIZE="${P2P_PREFILL_BUFFER_SIZE:-1e1}"
P2P_DECODE_BUFFER_SIZE="${P2P_DECODE_BUFFER_SIZE:-8e9}"
PREFILL_ENGINE_ID="${PREFILL_ENGINE_ID:-fp8-1p1d-v1-prefill}"
DECODE_ENGINE_ID="${DECODE_ENGINE_ID:-fp8-1p1d-v1-decode}"

stage_nsys_role_enabled() {
  local role_name="$1"
  [[ "${VLLM_STAGE_NSYS_LAUNCH}" == "1" ]] || return 1
  local item
  IFS=',' read -r -a role_items <<< "${VLLM_STAGE_NSYS_ROLES}"
  for item in "${role_items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ "${item}" == "all" || "${item}" == "${role_name}" || "${item}" == "${ROLE}" ]]; then
      return 0
    fi
  done
  return 1
}

stage_nsys_session_name() {
  local role_name="$1"
  printf '%s_%s_rank%s' "${VLLM_STAGE_NSYS_SESSION_PREFIX}" "${role_name}" "${NODE_RANK}"
}

stage_nsys_exec() {
  local role_name="$1"
  local log="$2"
  shift 2
  if stage_nsys_role_enabled "${role_name}"; then
    mkdir -p "${VLLM_STAGE_NSYS_OUT_DIR}"
    local session_name
    session_name="$(stage_nsys_session_name "${role_name}")"
    local extra_args=()
    if [[ -n "${VLLM_STAGE_NSYS_LAUNCH_EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      extra_args=(${VLLM_STAGE_NSYS_LAUNCH_EXTRA_ARGS})
    fi
    exec "${VLLM_STAGE_NSYS_BIN}" launch \
      --session-new "${session_name}" \
      --trace="${VLLM_STAGE_NSYS_TRACE}" \
      --trace-fork-before-exec=true \
      --cuda-memory-usage="${VLLM_STAGE_NSYS_CUDA_MEMORY}" \
      --wait=primary \
      --show-output=true \
      "${extra_args[@]}" \
      "$@" > "${log}" 2>&1
  fi
  exec "$@" > "${log}" 2>&1
}

prefix_cache_arg() {
  case "${VLLM_ENABLE_PREFIX_CACHING:-1}" in
    1|true|TRUE|yes|YES|on|ON)
      printf '%s\n' "--enable-prefix-caching"
      ;;
    0|false|FALSE|no|NO|off|OFF)
      printf '%s\n' "--no-enable-prefix-caching"
      ;;
    *)
      echo "invalid VLLM_ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING}" >&2
      exit 2
      ;;
  esac
}

prompt_tokens_details_arg() {
  case "${VLLM_ENABLE_PROMPT_TOKENS_DETAILS:-1}" in
    1|true|TRUE|yes|YES|on|ON)
      printf '%s\n' "--enable-prompt-tokens-details"
      ;;
    0|false|FALSE|no|NO|off|OFF)
      return 0
      ;;
    *)
      echo "invalid VLLM_ENABLE_PROMPT_TOKENS_DETAILS=${VLLM_ENABLE_PROMPT_TOKENS_DETAILS}" >&2
      exit 2
      ;;
  esac
}

run_serve() {
  local role_name="$1" port="$2" master_addr="$3" master_port="$4" kv_role="$5" headless_arg="${6:-}"
  local engine_id="${PREFILL_ENGINE_ID}"
  local role_max_num_seqs="${PREFILL_MAX_NUM_SEQS}"
  local role_gpu_memory_utilization="${PREFILL_GPU_MEMORY_UTILIZATION}"
  local role_extra_serve_args="${EXTRA_SERVE_ARGS}"
  local role_nnodes="${ROLE_NNODES:-2}"
  if [[ "${role_name}" == "prefill" ]]; then role_nnodes="${PREFILL_NNODES:-${role_nnodes}}"; fi
  if [[ "${role_name}" == "decode" ]]; then role_nnodes="${DECODE_NNODES:-${role_nnodes}}"; fi
  if [[ "${role_name}" == "decode" ]]; then engine_id="${DECODE_ENGINE_ID}"; fi
  if [[ "${role_name}" == "decode" ]]; then role_max_num_seqs="${DECODE_MAX_NUM_SEQS}"; fi
  if [[ "${role_name}" == "decode" ]]; then role_gpu_memory_utilization="${DECODE_GPU_MEMORY_UTILIZATION}"; fi
  if [[ "${role_name}" == "prefill" && -n "${PREFILL_EXTRA_SERVE_ARGS}" ]]; then role_extra_serve_args="${PREFILL_EXTRA_SERVE_ARGS}"; fi
  if [[ "${role_name}" == "decode" && -n "${DECODE_EXTRA_SERVE_ARGS}" ]]; then role_extra_serve_args="${DECODE_EXTRA_SERVE_ARGS}"; fi
  local log_role="${role_name}${ROLE_LOG_SUFFIX:+_${ROLE_LOG_SUFFIX}}"
  local log="${LOG_DIR}/${TS}_${log_role}_rank${NODE_RANK}.log"
  local compilation_config_arg=()
  if [[ -n "${COMPILATION_CONFIG_JSON}" ]]; then
    compilation_config_arg=(--compilation-config "${COMPILATION_CONFIG_JSON}")
  fi
  local profiler_config_arg=()
  if [[ -n "${PROFILER_CONFIG_JSON}" ]]; then
    profiler_config_arg=(--profiler-config "${PROFILER_CONFIG_JSON}")
  fi
  local dcp_arg=()
  if [[ -n "${DCP_SIZE}" && "${DCP_SIZE}" != "1" ]]; then
    dcp_arg=(--decode-context-parallel-size "${DCP_SIZE}")
    if [[ -n "${DCP_COMM_BACKEND}" ]]; then
      dcp_arg+=(--dcp-comm-backend "${DCP_COMM_BACKEND}")
    fi
    if [[ -n "${DCP_KV_CACHE_INTERLEAVE_SIZE}" ]]; then
      dcp_arg+=(--dcp-kv-cache-interleave-size "${DCP_KV_CACHE_INTERLEAVE_SIZE}")
    fi
  fi
  local enforce_eager_arg=()
  if [[ "${VLLM_ENFORCE_EAGER}" != "0" ]]; then
    enforce_eager_arg=(--enforce-eager)
  fi
  local custom_all_reduce_arg=()
  if [[ "${VLLM_DISABLE_CUSTOM_ALL_REDUCE}" != "0" ]]; then
    custom_all_reduce_arg=(--disable-custom-all-reduce)
  fi
  local moe_backend_arg=()
  if [[ -n "${MOE_BACKEND}" ]]; then
    moe_backend_arg=(--moe-backend "${MOE_BACKEND}")
  fi
  local prefix_cache_arg_value
  prefix_cache_arg_value="$(prefix_cache_arg)"
  local prompt_tokens_details_arg_value
  prompt_tokens_details_arg_value="$(prompt_tokens_details_arg)"
  local kv_config
  if [[ "${KV_CONNECTOR}" == "P2pNcclConnector" ]]; then
    local p2p_kv_port="${DECODE_KV_PORT}"
    local p2p_buffer_size="${P2P_DECODE_BUFFER_SIZE}"
    if [[ "${role_name}" == "prefill" ]]; then
      p2p_kv_port="${PREFILL_KV_PORT}"
      p2p_buffer_size="${P2P_PREFILL_BUFFER_SIZE}"
    fi
    kv_config='{"kv_connector":"P2pNcclConnector","engine_id":"'"${engine_id}"'","kv_role":"'"${kv_role}"'","kv_load_failure_policy":"'"${KV_LOAD_FAILURE_POLICY}"'","kv_buffer_size":"'"${p2p_buffer_size}"'","kv_port":"'"${p2p_kv_port}"'","kv_connector_extra_config":{"proxy_ip":"'"${P2P_PROXY_IP}"'","proxy_port":"'"${P2P_PROXY_PORT}"'","http_port":"'"${port}"'","send_type":"'"${P2P_SEND_TYPE}"'","nccl_num_channels":"'"${P2P_NCCL_NUM_CHANNELS}"'"}}'
  else
    if [[ "${NIXL_SIMPLE_EXTRA_CONFIG,,}" == "true" || "${NIXL_SIMPLE_EXTRA_CONFIG}" == "1" ]]; then
      kv_config='{"kv_connector":"NixlConnector","engine_id":"'"${engine_id}"'","kv_role":"'"${kv_role}"'","kv_load_failure_policy":"'"${KV_LOAD_FAILURE_POLICY}"'","kv_buffer_device":"'"${KV_BUFFER_DEVICE}"'","kv_connector_extra_config":{"backends":["UCX"]}}'
    else
      kv_config='{"kv_connector":"NixlConnector","engine_id":"'"${engine_id}"'","kv_role":"'"${kv_role}"'","kv_load_failure_policy":"'"${KV_LOAD_FAILURE_POLICY}"'","kv_buffer_device":"'"${KV_BUFFER_DEVICE}"'","kv_connector_extra_config":{"backends":["UCX"],"enable_cross_layers_blocks":"'"${NIXL_ENABLE_CROSS_LAYER_BLOCKS}"'","num_threads":'"${NIXL_NUM_THREADS}"'}}'
    fi
  fi
  if [[ "${role_name}" == "prefill" ]]; then
    if [[ "${NODE_RANK}" == "0" ]]; then export VLLM_HOST_IP="${PREFILL_HEAD}"; else export VLLM_HOST_IP="${PREFILL_WORKER}"; fi
  else
    if [[ "${NODE_RANK}" == "0" ]]; then export VLLM_HOST_IP="${DECODE_HEAD}"; else export VLLM_HOST_IP="${DECODE_WORKER}"; fi
  fi
  export VLLM_NIXL_SIDE_CHANNEL_HOST="${VLLM_HOST_IP}"
  export VLLM_NIXL_SIDE_CHANNEL_PORT="${SIDE_PORT}"
  stage_nsys_exec "${role_name}" "${log}" "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
    --served-model-name "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --distributed-executor-backend mp \
    --nnodes "${role_nnodes}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${master_addr}" \
    --master-port "${master_port}" \
    ${headless_arg} \
    --distributed-timeout-seconds 14400 \
    --trust-remote-code \
    --seed "${SERVER_SEED}" \
    --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}" \
    --gpu-memory-utilization "${role_gpu_memory_utilization}" \
    --max-model-len 202752 \
    --max-num-seqs "${role_max_num_seqs}" \
    "${prefix_cache_arg_value}" \
    ${prompt_tokens_details_arg_value:+"${prompt_tokens_details_arg_value}"} \
    --enable-chunked-prefill \
    ${HYBRID_KV_ARG} \
    --performance-mode interactivity \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    "${custom_all_reduce_arg[@]}" \
    --all2all-backend deepep_low_latency \
    --ubatch-size "${UBATCH_SIZE}" \
    "${enforce_eager_arg[@]}" \
    "${compilation_config_arg[@]}" \
    "${profiler_config_arg[@]}" \
    "${dcp_arg[@]}" \
    "${moe_backend_arg[@]}" \
    ${role_extra_serve_args} \
    --kv-transfer-config "${kv_config}"
}

run_nonpd() {
  local headless_arg="${1:-}"
  local log_role="nonpd${ROLE_LOG_SUFFIX:+_${ROLE_LOG_SUFFIX}}"
  local log="${LOG_DIR}/${TS}_${log_role}_rank${NODE_RANK}.log"
  local compilation_config_arg=()
  if [[ -n "${COMPILATION_CONFIG_JSON}" ]]; then
    compilation_config_arg=(--compilation-config "${COMPILATION_CONFIG_JSON}")
  fi
  local profiler_config_arg=()
  if [[ -n "${PROFILER_CONFIG_JSON}" ]]; then
    profiler_config_arg=(--profiler-config "${PROFILER_CONFIG_JSON}")
  fi
  local dcp_arg=()
  if [[ -n "${DCP_SIZE}" && "${DCP_SIZE}" != "1" ]]; then
    dcp_arg=(--decode-context-parallel-size "${DCP_SIZE}")
    if [[ -n "${DCP_COMM_BACKEND}" ]]; then
      dcp_arg+=(--dcp-comm-backend "${DCP_COMM_BACKEND}")
    fi
    if [[ -n "${DCP_KV_CACHE_INTERLEAVE_SIZE}" ]]; then
      dcp_arg+=(--dcp-kv-cache-interleave-size "${DCP_KV_CACHE_INTERLEAVE_SIZE}")
    fi
  fi
  local moe_backend_arg=()
  if [[ -n "${MOE_BACKEND}" ]]; then
    moe_backend_arg=(--moe-backend "${MOE_BACKEND}")
  fi
  local prefix_cache_arg_value
  prefix_cache_arg_value="$(prefix_cache_arg)"
  local prompt_tokens_details_arg_value
  prompt_tokens_details_arg_value="$(prompt_tokens_details_arg)"
  if [[ "${NODE_RANK}" == "0" ]]; then
    export VLLM_HOST_IP="${DECODE_HEAD}"
  else
    export VLLM_HOST_IP="${DECODE_WORKER}"
  fi
  exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
    --served-model-name "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${DECODE_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --distributed-executor-backend mp \
    --nnodes 2 \
    --node-rank "${NODE_RANK}" \
    --master-addr "${DECODE_HEAD}" \
    --master-port "${DECODE_MASTER_PORT}" \
    ${headless_arg} \
    --distributed-timeout-seconds 14400 \
    --trust-remote-code \
    --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len 202752 \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    "${prefix_cache_arg_value}" \
    ${prompt_tokens_details_arg_value:+"${prompt_tokens_details_arg_value}"} \
    --enable-chunked-prefill \
    ${HYBRID_KV_ARG} \
    --performance-mode interactivity \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --disable-custom-all-reduce \
    --all2all-backend deepep_low_latency \
    --ubatch-size "${UBATCH_SIZE}" \
    --enforce-eager \
    "${compilation_config_arg[@]}" \
    "${profiler_config_arg[@]}" \
    "${dcp_arg[@]}" \
    "${moe_backend_arg[@]}" \
    ${EXTRA_SERVE_ARGS} \
    > "${log}" 2>&1
}

case "${ROLE}" in
  nonpd)
    extra=""; if [[ "${NODE_RANK}" != "0" ]]; then extra="--headless"; fi
    run_nonpd "${extra}"
    ;;
  prefill)
    extra=""; if [[ "${NODE_RANK}" != "0" ]]; then extra="--headless"; fi
    run_serve "prefill" "${PREFILL_PORT}" "${PREFILL_HEAD}" "${PREFILL_MASTER_PORT}" "kv_producer" "${extra}"
    ;;
  decode)
    extra=""; if [[ "${NODE_RANK}" != "0" ]]; then extra="--headless"; fi
    run_serve "decode" "${DECODE_PORT}" "${DECODE_HEAD}" "${DECODE_MASTER_PORT}" "kv_consumer" "${extra}"
    ;;
  proxy)
    proxy_log="${LOG_DIR}/${TS}_proxy${ROLE_LOG_SUFFIX:+_${ROLE_LOG_SUFFIX}}.log"
    if [[ "${KV_CONNECTOR}" == "P2pNcclConnector" ]]; then
      exec env P2P_PROXY_DISCOVERY_HOST=0.0.0.0 P2P_PROXY_DISCOVERY_PORT="${P2P_PROXY_PORT}" P2P_PROXY_HTTP_HOST=0.0.0.0 P2P_PROXY_HTTP_PORT="${PROXY_PORT}" \
        "${PYTHON_BIN}" "${SOURCE_DIR}/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py" \
        > "${proxy_log}" 2>&1
    else
      prefill_host_args=(--prefiller-host "${PREFILL_HEAD}")
      prefill_port_args=(--prefiller-port "${PREFILL_PORT}")
      decode_host_args=(--decoder-host "${DECODE_HEAD}")
      decode_port_args=(--decoder-port "${DECODE_PORT}")
      if [[ -n "${PREFILL_HOSTS:-}" ]]; then
        # shellcheck disable=SC2206
        prefill_hosts=(${PREFILL_HOSTS})
        prefill_host_args=(--prefiller-hosts "${prefill_hosts[@]}")
      fi
      if [[ -n "${PREFILL_PORTS:-}" ]]; then
        # shellcheck disable=SC2206
        prefill_ports=(${PREFILL_PORTS})
        prefill_port_args=(--prefiller-ports "${prefill_ports[@]}")
      fi
      if [[ -n "${DECODE_HOSTS:-}" ]]; then
        # shellcheck disable=SC2206
        decode_hosts=(${DECODE_HOSTS})
        decode_host_args=(--decoder-hosts "${decode_hosts[@]}")
      fi
      if [[ -n "${DECODE_PORTS:-}" ]]; then
        # shellcheck disable=SC2206
        decode_ports=(${DECODE_PORTS})
        decode_port_args=(--decoder-ports "${decode_ports[@]}")
      fi
      exec "${PYTHON_BIN}" "${SOURCE_DIR}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
        --host 0.0.0.0 \
        --port "${PROXY_PORT}" \
        "${prefill_host_args[@]}" \
        "${prefill_port_args[@]}" \
        "${decode_host_args[@]}" \
        "${decode_port_args[@]}" \
        > "${proxy_log}" 2>&1
    fi
    ;;
  *) echo "bad role ${ROLE}" >&2; exit 2 ;;
esac
