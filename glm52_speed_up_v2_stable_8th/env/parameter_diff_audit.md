# 8th Parameter Diff Audit

The 8th package preserves the verified 7th service behavior by default and moves
deployment-specific values into `configs/deploy_8th.env`.

Configurable migration fields:

- Hosts: `BUILD_HOST`, `PREFILL0_HEAD`, `PREFILL0_WORKER`, `PREFILL1_HEAD`,
  `PREFILL1_WORKER`, `DECODE_HEAD`, `DECODE_WORKER`, `TARGET_HOSTS`,
  `ALLOWED_HOSTS`, `EXPECTED_HOSTS_CSV`.
- Model: `MODEL_PATH`, `MODEL_ID`.
- Image: `IMAGE_TAG`, `IMAGE_TAR`.
- Runtime source: `SOURCE_DIR`, `RUNTIME_PATCH_SOURCE_DIR`.
- Container identity: `DEPLOY_LABEL` and optional explicit container-name vars.

Verified 7th behavior kept unless explicitly changed:

```text
topology = 2P1D
TP = 2
PP = 8
max_model_len = 202752
max_num_seqs = 16
max_num_batched_tokens = 2048
gpu_memory_utilization = 0.92
prefix_cache = enabled
shape bucket multiple = 92160
shape bucket max = 202752
safetensors_load_strategy = lazy
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS = 0
```
