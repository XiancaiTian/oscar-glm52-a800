# 8th Source Service Contract

The 8th deployment package is derived from the verified 7th ClaudeCLI-2.1.204
2P1D flow, but deployment identity is now controlled by `configs/deploy_8th.env`.

Required local source:

```text
source/runtime_patch_source/
```

The precheck verifies that the local proxy source still contains:

```text
_normalize_anthropic_system_messages
PROXY_ANTHROPIC_SYSTEM_NORMALIZED
PROXY_PREFILL_INTERNAL_WARMUP_ALLOWED
/messages/count_tokens handling
/messages handling
```

Runtime patch source is configured by `RUNTIME_PATCH_SOURCE_DIR`. By default it
points to the self-contained 8th runtime patch source:

```text
source/runtime_patch_source/
```

The precheck verifies the required runtime patch files and the shape padding
markers in `vllm/v1/worker/gpu_model_runner.py`.

All deployment, warmup, verification, and acceptance output must be written under
`glm52_speed_up_v2_stable_8th`.
