# 8th Source Service Contract

The 8th deployment package is derived from the verified 7th ClaudeCLI-2.1.204
2P1D flow, but deployment identity is now controlled by `configs/deploy_8th.env`.

Required local source:

```text
source/vllm_glm52_v1/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
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
points to the same source tree used by the 7th deployment:

```text
/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v2_2nd_v2
```

All deployment, warmup, verification, and acceptance output must be written under
`glm52_speed_up_v2_stable_8th`.
