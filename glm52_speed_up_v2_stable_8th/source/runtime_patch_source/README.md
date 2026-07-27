# Runtime Patch Source

This directory contains the runtime patch files that `scripts/deploy/lib/manage_stack_2p1d.sh` copies into the GLM-5.2 vLLM image before service start.

The shape padding implementation is in:

```text
vllm/v1/worker/gpu_model_runner.py
```

Required markers checked by precheck:

```text
VLLM_PREFILL_SHAPE_BUCKET
_pad_for_prefill_shape_bucket
Prefill shape bucket enabled
PROXY_ANTHROPIC_SYSTEM_NORMALIZED
PROXY_PREFILL_INTERNAL_WARMUP_ALLOWED
```

Verify file integrity from this directory with:

```bash
sha256sum -c MANIFEST.sha256
```
