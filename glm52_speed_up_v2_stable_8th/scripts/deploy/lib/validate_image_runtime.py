#!/usr/bin/env python3
import importlib
import os
import pathlib

import tokenizers
import torch
import transformers
import vllm


root = pathlib.Path("/opt/vllm_glm52_v1")
vllm_file = pathlib.Path(vllm.__file__).resolve()
print("vllm_file", vllm_file)
print("torch", torch.__version__, torch.version.cuda, torch.cuda.nccl.version())
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)

assert vllm_file.is_relative_to(root), vllm_file
assert torch.__version__.startswith("2.11.0"), torch.__version__
assert torch.cuda.nccl.version() == (2, 28, 9), torch.cuda.nccl.version()

required_paths = [
    "vllm/_C.abi3.so",
    "vllm/_moe_C.abi3.so",
    "vllm/_C_stable_libtorch.abi3.so",
    "vllm/cumem_allocator.abi3.so",
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
    "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
]
for rel in required_paths:
    path = root / rel
    assert path.exists(), path

skip_native_imports = os.environ.get("VALIDATE_SKIP_NATIVE_IMPORTS") == "1"
if skip_native_imports:
    print("native imports skipped by VALIDATE_SKIP_NATIVE_IMPORTS=1")
else:
    for module in [
        "vllm._C",
        "vllm._moe_C",
        "vllm._C_stable_libtorch",
        "vllm.cumem_allocator",
        "vllm.vllm_flash_attn._vllm_fa2_C",
        "vllm.vllm_flash_attn._vllm_fa3_C",
    ]:
        importlib.import_module(module)

splitmerge_lib = pathlib.Path(
    "/opt/glm52_speed_up_v1_stable/artifacts/native_ext/"
    "stage50_sparse_mla_m1_splitmerge_final_ops.so"
)
assert splitmerge_lib.exists(), splitmerge_lib

container_entry = pathlib.Path(
    "/opt/glm52_speed_up_v1_stable/scripts/deploy/lib/"
    "container_entry.sh"
)
assert container_entry.exists(), container_entry
