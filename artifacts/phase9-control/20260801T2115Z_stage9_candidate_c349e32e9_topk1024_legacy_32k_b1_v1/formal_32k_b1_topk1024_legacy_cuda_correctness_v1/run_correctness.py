from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch
import vllm._C  # noqa: F401


TOP_K = 1024
EXPECTED_ENVIRONMENT = {
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND": "legacy",
    "VLLM_TOPK_ENV_CACHE": "1",
    "VLLM_TOPK_PREFILL_SORT_INDICES": "1",
}


def make_logits(columns: int, pattern: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    if pattern == "random":
        return torch.randn(1, columns, dtype=torch.float32, device="cuda")
    if pattern == "10LSBits":
        bits = torch.randint(
            0, 2**10, (1, columns), dtype=torch.int32, device="cuda"
        )
        return (0x3F900000 | bits).view(torch.float32)
    raise AssertionError(pattern)


def run_case(columns: int, pattern: str, seed: int) -> dict[str, object]:
    logits = make_logits(columns, pattern, seed)
    seq_lens = torch.tensor([columns], dtype=torch.int32, device="cuda")
    indices = torch.empty((1, TOP_K), dtype=torch.int32, device="cuda")

    torch.ops._C.top_k_per_row_decode(
        logits,
        1,
        seq_lens,
        indices,
        1,
        logits.stride(0),
        logits.stride(1),
        TOP_K,
    )
    torch.cuda.synchronize()

    actual_indices = indices[0]
    reference_indices = logits[0].topk(TOP_K).indices
    assert int(actual_indices.min()) >= 0
    assert int(actual_indices.max()) < columns
    assert int(actual_indices.unique().numel()) == TOP_K

    actual_values = logits[0, actual_indices].sort(descending=True).values
    reference_values = logits[0, reference_indices].sort(descending=True).values
    max_abs = float((actual_values - reference_values).abs().max().item())
    values_match = bool(
        torch.allclose(actual_values, reference_values, rtol=1e-5, atol=1e-5)
    )
    assert values_match

    sets_match = bool(
        torch.equal(
            actual_indices.sort().values,
            reference_indices.sort().values,
        )
    )
    branch = "insertion" if columns < 12288 else "single_block_radix"
    return {
        "columns": columns,
        "top_k": TOP_K,
        "batch_size": 1,
        "next_n": 1,
        "pattern": pattern,
        "seed": seed,
        "expected_branch": branch,
        "unique_indices": int(actual_indices.unique().numel()),
        "sets_match": sets_match,
        "values_match": values_match,
        "max_abs_value_difference": max_abs,
    }


def main() -> None:
    output = Path(os.environ["RESULT_JSON"])
    output.parent.mkdir(parents=True, exist_ok=False)
    actual_environment = {name: os.environ.get(name) for name in EXPECTED_ENVIRONMENT}
    assert actual_environment == EXPECTED_ENVIRONMENT
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert torch.version.cuda == "12.9"

    started = time.perf_counter()
    cases = [
        run_case(columns, pattern, seed)
        for columns in (8192, 32768)
        for pattern, seed in (("random", 42), ("10LSBits", 43))
    ]
    torch.cuda.synchronize()
    result = {
        "format_version": 1,
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "K=1024 legacy decode top-k CUDA correctness only",
        "environment": actual_environment,
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "cases_passed": len(cases),
        "cases_total": len(cases),
        "duration_seconds": time.perf_counter() - started,
        "cases": cases,
        "boundary": "No model, accuracy, PPL, TTFT, TPOT, or throughput measurement.",
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
