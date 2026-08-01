#!/usr/bin/env python3
"""Screen compact history dequant loads on the standalone 32K final chunk."""

from __future__ import annotations

from pathlib import Path

import benchmark_oscar_history_manual_value as history_bench

FORMAT_VERSION = 1
REFERENCE_NAME = "history_h8_t16_w8_reference"
CANDIDATE_NAME = "history_h8_t16_w8_compact_loads_candidate"
VARIANTS = [
    {
        "name": REFERENCE_NAME,
        "block_h": 8,
        "block_t": 16,
        "num_warps": 8,
        "manual_history_value_reduce": False,
        "compact_history_loads": False,
    },
    {
        "name": CANDIDATE_NAME,
        "block_h": 8,
        "block_t": 16,
        "num_warps": 8,
        "manual_history_value_reduce": False,
        "compact_history_loads": True,
    },
]

promotion_eligible = history_bench.candidate_is_faster_than_all


def main() -> int:
    return history_bench.run_benchmark(
        history_bench.parse_args(),
        format_version=FORMAT_VERSION,
        scope="oscar_history_compact_loads_microbenchmark",
        variants=VARIANTS,
        reference_name=REFERENCE_NAME,
        candidate_name=CANDIDATE_NAME,
        benchmark_script=Path(__file__),
    )


if __name__ == "__main__":
    raise SystemExit(main())
