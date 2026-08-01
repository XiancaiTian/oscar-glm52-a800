#!/usr/bin/env python3
"""Screen maxnreg=128 on the standalone 32K final-chunk history kernel."""

from __future__ import annotations

from pathlib import Path

import benchmark_oscar_history_manual_value as history_bench

FORMAT_VERSION = 1
REFERENCE_NAME = "history_h8_t16_w8_reference"
CONTROL_NAME = "history_h4_t16_w8_uncapped_control"
CANDIDATE_NAME = "history_h4_t16_w8_maxnreg128_candidate"
VARIANTS = [
    {
        "name": REFERENCE_NAME,
        "block_h": 8,
        "block_t": 16,
        "num_warps": 8,
        "manual_history_value_reduce": False,
        "maxnreg": None,
    },
    {
        "name": CONTROL_NAME,
        "block_h": 4,
        "block_t": 16,
        "num_warps": 8,
        "manual_history_value_reduce": False,
        "maxnreg": None,
    },
    {
        "name": CANDIDATE_NAME,
        "block_h": 4,
        "block_t": 16,
        "num_warps": 8,
        "manual_history_value_reduce": False,
        "maxnreg": 128,
    },
]

launch_options = history_bench.launch_options
promotion_eligible = history_bench.candidate_is_faster_than_all


def main() -> int:
    return history_bench.run_benchmark(
        history_bench.parse_args(),
        format_version=FORMAT_VERSION,
        scope="oscar_history_maxnreg_microbenchmark",
        variants=VARIANTS,
        reference_name=REFERENCE_NAME,
        candidate_name=CANDIDATE_NAME,
        control_names=(CONTROL_NAME,),
        benchmark_script=Path(__file__),
    )


if __name__ == "__main__":
    raise SystemExit(main())
