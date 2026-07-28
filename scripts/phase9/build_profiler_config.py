#!/usr/bin/env python3
"""Emit the frozen Stage 9 torch-profiler server configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.profile_dir.is_absolute():
        raise SystemExit("--profile-dir must be absolute")
    project_root = Path(__file__).resolve().parents[2]
    profile_dir = args.profile_dir.resolve()
    roots = (project_root / "artifacts", Path("/dev/shm"))
    if not any(profile_dir.is_relative_to(root) for root in roots):
        raise SystemExit("--profile-dir must be under project artifacts/ or /dev/shm/")
    config = json.loads(
        (project_root / "configs/phase9/performance_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    profiler = config["profiler"]
    payload = {
        "profiler": "torch",
        "torch_profiler_dir": str(profile_dir),
        "torch_profiler_with_stack": profiler["torch_profiler_with_stack"],
        "torch_profiler_record_shapes": profiler["torch_profiler_record_shapes"],
        "torch_profiler_with_memory": profiler["torch_profiler_with_memory"],
        "torch_profiler_use_gzip": profiler["torch_profiler_use_gzip"],
        "torch_profiler_dump_cuda_time_total": profiler[
            "torch_profiler_dump_cuda_time_total"
        ],
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
