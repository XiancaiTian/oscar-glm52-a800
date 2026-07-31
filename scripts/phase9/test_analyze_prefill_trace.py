from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("analyze_prefill_trace.py")
SPEC = importlib.util.spec_from_file_location("analyze_prefill_trace", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


class AnalyzePrefillTraceTest(unittest.TestCase):
    def test_isolates_prefill_kernel_window(self) -> None:
        events = [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "execute_context_1(1024)_generation_0(0)",
                "ts": 1000,
                "dur": 500,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "_mixed_sparse_decode_stage1",
                "ts": 1100,
                "dur": 300,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "other_prefill",
                "ts": 1400,
                "dur": 50,
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "execute_context_0(0)_generation_1(1)",
                "ts": 1600,
                "dur": 200,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "decode_only",
                "ts": 1650,
                "dur": 100,
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "execute_context_0(0)_generation_0(0)",
                "ts": 1900,
                "dur": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dp0_rank3.1.pt.trace.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump({"traceEvents": events}, handle)
            result = ANALYZE.analyze_trace(str(path))

        self.assertEqual(result["rank"], 3)
        self.assertEqual(result["execute_context_count"], 3)
        self.assertEqual(result["prefill"]["duration_ms"], 0.5)
        self.assertEqual(result["prefill"]["kernel_total_ms"], 0.35)
        self.assertEqual(
            result["prefill"]["kernels"]["_mixed_sparse_decode_stage1"]["calls"],
            1,
        )
        self.assertNotIn("decode_only", result["prefill"]["kernels"])
        self.assertEqual(result["generation_duration_ms"]["median"], 0.2)

    def test_aggregates_chunked_prefill_windows(self) -> None:
        events = [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "execute_context_1(2048)_generation_0(0)",
                "ts": 1000,
                "dur": 500,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "prefill_kernel",
                "ts": 1100,
                "dur": 300,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "between_chunks",
                "ts": 1550,
                "dur": 25,
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "execute_context_1(2048)_generation_0(0)",
                "ts": 1600,
                "dur": 600,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "prefill_kernel",
                "ts": 1700,
                "dur": 400,
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "execute_context_0(0)_generation_1(1)",
                "ts": 2300,
                "dur": 200,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dp0_rank5.1.pt.trace.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump({"traceEvents": events}, handle)
            result = ANALYZE.analyze_trace(str(path))

        self.assertEqual(result["prefill"]["chunk_count"], 2)
        self.assertEqual(result["prefill"]["tokens"], 4096)
        self.assertEqual(result["prefill"]["duration_ms"], 1.1)
        self.assertEqual(result["prefill"]["kernel_total_ms"], 0.7)
        self.assertEqual(len(result["prefill"]["chunks"]), 2)
        self.assertEqual(result["prefill"]["chunks"][0]["tokens"], 2048)
        self.assertEqual(result["prefill"]["chunks"][1]["duration_ms"], 0.6)
        self.assertEqual(
            result["prefill"]["kernels"]["prefill_kernel"]["calls"],
            2,
        )
        self.assertNotIn("between_chunks", result["prefill"]["kernels"])


if __name__ == "__main__":
    unittest.main()
