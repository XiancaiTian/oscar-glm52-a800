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
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dp0_rank3.1.pt.trace.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump({"traceEvents": events}, handle)
            result = ANALYZE.analyze_trace(str(path))

        self.assertEqual(result["rank"], 3)
        self.assertEqual(result["execute_context_count"], 2)
        self.assertEqual(result["prefill"]["duration_ms"], 0.5)
        self.assertEqual(result["prefill"]["kernel_total_ms"], 0.35)
        self.assertEqual(
            result["prefill"]["kernels"]["_mixed_sparse_decode_stage1"]["calls"],
            1,
        )
        self.assertNotIn("decode_only", result["prefill"]["kernels"])
        self.assertEqual(result["generation_duration_ms"]["median"], 0.2)


if __name__ == "__main__":
    unittest.main()
