from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "compile_oscar_history_score_pipeline.py"
SPEC = importlib.util.spec_from_file_location(
    "compile_oscar_history_score_pipeline",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CompileOscarHistoryScorePipelineTest(unittest.TestCase):
    def test_variant_matrix_covers_all_three_pipeline_stages(self) -> None:
        self.assertEqual(
            [(variant.name, variant.kernel_mode) for variant in MODULE.VARIANTS],
            [
                ("score_h8_t16_w8", "score"),
                ("score_h4_t16_w8", "score"),
                ("lse_tiles128_w4", "lse"),
                ("value_h2_d128_t16_w4", "value"),
                ("value_h1_d128_t16_w4", "value"),
            ],
        )

    def test_scratch_bytes_match_frozen_32k_final_chunk(self) -> None:
        scratch = MODULE.scratch_layout_bytes(
            query_tokens=2_048,
            num_heads=8,
            topk=2_048,
            block_t=16,
        )

        self.assertEqual(scratch["scores"], 134_217_728)
        self.assertEqual(scratch["tile_lse"], 8_388_608)
        self.assertEqual(scratch["final_lse"], 65_536)
        self.assertEqual(scratch["total"], 142_671_872)

    def test_kernels_form_real_compiler_visible_boundaries(self) -> None:
        score_source = inspect.getsource(MODULE._history_score_kernel.fn)
        lse_source = inspect.getsource(MODULE._history_lse_kernel.fn)
        value_source = inspect.getsource(MODULE._history_value_kernel.fn)

        self.assertIn("score_scratch_ptr", score_source)
        self.assertIn("tile_lse_ptr", score_source)
        self.assertNotIn("history_acc", score_source)

        self.assertIn("tile_lse_ptr", lse_source)
        self.assertIn("final_lse_ptr", lse_source)
        self.assertNotIn("history_data_ptr", lse_source)

        self.assertIn("score_scratch_ptr", value_source)
        self.assertIn("final_lse_ptr", value_source)
        self.assertIn("history_acc", value_source)
        self.assertNotIn("query_rotated_ptr", value_source)
        self.assertNotIn("rope_ptr", value_source)

    def test_pipeline_gate_requires_strict_candidate_for_every_stage(self) -> None:
        rows = [
            {
                "name": "score",
                "kernel_mode": "score",
                "status": "compiled",
                "strict_promotion_candidate": True,
            },
            {
                "name": "lse",
                "kernel_mode": "lse",
                "status": "compiled",
                "strict_promotion_candidate": True,
            },
            {
                "name": "value",
                "kernel_mode": "value",
                "status": "compiled",
                "strict_promotion_candidate": False,
            },
        ]

        gate = MODULE.summarize_pipeline_gate(rows)

        self.assertFalse(gate["pipeline_strict_promotion_feasible"])
        self.assertEqual(gate["missing_strict_stages"], ["value"])


if __name__ == "__main__":
    unittest.main()
