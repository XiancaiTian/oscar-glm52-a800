from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "compile_oscar_prefill_cache_split.py"
SPEC = importlib.util.spec_from_file_location(
    "compile_oscar_prefill_cache_split",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CompileOscarPrefillCacheSplitTest(unittest.TestCase):
    def test_matrix_contains_control_and_both_specialized_paths(self) -> None:
        names = [variant.name for variant in MODULE.VARIANTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "mixed_h8_t16_w8")
        for required in (
            "history_h8_t16_w8",
            "history_h8_t16_w4",
            "bf16_h8_t16_w8",
            "bf16_h8_t16_w4",
        ):
            self.assertIn(required, names)

    def test_specialized_kernels_keep_only_one_value_accumulator(self) -> None:
        history_source = inspect.getsource(MODULE._history_prefill_stage1.fn)
        bf16_source = inspect.getsource(MODULE._bf16_prefill_stage1.fn)

        self.assertIn("history_acc =", history_source)
        self.assertNotIn("bf16_acc =", history_source)
        self.assertIn("bf16_acc =", bf16_source)
        self.assertNotIn("history_acc =", bf16_source)

    def test_history_reload_matrix_is_explicit_and_history_only(self) -> None:
        reload_variants = [
            variant for variant in MODULE.VARIANTS if variant.reload_history_for_value
        ]

        self.assertEqual(
            [variant.name for variant in reload_variants],
            [
                "history_reload_h8_t16_w8",
                "history_reload_h4_t16_w8",
                "history_reload_h4_t16_w4",
                "history_reload_h2_t16_w4",
                "history_reload_h1_t16_w4",
            ],
        )
        self.assertTrue(
            all(variant.kernel_mode == "history" for variant in reload_variants)
        )

    def test_dual_block_gate_requires_shared_and_register_capacity(self) -> None:
        feasible = MODULE.classify_resources(
            shared_bytes=80_000,
            registers_per_thread=120,
            stack_bytes_per_thread=0,
            num_warps=8,
        )
        self.assertTrue(feasible["dual_block_resource_feasible"])

        shared_failure = MODULE.classify_resources(
            shared_bytes=90_000,
            registers_per_thread=120,
            stack_bytes_per_thread=0,
            num_warps=8,
        )
        self.assertFalse(shared_failure["dual_block_resource_feasible"])
        self.assertFalse(shared_failure["shared_allows_two_blocks"])

        register_failure = MODULE.classify_resources(
            shared_bytes=80_000,
            registers_per_thread=129,
            stack_bytes_per_thread=0,
            num_warps=8,
        )
        self.assertFalse(register_failure["dual_block_resource_feasible"])
        self.assertFalse(register_failure["registers_allow_two_blocks"])

        spill_failure = MODULE.classify_resources(
            shared_bytes=80_000,
            registers_per_thread=120,
            stack_bytes_per_thread=8,
            num_warps=8,
        )
        self.assertTrue(spill_failure["dual_block_resource_feasible"])
        self.assertFalse(spill_failure["stack_free"])
        self.assertFalse(spill_failure["strict_promotion_candidate"])

    def test_split_gate_requires_both_specialized_paths(self) -> None:
        rows = [
            {
                "name": "history",
                "kernel_mode": "history",
                "status": "compiled",
                "dual_block_resource_feasible": True,
                "strict_promotion_candidate": False,
            },
            {
                "name": "bf16",
                "kernel_mode": "bf16",
                "status": "compiled",
                "dual_block_resource_feasible": True,
                "strict_promotion_candidate": True,
            },
        ]

        gate = MODULE.summarize_split_gate(rows)

        self.assertTrue(gate["cache_split_dual_block_feasible"])
        self.assertFalse(gate["cache_split_strict_promotion_feasible"])
        self.assertEqual(gate["history_dual_block_candidates"], ["history"])
        self.assertEqual(gate["history_strict_candidates"], [])

    def test_reload_comparison_detects_compiler_elimination(self) -> None:
        common = {
            "status": "compiled",
            "shared_bytes": 76_288,
            "registers_per_thread": 255,
            "stack_bytes_per_thread": 176,
            "cubin_sha256": "same-cubin",
            "resource_usage_sha256": "same-resource",
        }
        rows = [
            {"name": "history_h4_t16_w4", **common},
            {"name": "history_reload_h4_t16_w4", **common},
        ]

        comparison = MODULE.summarize_reload_comparison(rows)

        self.assertTrue(comparison["all_pairs_binary_and_resource_identical"])
        self.assertFalse(comparison["reload_changed_any_candidate"])
        self.assertEqual(comparison["pairs"][0]["shared_bytes_delta"], 0)


if __name__ == "__main__":
    unittest.main()
