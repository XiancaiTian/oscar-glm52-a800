from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import unittest


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


if __name__ == "__main__":
    unittest.main()
