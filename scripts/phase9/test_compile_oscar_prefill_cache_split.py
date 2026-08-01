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
    def test_format_version_is_nine_for_partial_compact_load_screen(self) -> None:
        self.assertEqual(MODULE.FORMAT_VERSION, 9)

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

    def test_history_narrow_token_matrix_is_explicit(self) -> None:
        narrow_variants = [
            variant
            for variant in MODULE.VARIANTS
            if variant.kernel_mode == "history"
            and variant.block_t == 8
            and not variant.manual_history_value_reduce
        ]

        self.assertEqual(
            [variant.name for variant in narrow_variants],
            [
                "history_h4_t8_w4",
                "history_h2_t8_w4",
                "history_h1_t8_w4",
            ],
        )
        self.assertTrue(all(variant.num_warps == 4 for variant in narrow_variants))
        self.assertTrue(
            all(not variant.reload_history_for_value for variant in narrow_variants)
        )

    def test_history_manual_value_reduce_matrix_is_explicit(self) -> None:
        manual_variants = [
            variant
            for variant in MODULE.VARIANTS
            if variant.manual_history_value_reduce
        ]

        self.assertEqual(
            [variant.name for variant in manual_variants],
            [
                "history_manual_value_h4_t16_w4",
                "history_manual_value_h2_t16_w4",
                "history_manual_value_h1_t16_w4",
                "history_manual_value_h4_t8_w4",
                "history_manual_value_h2_t8_w4",
                "history_manual_value_h1_t8_w4",
            ],
        )
        self.assertTrue(
            all(variant.kernel_mode == "history" for variant in manual_variants)
        )
        self.assertEqual(
            [variant.block_t for variant in manual_variants],
            [16, 16, 16, 8, 8, 8],
        )
        self.assertTrue(all(variant.num_warps == 4 for variant in manual_variants))

    def test_history_maxnreg_matrix_and_compile_options_are_explicit(self) -> None:
        variants = [
            variant for variant in MODULE.VARIANTS if variant.maxnreg is not None
        ]

        self.assertEqual(
            [(variant.name, variant.maxnreg) for variant in variants],
            [
                ("history_h4_t16_w8_maxnreg128", 128),
                ("history_h4_t16_w8_maxnreg120", 120),
                ("history_h4_t16_w8_maxnreg112", 112),
                ("history_h4_t16_w8_maxnreg96", 96),
            ],
        )
        self.assertTrue(
            all(
                variant.kernel_mode == "history"
                and variant.block_h == 4
                and variant.block_t == 16
                and variant.num_warps == 8
                for variant in variants
            )
        )
        self.assertEqual(
            MODULE.compile_options(variants[0]),
            {"num_warps": 8, "num_stages": 1, "maxnreg": 128},
        )
        baseline = next(
            variant
            for variant in MODULE.VARIANTS
            if variant.name == "history_h4_t16_w8"
        )
        self.assertEqual(
            MODULE.compile_options(baseline),
            {"num_warps": 8, "num_stages": 1},
        )

    def test_compact_history_load_candidate_is_explicit(self) -> None:
        candidates = [
            variant for variant in MODULE.VARIANTS if variant.compact_history_loads
        ]

        self.assertEqual(
            [variant.name for variant in candidates],
            ["history_compact_loads_h8_t16_w8"],
        )
        candidate = candidates[0]
        self.assertEqual(candidate.kernel_mode, "history")
        self.assertEqual(candidate.block_h, 8)
        self.assertEqual(candidate.block_t, 16)
        self.assertEqual(candidate.num_warps, 8)
        self.assertTrue(
            MODULE.constants_for(MODULE._history_prefill_stage1, candidate)[
                "compact_history_loads"
            ]
        )

    def test_partial_compact_load_candidates_are_explicit_and_disjoint(self) -> None:
        candidates = [
            variant
            for variant in MODULE.VARIANTS
            if variant.compact_packed_loads or variant.compact_qparam_loads
        ]

        self.assertEqual(
            [variant.name for variant in candidates],
            [
                "history_compact_packed_loads_h8_t16_w8",
                "history_compact_qparam_loads_h8_t16_w8",
            ],
        )
        self.assertEqual(
            [
                (variant.compact_packed_loads, variant.compact_qparam_loads)
                for variant in candidates
            ],
            [(True, False), (False, True)],
        )
        for candidate in candidates:
            constants = MODULE.constants_for(MODULE._history_prefill_stage1, candidate)
            self.assertEqual(
                constants["compact_packed_loads"], candidate.compact_packed_loads
            )
            self.assertEqual(
                constants["compact_qparam_loads"], candidate.compact_qparam_loads
            )

    def test_compact_load_gate_requires_changed_binary_fewer_loads_and_no_spill(
        self,
    ) -> None:
        rows = [
            {
                "name": "history_h8_t16_w8",
                "status": "compiled",
                "cubin_sha256": "baseline",
                "ptx_ld_global_instruction_count": 165,
                "stack_bytes_per_thread": 0,
                "shared_bytes": 84_992,
                "registers_per_thread": 199,
            },
            {
                "name": "history_compact_loads_h8_t16_w8",
                "status": "compiled",
                "cubin_sha256": "candidate",
                "ptx_ld_global_instruction_count": 120,
                "stack_bytes_per_thread": 0,
                "shared_bytes": 84_992,
                "registers_per_thread": 190,
            },
        ]

        comparison = MODULE.summarize_compact_load_comparison(rows)

        self.assertTrue(comparison["binary_changed"])
        self.assertEqual(comparison["ptx_ld_global_instruction_delta"], -45)
        self.assertTrue(comparison["offline_promotion_candidate"])

    def test_partial_compact_gate_also_requires_fewer_registers_than_full(
        self,
    ) -> None:
        common = {
            "status": "compiled",
            "stack_bytes_per_thread": 0,
            "shared_bytes": 84_992,
        }
        rows = [
            {
                "name": "history_h8_t16_w8",
                "cubin_sha256": "baseline",
                "ptx_ld_global_instruction_count": 165,
                "registers_per_thread": 199,
                **common,
            },
            {
                "name": "history_compact_loads_h8_t16_w8",
                "cubin_sha256": "full",
                "ptx_ld_global_instruction_count": 71,
                "registers_per_thread": 230,
                **common,
            },
            {
                "name": "history_compact_packed_loads_h8_t16_w8",
                "cubin_sha256": "packed",
                "ptx_ld_global_instruction_count": 120,
                "registers_per_thread": 229,
                **common,
            },
            {
                "name": "history_compact_qparam_loads_h8_t16_w8",
                "cubin_sha256": "qparam",
                "ptx_ld_global_instruction_count": 100,
                "registers_per_thread": 230,
                **common,
            },
        ]

        comparison = MODULE.summarize_partial_compact_load_comparison(rows)

        self.assertTrue(comparison["comparison_available"])
        self.assertEqual(comparison["full_compact_registers_per_thread"], 230)
        by_name = {row["candidate_name"]: row for row in comparison["candidates"]}
        packed = by_name["history_compact_packed_loads_h8_t16_w8"]
        qparam = by_name["history_compact_qparam_loads_h8_t16_w8"]
        self.assertEqual(packed["ptx_ld_global_instruction_delta"], -45)
        self.assertTrue(packed["registers_below_full_compact"])
        self.assertTrue(packed["offline_promotion_candidate"])
        self.assertFalse(qparam["registers_below_full_compact"])
        self.assertFalse(qparam["offline_promotion_candidate"])

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
