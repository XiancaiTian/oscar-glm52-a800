#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).with_name("compare_official_v5_gsm8k.py")
SPEC = importlib.util.spec_from_file_location("compare_official_v5_gsm8k", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def result(scores: list[float], fingerprint: str = "same") -> dict:
    predictions = [
        {
            "id": f"gsm8k:{index}",
            "benchmark": "GSM8K",
            "task_type": "math_reasoning",
            "prompt_hash": f"prompt-{index}",
            "gold": str(index),
            "score": score,
            "evaluator_status": "scored",
            "protocol_fingerprint": fingerprint,
        }
        for index, score in enumerate(scores)
    ]
    accuracy = sum(scores) / len(scores)
    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "summary": {"protocol_fingerprint": fingerprint},
        "runtime_manifest": {
            "main_commit": "main",
            "source_repository_commit": "source",
            "model_filename_size_mtime_ns_manifest_sha256": "model",
            "evaluation_protocol": "official_v5",
            "evaluation_scope": "current_stage_gsm8k",
            "evaluation_manifest_sha256": "suite",
        },
    }


class CompareTest(unittest.TestCase):
    def test_passes_at_threshold_and_classifies(self) -> None:
        baseline = result([1.0] * 100)
        candidate = result([0.0] * 3 + [1.0] * 97)
        comparison, differences = MODULE.compare(baseline, candidate, 0.03)
        self.assertEqual(comparison["status"], "passed")
        self.assertAlmostEqual(comparison["accuracy_drop"], 0.03)
        self.assertEqual(comparison["categories"]["baseline_only_correct"], 3)
        self.assertEqual(len(differences), 3)
        self.assertTrue(comparison["final_full_evaluation_still_required"])

    def test_fails_above_threshold(self) -> None:
        baseline = result([1.0] * 100)
        candidate = result([0.0] * 4 + [1.0] * 96)
        comparison, _ = MODULE.compare(baseline, candidate, 0.03)
        self.assertEqual(comparison["status"], "failed")

    def test_rejects_provenance_mismatch(self) -> None:
        baseline = result([1.0])
        candidate = result([1.0])
        candidate["runtime_manifest"]["evaluation_manifest_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "runtime provenance mismatch"):
            MODULE.compare(baseline, candidate, 0.03)

    def test_rejects_protocol_fingerprint_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "protocol fingerprints differ"):
            MODULE.compare(result([1.0], "one"), result([1.0], "two"), 0.03)


if __name__ == "__main__":
    unittest.main()
