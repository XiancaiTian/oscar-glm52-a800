from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import torch


MODULE_PATH = Path(__file__).with_name("compare_hidden_captures.py")
SPEC = importlib.util.spec_from_file_location("compare_hidden_captures", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def capture(hidden_states: torch.Tensor, positions: list[int]) -> dict:
    return {
        "hidden_states": hidden_states,
        "positions": torch.tensor(positions, dtype=torch.int64),
        "shape": tuple(hidden_states.shape),
        "dtype": str(hidden_states.dtype),
        "hook": "model.layers.38.input_layernorm",
        "layer_idx": 38,
        "counter": 7,
        "tp_rank": 0,
        "pp_rank": 0,
    }


class CompareHiddenCapturesTest(unittest.TestCase):
    def test_computes_token_aligned_metrics(self) -> None:
        baseline = capture(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            [10, 11],
        )
        candidate = capture(
            torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
            [10, 11],
        )

        result = COMPARE.compare_payloads(baseline, candidate)

        self.assertEqual(result["token_count"], 2)
        self.assertAlmostEqual(result["cosine"]["min"], 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(
            result["cosine"]["mean"],
            (1.0 + 1.0 / math.sqrt(2.0)) / 2.0,
        )
        self.assertAlmostEqual(result["relative_l2"]["mean"], 0.5)
        self.assertEqual(result["max_abs"], 1.0)

    def test_rejects_position_mismatch(self) -> None:
        baseline = capture(torch.ones((2, 4)), [10, 11])
        candidate = capture(torch.ones((2, 4)), [10, 12])

        with self.assertRaisesRegex(ValueError, "positions mismatch"):
            COMPARE.compare_payloads(baseline, candidate)

    def test_derives_layer_from_aux_runner_hook(self) -> None:
        payload = capture(torch.ones((2, 4)), [10, 11])
        del payload["layer_idx"]

        identity = COMPARE.semantic_identity(payload)

        self.assertEqual(COMPARE.identity_json(identity)["layer_idx"], 38)

    def test_pairs_by_semantic_identity_not_filename(self) -> None:
        baseline = capture(torch.ones((2, 4)), [10, 11])
        candidate = capture(torch.ones((2, 4)), [10, 11])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_dir = root / "baseline"
            candidate_dir = root / "candidate"
            baseline_dir.mkdir()
            candidate_dir.mkdir()
            torch.save(baseline, baseline_dir / "bf16_pid1_time1.pt")
            torch.save(candidate, candidate_dir / "oscar_pid9_time9.pt")

            result = COMPARE.compare_capture_dirs(baseline_dir, candidate_dir)

        self.assertEqual(result["pair_count"], 1)
        self.assertEqual(result["token_count"], 2)
        self.assertEqual(result["pairs"][0]["identity"]["counter"], 7)

    def test_rejects_duplicate_semantic_identity(self) -> None:
        payload = capture(torch.ones((2, 4)), [10, 11])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_dir = root / "baseline"
            candidate_dir = root / "candidate"
            baseline_dir.mkdir()
            candidate_dir.mkdir()
            torch.save(payload, baseline_dir / "one.pt")
            torch.save(payload, baseline_dir / "two.pt")
            torch.save(payload, candidate_dir / "only.pt")

            with self.assertRaisesRegex(ValueError, "duplicate capture identity"):
                COMPARE.compare_capture_dirs(baseline_dir, candidate_dir)


if __name__ == "__main__":
    unittest.main()
