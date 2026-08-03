#!/usr/bin/env python3
from pathlib import Path
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent


class CandidateNativePathContractTest(unittest.TestCase):
    def test_native_link_comparison_canonicalizes_both_paths(self) -> None:
        wrapper = (SCRIPT_DIR / "run_candidate_tp8.sh").read_text(encoding="utf-8")

        self.assertIn('base_resolved="$(readlink -f "${base_path}")"', wrapper)
        self.assertIn(
            '"$(readlink -f "${runtime_path}")" == "${base_resolved}"',
            wrapper,
        )
        self.assertNotIn(
            '"$(readlink -f "${runtime_path}")" == "${base_path}"',
            wrapper,
        )


if __name__ == "__main__":
    unittest.main()
