from pathlib import Path
import tempfile
import unittest

from scripts.phase6.build_candidate_oci import build_layer


class BuildCandidateOciTest(unittest.TestCase):
    def test_build_layer_is_deterministic_with_pax_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = root / "payload"
            source = payload / "opt" / "vllm_glm52_v1"
            rotation = payload / "opt" / "oscar_artifacts" / "rotation_fit_v2"
            source.mkdir(parents=True)
            rotation.mkdir(parents=True)
            (source / ("long-" + "x" * 120)).write_text("source", encoding="utf-8")
            (rotation / "rotation.pt").write_text("rotation", encoding="utf-8")
            (
                payload / "opt" / "oscar_artifacts" / "oscar_runtime_expectation.json"
            ).write_text("{}\n", encoding="utf-8")

            first_work = root / "first"
            second_work = root / "second"
            first_work.mkdir()
            second_work.mkdir()
            first = build_layer(payload, first_work, 1_700_000_000)
            second = build_layer(payload, second_work, 1_700_000_000)

            self.assertEqual(first[1:], second[1:])
            self.assertEqual(first[0].read_bytes(), second[0].read_bytes())
            self.assertEqual(
                (first_work / "candidate-layer.tar").read_bytes(),
                (second_work / "candidate-layer.tar").read_bytes(),
            )
            self.assertIn(
                b"PaxHeaders/",
                (first_work / "candidate-layer.tar").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
