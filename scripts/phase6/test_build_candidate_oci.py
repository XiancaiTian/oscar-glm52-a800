import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.phase6.build_candidate_oci import build_layer


class BuildCandidateOciTest(unittest.TestCase):
    def test_split_topk_source_identity_is_frozen(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        manifest_path = project_root / "configs/phase6/candidate_inputs.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dockerfile = project_root / manifest["dockerfile"]["path"]
        dockerfile_text = dockerfile.read_text(encoding="utf-8")

        expected_commit = "1e768aef6a3916b05f29db0a1fa21a9ad1074712"
        expected_tree = "178aeebdc7dda2b0d21bc565d60da05d668b293a"
        self.assertEqual(manifest["source"]["commit"], expected_commit)
        self.assertEqual(manifest["source"]["tree"], expected_tree)
        self.assertEqual(
            manifest["output_tag"],
            "glm52-oscar-a800-phase6-1e768aef6-0275043c",
        )
        self.assertIn(f"ARG SOURCE_COMMIT={expected_commit}", dockerfile_text)
        self.assertIn(f"ARG SOURCE_TREE={expected_tree}", dockerfile_text)
        self.assertEqual(
            manifest["dockerfile"]["sha256"],
            hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        )

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
