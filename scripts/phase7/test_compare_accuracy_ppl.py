from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).with_name("compare_accuracy_ppl.py")
SPEC = importlib.util.spec_from_file_location("phase7_compare", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(SCRIPT_PATH)
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)

FINALIZER_PATH = Path(__file__).with_name("finalize_accuracy_evidence.py")
FINALIZER_SPEC = importlib.util.spec_from_file_location(
    "phase7_accuracy_finalizer",
    FINALIZER_PATH,
)
if FINALIZER_SPEC is None or FINALIZER_SPEC.loader is None:
    raise RuntimeError(FINALIZER_PATH)
finalizer = importlib.util.module_from_spec(FINALIZER_SPEC)
sys.modules[FINALIZER_SPEC.name] = finalizer
FINALIZER_SPEC.loader.exec_module(finalizer)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class CompareAccuracyPplTest(unittest.TestCase):
    def test_frozen_benchmark_contract(self) -> None:
        self.assertEqual(sum(comparison.EXPECTED_BENCHMARK_COUNTS.values()), 2360)
        self.assertTrue(
            comparison.is_scoped_artifact_path(
                Path("/dev/shm/oscar-stage7/result"),
                Path("/nfs/AE/txc/oscar-glm"),
            )
        )
        self.assertFalse(
            comparison.is_scoped_artifact_path(
                Path("/nfs/AE/zhanghong/workflow/vllm_a/result"),
                Path("/nfs/AE/txc/oscar-glm"),
            )
        )

    def test_summarize_rejects_invalid_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid scored values"):
            comparison.summarize(
                [
                    {
                        "id": "bad",
                        "evaluator_status": "scored",
                        "score": float("nan"),
                    }
                ]
            )

    def test_accuracy_validation_binds_all_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            predictions = output / "predictions.jsonl"
            predictions.write_text("{}\n", encoding="utf-8")
            (output / "summary.json").write_text("{}\n", encoding="utf-8")
            (output / "runner_command.txt").write_text("runner\n", encoding="utf-8")
            environment = {
                **comparison.EXPECTED_ACCURACY_ENVIRONMENT,
                "runtime_eval_config_sha256": "",
            }
            runtime_config = output / "runtime_suite/eval_config.json"
            runtime_config.parent.mkdir()
            runtime_config.write_text("{}\n", encoding="utf-8")
            environment["runtime_eval_config_sha256"] = comparison.sha256_file(
                runtime_config
            )
            (output / "runner_environment.txt").write_text(
                "".join(f"{name}={value}\n" for name, value in environment.items()),
                encoding="utf-8",
            )
            write_json(
                output / "validation.json",
                {
                    "status": "passed",
                    "total": 2360,
                    "scored": 2360,
                    "accuracy": 0.5,
                    "predictions_rows": 2360,
                    "predictions_sha256": comparison.sha256_file(predictions),
                    "summary_sha256": comparison.sha256_file(output / "summary.json"),
                    "runner_command_sha256": comparison.sha256_file(
                        output / "runner_command.txt"
                    ),
                    "runner_environment_sha256": comparison.sha256_file(
                        output / "runner_environment.txt"
                    ),
                },
            )
            validation = comparison.validate_accuracy_evidence(predictions)
            self.assertEqual(validation["accuracy"], 0.5)
            predictions.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "OSCAR predictions changed"):
                comparison.validate_accuracy_evidence(predictions)

    def test_ppl_validation_binds_frozen_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            summary = output / "summary.json"
            summary.write_text("{}\n", encoding="utf-8")
            (output / "runner_command.txt").write_text("runner\n", encoding="utf-8")
            (output / "runtime_environment.txt").write_text(
                "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n",
                encoding="utf-8",
            )
            write_json(
                output / "validation.json",
                {
                    "status": "passed",
                    "total": 1,
                    "scored": 1,
                    "perplexity": 6.5,
                    "mean_nll": 1.8,
                    "evaluated_tokens": 289708,
                    "windows": 563,
                    "summary_sha256": comparison.sha256_file(summary),
                    "frozen_evaluator_identity_sha256": (
                        comparison.EXPECTED_PPL_IDENTITY_SHA256
                    ),
                    "frozen_evaluator_manifest_sha256": (
                        comparison.EXPECTED_PPL_MANIFEST_SHA256
                    ),
                    "frozen_ppl_runner_sha256": (comparison.EXPECTED_PPL_RUNNER_SHA256),
                    "runner_command_sha256": comparison.sha256_file(
                        output / "runner_command.txt"
                    ),
                    "runtime_environment_sha256": comparison.sha256_file(
                        output / "runtime_environment.txt"
                    ),
                },
            )
            validation = comparison.validate_ppl_evidence(summary)
            self.assertEqual(validation["perplexity"], 6.5)

    def test_finalizer_preserves_original_validation(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            root = Path(temp)
            source = root / "source"
            output = root / "finalized"
            (source / "runtime_suite").mkdir(parents=True)
            predictions = source / "predictions.jsonl"
            predictions.write_text("{}\n" * 2360, encoding="utf-8")
            write_json(
                source / "summary.json",
                {"total": 2360, "scored": 2360, "accuracy": 0.5},
            )
            (source / "runner_command.txt").write_text(
                "runner\n",
                encoding="utf-8",
            )
            runtime_config = source / "runtime_suite/eval_config.json"
            runtime_config.write_text("{}\n", encoding="utf-8")
            environment = {
                **finalizer.EXPECTED_ENVIRONMENT,
                "runtime_eval_config_sha256": finalizer.sha256_file(runtime_config),
            }
            (source / "runner_environment.txt").write_text(
                "".join(f"{name}={value}\n" for name, value in environment.items()),
                encoding="utf-8",
            )
            runner_validation = {
                "status": "passed",
                "total": 2360,
                "scored": 2360,
                "accuracy": 0.5,
                "predictions_rows": 2360,
                "predictions_sha256": finalizer.sha256_file(predictions),
            }
            write_json(source / "validation.json", runner_validation)

            result = finalizer.finalize(source, output, Path("/unused"))
            self.assertTrue(result["finalized"])
            self.assertEqual(
                json.loads((output / "runner_validation.json").read_text()),
                runner_validation,
            )
            self.assertEqual(
                comparison.validate_accuracy_evidence(output / "predictions.jsonl")[
                    "accuracy"
                ],
                0.5,
            )
            self.assertEqual(
                (source / "validation.json").read_text(),
                json.dumps(runner_validation) + "\n",
            )


if __name__ == "__main__":
    unittest.main()
