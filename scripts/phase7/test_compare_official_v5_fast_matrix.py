#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("compare_official_v5_fast_matrix.py")
SPEC = importlib.util.spec_from_file_location(
    "compare_official_v5_fast_matrix",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def cell(scores: list[float], duration: float) -> dict:
    predictions = [
        {
            "id": f"gsm8k:{index:06d}",
            "prompt_hash": f"prompt-{index}",
            "gold": str(index),
            "task_type": "math_reasoning",
            "score": score,
            "extracted_answer": str(index),
            "truncated": False,
        }
        for index, score in enumerate(scores)
    ]
    accuracy = sum(scores) / len(scores)
    return {
        "predictions": predictions,
        "summary": {
            "protocol_fingerprint": "same",
            "duration_seconds": duration,
        },
        "validation": {
            "accuracy": accuracy,
            "truncation_rate": 0.0,
            "completion_tokens_mean": 100.0,
            "requests_per_hour": len(scores) * 3600 / duration,
        },
        "runtime_manifest": {
            "source_repository_commit": "source",
            "runtime_source_commit": "runtime-source",
            "model_filename_size_mtime_ns_manifest_sha256": "model",
            "evaluation_protocol": "official_v5_fast_screen",
            "evaluation_scope": "stage7_fast_gsm8k_256",
            "evaluation_manifest_sha256": "manifest",
        },
        "suite_identity": {
            "runtime_manifest_sha256": "runtime-manifest",
            "runtime_eval_config_sha256": "runtime-config",
            "selected_ids_sha256": "selected-ids",
            "sample_count": len(scores),
            "selection_seed": "seed",
        },
        "runner_environment": {
            "frozen_runner_sha256": "frozen-runner",
            "fast_runner_sha256": "fast-runner",
        },
    }


class FastMatrixTest(unittest.TestCase):
    def test_selects_16_when_outputs_match_and_pair_is_faster(self) -> None:
        scores = [1.0] * 100
        cells = {
            ("native", 8): cell(scores, 100.0),
            ("candidate", 8): cell(scores, 100.0),
            ("native", 16): cell(scores, 80.0),
            ("candidate", 16): cell(scores, 80.0),
        }
        result = MODULE.compare_matrix(cells, max_accuracy_drop=0.03)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["selected_concurrency"], 16)

    def test_selects_8_when_concurrency_changes_answer(self) -> None:
        scores = [1.0] * 100
        changed = [0.0] + [1.0] * 99
        cells = {
            ("native", 8): cell(scores, 100.0),
            ("candidate", 8): cell(scores, 100.0),
            ("native", 16): cell(changed, 80.0),
            ("candidate", 16): cell(scores, 80.0),
        }
        result = MODULE.compare_matrix(cells, max_accuracy_drop=0.03)
        self.assertEqual(result["selected_concurrency"], 8)
        self.assertEqual(result["cross_concurrency_differences"]["native"], 1)

    def test_fails_when_candidate_drop_exceeds_threshold(self) -> None:
        native = [1.0] * 100
        candidate = [0.0] * 4 + [1.0] * 96
        cells = {
            ("native", 8): cell(native, 100.0),
            ("candidate", 8): cell(candidate, 100.0),
            ("native", 16): cell(native, 80.0),
            ("candidate", 16): cell(candidate, 80.0),
        }
        result = MODULE.compare_matrix(cells, max_accuracy_drop=0.03)
        self.assertEqual(result["status"], "failed")

    def test_load_result_rejects_tampered_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            output = run / "cell" / "attempt"
            suite = output / "runtime_suite"
            suite.mkdir(parents=True)
            prediction = {
                "id": "gsm8k:000000",
                "benchmark": "GSM8K",
                "task_type": "math_reasoning",
                "prompt_hash": "prompt",
                "gold": "0",
                "score": 1.0,
                "extracted_answer": "0",
                "truncated": False,
                "evaluator_status": "scored",
                "protocol_fingerprint": "fingerprint",
            }
            summary = {
                "valid": True,
                "total": 1,
                "scored": 1,
                "status_counts": {"scored": 1},
                "protocol_fingerprint": "fingerprint",
            }
            identity = {
                "runtime_manifest_sha256": "suite-manifest",
                "runtime_eval_config_sha256": "suite-config",
                "selected_ids_sha256": "ids",
                "sample_count": 1,
                "selection_seed": "seed",
            }
            files = {
                output / "summary.json": json.dumps(summary),
                output / "summary_by_benchmark.json": "[]",
                output / "summary_by_task_type.json": "[]",
                output / "predictions.jsonl": json.dumps(prediction) + "\n",
                output / "failed_cases.jsonl": "",
                output / "fast_runner_state.json": "{}",
                output / "runner_command.txt": "command\n",
                suite / "manifest.jsonl": "suite manifest\n",
                suite / "eval_config.json": "suite config\n",
                suite / "fast_suite_identity.json": json.dumps(identity),
                run / "runtime_manifest.json": json.dumps(
                    {
                        "source_repository_commit": "source",
                        "runtime_source_commit": "runtime-source",
                        "model_filename_size_mtime_ns_manifest_sha256": "model",
                        "evaluation_protocol": "official_v5_fast_screen",
                        "evaluation_scope": "stage7_fast_gsm8k_1",
                        "evaluation_manifest_sha256": "manifest",
                    }
                ),
            }
            for path, text in files.items():
                path.write_text(text)
            identity["runtime_manifest_sha256"] = MODULE.sha256_file(
                suite / "manifest.jsonl"
            )
            identity["runtime_eval_config_sha256"] = MODULE.sha256_file(
                suite / "eval_config.json"
            )
            (suite / "fast_suite_identity.json").write_text(json.dumps(identity))
            environment = {
                "runtime_suite_manifest_sha256": MODULE.sha256_file(
                    suite / "manifest.jsonl"
                ),
                "runtime_suite_eval_config_sha256": MODULE.sha256_file(
                    suite / "eval_config.json"
                ),
                "selection_identity_sha256": MODULE.sha256_file(
                    suite / "fast_suite_identity.json"
                ),
                "runtime_manifest_sha256": MODULE.sha256_file(
                    run / "runtime_manifest.json"
                ),
                "frozen_runner_sha256": "frozen",
                "fast_runner_sha256": "fast",
            }
            (output / "runner_environment.txt").write_text(
                "".join(f"{key}={value}\n" for key, value in environment.items())
            )
            evidence = {
                "summary_sha256": output / "summary.json",
                "summary_by_benchmark_sha256": (output / "summary_by_benchmark.json"),
                "summary_by_task_type_sha256": (output / "summary_by_task_type.json"),
                "predictions_sha256": output / "predictions.jsonl",
                "failed_cases_sha256": output / "failed_cases.jsonl",
                "fast_runner_state_sha256": output / "fast_runner_state.json",
                "runner_command_sha256": output / "runner_command.txt",
                "runner_environment_sha256": output / "runner_environment.txt",
                "runtime_manifest_sha256": run / "runtime_manifest.json",
                "runtime_suite_manifest_sha256": suite / "manifest.jsonl",
                "runtime_suite_eval_config_sha256": suite / "eval_config.json",
                "fast_suite_identity_sha256": suite / "fast_suite_identity.json",
            }
            validation = {
                "status": "passed",
                "evaluation_role": "native",
                "protocol_version": "official_v5",
                "screening_protocol": "official_v5_fast_screen",
                "scope": "stage7_fast_gsm8k_1",
                "total": 1,
                "scored": 1,
                "request_failures": 0,
                "concurrency": 8,
                "server_max_model_len": 8192,
                "reasoning_effort": "high",
                "final_full_evaluation_still_required": True,
                "accuracy": 1.0,
                **{name: MODULE.sha256_file(path) for name, path in evidence.items()},
            }
            (output / "validation.json").write_text(json.dumps(validation))
            MODULE.load_result(
                output,
                role="native",
                concurrency=8,
                expected_total=1,
            )
            (output / "predictions.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "evidence hash"):
                MODULE.load_result(
                    output,
                    role="native",
                    concurrency=8,
                    expected_total=1,
                )


if __name__ == "__main__":
    unittest.main()
