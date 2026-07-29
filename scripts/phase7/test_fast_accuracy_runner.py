#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


def load(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = load("prepare_official_v5_fast_suite", "prepare_official_v5_fast_suite.py")
RUNNER = load("run_accuracy_suite_fast", "run_accuracy_suite_fast.py")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = PROJECT_ROOT / "artifacts/phase7/frozen_evaluator_v5_20260728"


class FastSuiteTest(unittest.TestCase):
    def test_selection_is_deterministic_and_keeps_source_order(self) -> None:
        manifest = [
            {"id": f"gsm8k:{index:06d}", "benchmark": "GSM8K"} for index in range(20)
        ]
        first = PREPARE.select_gsm8k(manifest, count=7, seed="fixed")
        second = PREPARE.select_gsm8k(manifest, count=7, seed="fixed")
        self.assertEqual(first, second)
        source_positions = [manifest.index(row) for row in first]
        self.assertEqual(source_positions, sorted(source_positions))

    def test_atomic_checkpoint_resume_uses_only_scored_rows(self) -> None:
        manifest = [
            {"id": "gsm8k:000000"},
            {"id": "gsm8k:000001"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scored = {
                "id": manifest[0]["id"],
                "evaluator_status": "scored",
                "protocol_fingerprint": "fingerprint",
            }
            failed = {
                "id": manifest[1]["id"],
                "evaluator_status": "request_failed",
                "protocol_fingerprint": "fingerprint",
            }
            RUNNER.save_checkpoint(root, index=0, result=scored)
            RUNNER.save_checkpoint(root, index=1, result=failed)
            cached = RUNNER.load_cached_results(
                manifest=manifest,
                predictions_path=root / "predictions.jsonl",
                checkpoint_dir=root,
                protocol_fingerprint="fingerprint",
            )
            self.assertEqual(cached, {0: scored})
            self.assertEqual(
                json.loads((root / "000000.json").read_text()),
                scored,
            )

    def test_resume_rejects_protocol_mismatch(self) -> None:
        manifest = [{"id": "gsm8k:000000"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RUNNER.save_checkpoint(
                root,
                index=0,
                result={
                    "id": manifest[0]["id"],
                    "evaluator_status": "scored",
                    "protocol_fingerprint": "old",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                RUNNER.load_cached_results(
                    manifest=manifest,
                    predictions_path=root / "predictions.jsonl",
                    checkpoint_dir=root,
                    protocol_fingerprint="new",
                )

    def test_prior_duration_is_loaded_and_bound_to_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "fast_runner_state.json"
            state.write_text(
                json.dumps(
                    {
                        "protocol_fingerprint": "fingerprint",
                        "active_duration_seconds": 12.5,
                    }
                )
            )
            self.assertEqual(
                RUNNER.load_prior_active_duration(
                    state_path=state,
                    protocol_fingerprint="fingerprint",
                ),
                12.5,
            )
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                RUNNER.load_prior_active_duration(
                    state_path=state,
                    protocol_fingerprint="different",
                )

    def test_fast_budget_is_fixed_to_full_gsm8k_maximum(self) -> None:
        config = {
            "screening_protocol": {
                "server_max_model_len": 8192,
                "benchmark_max_prompt_tokens": 218,
            }
        }
        budgets = RUNNER.normalize_fast_budgets(
            computed={
                "GSM8K": {
                    "server_max_model_len": 8192,
                    "max_prompt_tokens": 200,
                    "fixed_output_limit": 7992,
                }
            },
            config=config,
        )
        self.assertEqual(
            budgets["GSM8K"],
            {
                "server_max_model_len": 8192,
                "max_prompt_tokens": 218,
                "observed_max_prompt_tokens": 200,
                "fixed_output_limit": 7974,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            RUNNER.normalize_fast_budgets(
                computed={
                    "GSM8K": {
                        "server_max_model_len": 8192,
                        "max_prompt_tokens": 219,
                        "fixed_output_limit": 7973,
                    }
                },
                config=config,
            )

    @unittest.skipUnless(
        (FROZEN_ROOT / ".venv/bin/python").is_file(),
        "frozen official_v5 evaluator is unavailable",
    )
    def test_runner_resumes_without_repeating_completions(self) -> None:
        counters = {"tokenize": 0, "completion": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                json.loads(self.rfile.read(length))
                if self.path == "/tokenize":
                    with lock:
                        counters["tokenize"] += 1
                    payload = {"count": 100, "max_model_len": 8192}
                elif self.path == "/v1/chat/completions":
                    with lock:
                        counters["completion"] += 1
                    payload = {
                        "choices": [
                            {
                                "message": {"content": "#### 0"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 5,
                            "total_tokens": 105,
                        },
                    }
                else:
                    self.send_error(404)
                    return
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                suite = root / "suite"
                output = root / "output"
                suite.mkdir()
                source_suite = (
                    FROZEN_ROOT / "accuracy_suites/model_agnostic_accuracy_official_v5"
                )
                rows = [
                    row
                    for row in PREPARE.read_jsonl(source_suite / "manifest.jsonl")
                    if row["benchmark"] == "GSM8K"
                ][:2]
                (suite / "manifest.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                (suite / "eval_config.json").write_bytes(
                    (
                        PROJECT_ROOT / "configs/phase7/"
                        "official_v5_eval_config_fast_high_timeout_3600.json"
                    ).read_bytes()
                )
                command = [
                    str(FROZEN_ROOT / ".venv/bin/python"),
                    str(Path(__file__).with_name("run_accuracy_suite_fast.py")),
                    "--frozen-runner",
                    str(FROZEN_ROOT / "tools/run_accuracy_suite.py"),
                    "--suite-dir",
                    str(suite),
                    "--output-dir",
                    str(output),
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}/v1",
                    "--model",
                    "fake-model",
                    "--concurrency",
                    "2",
                    "--checkpoint-every",
                    "20",
                    "--code-eval-isolated",
                    "--resume",
                ]
                environment = {
                    **os.environ,
                    "NLTK_DATA": str(FROZEN_ROOT / "nltk_data"),
                    "PYTHONPATH": str(FROZEN_ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
                subprocess.run(
                    command,
                    check=True,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(counters["completion"], 2)
                subprocess.run(
                    command,
                    check=True,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(counters["completion"], 2)
                self.assertEqual(counters["tokenize"], 4)
                state = json.loads(
                    (output / "fast_runner_state.json").read_text(encoding="utf-8")
                )
                predictions = [
                    json.loads(line)
                    for line in (output / "predictions.jsonl").read_text().splitlines()
                ]
                self.assertEqual(state["status"], "completed")
                self.assertEqual(state["resumed"], 2)
                self.assertEqual(len(predictions), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
