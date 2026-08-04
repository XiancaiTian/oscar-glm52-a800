from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("run_hidden_similarity_replay.py")
SPEC = importlib.util.spec_from_file_location("run_hidden_similarity_replay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()


def fixture(root: Path, *, capture_ready: bool = True) -> tuple[Path, Path]:
    samples = []
    server_samples = []
    for index in range(9):
        token_ids = list(range(320 + index))
        digest = token_hash(token_ids)
        sample_id = f"gsm8k:{index:06d}"
        samples.append(
            {
                "id": sample_id,
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "token_ids_sha256": digest,
            }
        )
        server_samples.append(
            {
                "id": sample_id,
                "expected_count": len(token_ids),
                "server_count": len(token_ids),
                "server_max_model_len": 8192,
                "exact_token_ids": True,
                "server_token_ids_sha256": digest,
                "expected_token_ids_sha256": digest,
            }
        )
    selection = {
        "format_version": 1,
        "classification": "hidden_similarity_teacher_forced_replay_selection",
        "promotion_gate": False,
        "min_position": 320,
        "eligible_count": 9,
        "samples": samples,
    }
    selection_path = root / "selection.json"
    write_json(selection_path, selection)
    validation = {
        "format_version": 1,
        "classification": "hidden_similarity_server_token_ids_validation",
        "status": "passed" if capture_ready else "failed",
        "capture_ready": capture_ready,
        "promotion_gate": False,
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "samples": server_samples,
    }
    validation_path = root / "server_validation.json"
    write_json(validation_path, validation)
    return selection_path, validation_path


class HiddenSimilarityReplayTest(unittest.TestCase):
    def test_loads_exact_server_validated_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, validation = fixture(Path(directory))

            replay = REPLAY.load_validated_replay(selection, validation)

        self.assertEqual(len(replay), 9)
        self.assertEqual(replay[0]["id"], "gsm8k:000000")
        self.assertEqual(replay[-1]["token_count"], 328)

    def test_rejects_pending_server_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, validation = fixture(Path(directory), capture_ready=False)

            with self.assertRaisesRegex(ValueError, "capture-ready"):
                REPLAY.load_validated_replay(selection, validation)

    def test_builds_deterministic_completion_request(self) -> None:
        body = REPLAY.completion_request(
            model="glm-5.2-fp8-pruned-reap-e154",
            token_ids=[154822, 154824, 154826],
        )

        self.assertEqual(
            body,
            {
                "model": "glm-5.2-fp8-pruned-reap-e154",
                "prompt": [154822, 154824, 154826],
                "max_tokens": 1,
                "temperature": 0.0,
                "seed": 42,
            },
        )

    def test_rejects_capture_environment_mismatch(self) -> None:
        expected = {
            "VLLM_NORM_CAPTURE_MODE": "aux_runner",
            "VLLM_NORM_CAPTURE_DIR": "/dev/shm/replay/captures",
            "VLLM_NORM_CAPTURE_ENABLE_FILE": "/dev/shm/replay/capture.enable",
            "VLLM_NORM_CAPTURE_TP_RANK": "0",
            "VLLM_NORM_CAPTURE_LAYER": "36",
            "VLLM_NORM_CAPTURE_HOOK": (
                "model.layers.36.post_attention_layernorm"
            ),
        }
        actual = dict(expected)
        actual["VLLM_NORM_CAPTURE_LAYER"] = "35"

        with self.assertRaisesRegex(ValueError, "capture environment mismatch"):
            REPLAY.validate_capture_environment(actual, expected)


if __name__ == "__main__":
    unittest.main()
