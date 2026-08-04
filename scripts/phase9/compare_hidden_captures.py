#!/usr/bin/env python3
"""Compare strictly aligned BF16 and OSCAR hidden-state captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch
import torch.nn.functional as F


FORMAT_VERSION = 1
IDENTITY_FIELDS = ("hook", "layer_idx", "counter", "tp_rank", "pp_rank")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_identity(payload: dict[str, Any]) -> tuple[str, int, int, int, int]:
    hook = payload.get("capture_name") or payload.get("hook")
    if not isinstance(hook, str) or not hook:
        raise ValueError("capture is missing hook/capture_name")
    layer_idx = payload.get("layer_idx")
    if not isinstance(layer_idx, int):
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", hook)
        if layer_match is None:
            raise ValueError("capture identity field layer_idx cannot be derived")
        layer_idx = int(layer_match.group(1))
    values: list[Any] = [hook, layer_idx]
    for field in IDENTITY_FIELDS[2:]:
        value = payload.get(field)
        if not isinstance(value, int):
            raise ValueError(f"capture identity field {field} must be an integer")
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def identity_json(identity: tuple[str, int, int, int, int]) -> dict[str, Any]:
    return dict(zip(IDENTITY_FIELDS, identity, strict=True))


def _validated_tensor(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = payload.get("hidden_states")
    positions = payload.get("positions")
    if not isinstance(hidden_states, torch.Tensor):
        raise ValueError("capture hidden_states must be a tensor")
    if not isinstance(positions, torch.Tensor):
        raise ValueError("capture positions must be a tensor")
    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have at least two dimensions")
    if tuple(payload.get("shape", ())) != tuple(hidden_states.shape):
        raise ValueError("capture shape metadata mismatch")
    if payload.get("dtype") != str(hidden_states.dtype):
        raise ValueError("capture dtype metadata mismatch")
    rows = hidden_states.reshape(-1, hidden_states.shape[-1]).detach().cpu()
    flat_positions = positions.reshape(-1).detach().cpu()
    if rows.shape[0] != flat_positions.numel():
        raise ValueError("positions count does not match hidden-state rows")
    if not torch.isfinite(rows).all().item():
        raise ValueError("hidden_states contain non-finite values")
    return rows, flat_positions


def _summary(values: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


def compare_payloads(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_identity = semantic_identity(baseline)
    candidate_identity = semantic_identity(candidate)
    if baseline_identity != candidate_identity:
        raise ValueError("capture identity mismatch")
    baseline_rows, baseline_positions = _validated_tensor(baseline)
    candidate_rows, candidate_positions = _validated_tensor(candidate)
    if baseline_rows.shape != candidate_rows.shape:
        raise ValueError("hidden-state shape mismatch")
    if not torch.equal(baseline_positions, candidate_positions):
        raise ValueError("positions mismatch")

    baseline_fp64 = baseline_rows.to(torch.float64)
    candidate_fp64 = candidate_rows.to(torch.float64)
    baseline_norm = torch.linalg.vector_norm(baseline_fp64, dim=-1)
    candidate_norm = torch.linalg.vector_norm(candidate_fp64, dim=-1)
    if (baseline_norm == 0).any().item() or (candidate_norm == 0).any().item():
        raise ValueError("zero-norm hidden-state row cannot be compared")
    difference = candidate_fp64 - baseline_fp64
    cosine = F.cosine_similarity(baseline_fp64, candidate_fp64, dim=-1)
    relative_l2 = torch.linalg.vector_norm(difference, dim=-1) / baseline_norm

    return {
        "identity": identity_json(baseline_identity),
        "shape": list(baseline_rows.shape),
        "token_count": baseline_rows.shape[0],
        "position_min": int(baseline_positions.min().item()),
        "position_max": int(baseline_positions.max().item()),
        "cosine": _summary(cosine),
        "relative_l2": _summary(relative_l2),
        "max_abs": float(difference.abs().max().item()),
        "mse": float(difference.square().mean().item()),
    }


def _load_capture(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"capture payload must be a dictionary: {path}")
    return payload


def _indexed_captures(
    directory: Path,
) -> dict[tuple[str, int, int, int, int], tuple[Path, dict[str, Any]]]:
    paths = sorted(directory.rglob("*.pt"))
    if not paths:
        raise ValueError(f"capture directory contains no .pt files: {directory}")
    indexed = {}
    for path in paths:
        payload = _load_capture(path)
        identity = semantic_identity(payload)
        if identity in indexed:
            raise ValueError(
                f"duplicate capture identity in {directory}: "
                f"{identity_json(identity)}"
            )
        indexed[identity] = (path, payload)
    return indexed


def compare_capture_dirs(
    baseline_dir: Path,
    candidate_dir: Path,
) -> dict[str, Any]:
    baseline = _indexed_captures(baseline_dir)
    candidate = _indexed_captures(candidate_dir)
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "capture identity sets differ: "
            f"missing_candidate={missing_candidate}, "
            f"missing_baseline={missing_baseline}"
        )

    pairs = []
    for identity in sorted(baseline):
        baseline_path, baseline_payload = baseline[identity]
        candidate_path, candidate_payload = candidate[identity]
        metrics = compare_payloads(baseline_payload, candidate_payload)
        metrics["baseline"] = {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
        }
        metrics["candidate"] = {
            "path": str(candidate_path.resolve()),
            "sha256": sha256_file(candidate_path),
        }
        pairs.append(metrics)

    token_count = sum(pair["token_count"] for pair in pairs)
    cosine_mean = math.fsum(
        pair["cosine"]["mean"] * pair["token_count"] for pair in pairs
    ) / token_count
    relative_l2_mean = math.fsum(
        pair["relative_l2"]["mean"] * pair["token_count"] for pair in pairs
    ) / token_count
    return {
        "format_version": FORMAT_VERSION,
        "classification": "hidden_state_similarity_proxy_unthresholded",
        "promotion_gate": False,
        "pair_count": len(pairs),
        "token_count": token_count,
        "aggregate": {
            "cosine": {
                "min": min(pair["cosine"]["min"] for pair in pairs),
                "mean": cosine_mean,
                "max": max(pair["cosine"]["max"] for pair in pairs),
            },
            "relative_l2": {
                "min": min(pair["relative_l2"]["min"] for pair in pairs),
                "mean": relative_l2_mean,
                "max": max(pair["relative_l2"]["max"] for pair in pairs),
            },
            "max_abs": max(pair["max_abs"] for pair in pairs),
        },
        "pairs": pairs,
    }


def _is_scoped_path(path: Path, project_root: Path) -> bool:
    roots = (project_root / "artifacts", Path("/dev/shm"))
    return any(path == root or path.is_relative_to(root) for root in roots)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o644)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    baseline_dir = args.baseline_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    output = args.output.resolve()
    for label, path in (
        ("baseline capture directory", baseline_dir),
        ("candidate capture directory", candidate_dir),
        ("output", output),
    ):
        if not _is_scoped_path(path, project_root):
            raise SystemExit(f"{label} must be under artifacts/ or /dev/shm: {path}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    result = compare_capture_dirs(baseline_dir, candidate_dir)
    result["baseline_dir"] = str(baseline_dir)
    result["candidate_dir"] = str(candidate_dir)
    _atomic_write_json(output, result)
    print(json.dumps(result["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
