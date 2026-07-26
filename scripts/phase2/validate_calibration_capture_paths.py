#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-capture-dir", type=Path, required=True)
    parser.add_argument("--holdout-capture-dir", type=Path, required=True)
    return parser.parse_args()


def expected_paths(
    capture_dir: Path,
    *,
    layer_template: str,
    num_layers: int,
    tp_size: int,
) -> set[Path]:
    return {
        capture_dir
        / f"tp_rank_{tp_rank:02d}"
        / "layers"
        / f"{layer_template.format(layer=layer).replace('/', '_')}.pt"
        for tp_rank in range(tp_size)
        for layer in range(num_layers)
    }


def validate_split(
    capture_dir: Path,
    *,
    layer_template: str,
    num_layers: int,
    tp_size: int,
) -> dict:
    expected = expected_paths(
        capture_dir,
        layer_template=layer_template,
        num_layers=num_layers,
        tp_size=tp_size,
    )
    actual = set(capture_dir.rglob("*.pt"))
    missing = sorted(str(path) for path in expected - actual)
    extra = sorted(str(path) for path in actual - expected)
    if missing or extra:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "capture_dir": str(capture_dir),
                    "expected_files": len(expected),
                    "actual_files": len(actual),
                    "first_missing": missing[:8],
                    "first_extra": extra[:8],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return {
        "capture_dir": str(capture_dir),
        "files": len(actual),
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    common = {
        "layer_template": config["layer_name_template"],
        "num_layers": int(config["num_layers"]),
        "tp_size": int(config["tp_size"]),
    }
    result = {
        "status": "passed",
        "layer_name_template": common["layer_template"],
        "train": validate_split(args.train_capture_dir, **common),
        "holdout": validate_split(args.holdout_capture_dir, **common),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
