#!/usr/bin/env python3
"""Run the stage-1 short, >320-token, decode, and 32K native smoke cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def post_json(url: str, body: dict[str, Any], timeout: int) -> tuple[int, Any, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            return response.status, payload, time.perf_counter() - started
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw_error": raw}
        return error.code, payload, time.perf_counter() - started


def post_json_with_progress(
    *,
    name: str,
    url: str,
    body: dict[str, Any],
    timeout: int,
) -> tuple[int, Any, float]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(post_json, url, body, timeout)
        started = time.monotonic()
        next_progress = 600
        while True:
            try:
                return future.result(timeout=60)
            except concurrent.futures.TimeoutError:
                elapsed = int(time.monotonic() - started)
                if elapsed >= next_progress:
                    print(
                        json.dumps(
                            {
                                "name": name,
                                "status": "running",
                                "elapsed_seconds": elapsed,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    next_progress += 600


def rendered_tokens(tokenizer: Any, text: str) -> int:
    messages = [{"role": "user", "content": text}]
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="high",
        enable_thinking=True,
    )
    if hasattr(tokens, "keys") and "input_ids" in tokens:
        tokens = tokens["input_ids"]
    return len(tokens)


def prompt_for_target(tokenizer: Any, target: int) -> tuple[str, int]:
    prefix = "请阅读下面的重复文本，并只回答“完成”。\n"
    unit = "这是用于验证上下文长度的固定文本。 "
    low, high = 1, target
    best_text = prefix
    best_tokens = rendered_tokens(tokenizer, best_text)
    while low <= high:
        middle = (low + high) // 2
        text = prefix + unit * middle
        count = rendered_tokens(tokenizer, text)
        if count <= target:
            best_text, best_tokens = text, count
            low = middle + 1
        else:
            high = middle - 1
    return best_text, best_tokens


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001",
    )
    parser.add_argument(
        "--served-model", default="glm-5.2-fp8-pruned-reap-e154"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    over_320_prompt, over_320_local_tokens = prompt_for_target(tokenizer, 512)
    context_32k_prompt, context_32k_local_tokens = prompt_for_target(tokenizer, 32000)

    cases = [
        {
            "name": "short",
            "prompt": "用一句话说明一加一等于几。",
            "max_tokens": 64,
            "ignore_eos": False,
            "timeout": 600,
        },
        {
            "name": "over_320_input",
            "prompt": over_320_prompt,
            "max_tokens": 64,
            "ignore_eos": False,
            "timeout": 600,
        },
        {
            "name": "continuous_decode_384",
            "prompt": "从1开始逐项输出整数，每个整数单独占一行。",
            "max_tokens": 384,
            "ignore_eos": True,
            "timeout": 1200,
        },
        {
            "name": "context_32k",
            "prompt": context_32k_prompt,
            "max_tokens": 64,
            "ignore_eos": False,
            "timeout": 3600,
        },
    ]

    results: list[dict[str, Any]] = []
    all_passed = True
    for case in cases:
        local_prompt_tokens = rendered_tokens(tokenizer, case["prompt"])
        body = {
            "model": args.served_model,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_tokens": case["max_tokens"],
            "ignore_eos": case["ignore_eos"],
        }
        status, response, elapsed = post_json_with_progress(
            name=case["name"],
            url=f"{args.base_url.rstrip('/')}/chat/completions",
            body=body,
            timeout=case["timeout"],
        )
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        passed = status == 200 and completion_tokens > 0
        if case["name"] == "over_320_input":
            passed = passed and prompt_tokens > 320
        elif case["name"] == "continuous_decode_384":
            passed = passed and completion_tokens == 384
        elif case["name"] == "context_32k":
            passed = passed and prompt_tokens >= 31500
        all_passed = all_passed and passed
        result = {
            "name": case["name"],
            "status": "passed" if passed else "failed",
            "http_status": status,
            "elapsed_seconds": elapsed,
            "local_prompt_tokens": local_prompt_tokens,
            "request_sha256": sha256_json(body),
            "usage": usage,
            "response": response,
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "name": case["name"],
                    "status": result["status"],
                    "http_status": status,
                    "elapsed_seconds": elapsed,
                    "local_prompt_tokens": local_prompt_tokens,
                    "usage": usage,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    output = {
        "format_version": 1,
        "status": "passed" if all_passed else "failed",
        "model_path": args.model_path,
        "served_model": args.served_model,
        "base_url": args.base_url,
        "prepared_prompt_tokens": {
            "over_320_input": over_320_local_tokens,
            "context_32k": context_32k_local_tokens,
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "status": output["status"]}))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
