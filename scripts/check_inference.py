#!/usr/bin/env python3
"""Check the real tool-call protocol and concurrent request throughput."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
NATIVE_TOOL_SCHEMA = ROOT / "submission_stage2/inference/pi/tool-schema.json"
TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def native_tools() -> list[dict]:
    """This fixture is generated from the pinned Pi SDK and checked against its wire payload."""
    tools = json.loads(NATIVE_TOOL_SCHEMA.read_text())
    if (not isinstance(tools, list)
            or [item.get("function", {}).get("name") for item in tools] != ["read", "bash", "edit", "write"]
            or any(item.get("type") != "function" or not isinstance(item["function"].get("parameters"), dict) for item in tools)):
        raise ValueError("Native Pi tool-schema fixture is missing or has an unexpected interface")
    return tools


def tool_message(response: dict, expected_name: str = "bash", expected_arguments: dict | None = None) -> dict:
    if response.get("object") != "chat.completion":
        raise ValueError("not a chat.completion")
    choice = response["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
        raise ValueError(f"expected one tool call, got {choice.get('finish_reason')}, {len(calls)}")
    call = calls[0]
    if not isinstance(call.get("id"), str) or not call["id"] or call.get("type") != "function":
        raise ValueError("tool call requires function type and a non-empty ID")
    if call["function"]["name"] != expected_name:
        raise ValueError("wrong tool name")
    arguments = json.loads(call["function"]["arguments"])
    if arguments != ({"command": "pwd"} if expected_arguments is None else expected_arguments):
        raise ValueError(f"wrong tool arguments: {arguments!r}")
    message = {"role": "assistant", "content": choice["message"].get("content"), "tool_calls": calls}
    reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
    if reasoning:
        message["reasoning_content"] = reasoning
        message["reasoning"] = reasoning
    return message


async def check(base: str, model: str, key: str, concurrency: int, *, require_pi_service: bool = True) -> dict:
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    async with httpx.AsyncClient(
        base_url=base.rstrip("/") + "/", headers={"Authorization": f"Bearer {key}"}, timeout=300,
        limits=httpx.Limits(max_connections=concurrency + 2), trust_env=False,
    ) as client:
        response = await client.get("models")
        response.raise_for_status()
        ids = [entry["id"] for entry in response.json()["data"]]
        if model not in ids:
            raise ValueError(f"model alias missing: {model}; served: {ids}")
        if require_pi_service:
            health = await client.get(base.rstrip("/").removesuffix("/v1") + "/mera/pi/health")
            health.raise_for_status()
            state = health.json()
            if (state.get("service") != "native-pi-harbor-bridge" or state.get("model") != model
                    or state.get("worker_present") is not True or state.get("node_present") is not True):
                raise ValueError("Native Pi service is unavailable or points at the wrong model")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Call the bash tool exactly once with the command pwd. Do not answer in plain text."}],
            "tools": [TOOL], "tool_choice": "auto", "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        async def request(body: dict) -> tuple[dict, float]:
            start = time.monotonic()
            reply = await client.post("chat/completions", json=body)
            reply.raise_for_status()
            data = reply.json()
            if data.get("object") != "chat.completion":
                raise ValueError("not a chat.completion")
            if data.get("model") != model:
                raise ValueError(f"wrong response model: {data.get('model')}")
            return data, time.monotonic() - start

        data, contract_latency = await request(payload)
        assistant = tool_message(data)
        # Replay an actual tool result; catches chat-template errors absent from a one-turn check.
        multi = dict(payload)
        multi["messages"] = payload["messages"] + [assistant, {
            "role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"],
            "content": "/testbed\n", "name": "bash",
        }, {"role": "user", "content": "Call bash exactly once again with command pwd."}]
        multi_data, multi_latency = await request(multi)
        tool_message(multi_data)

        # Exercise the real four SDK schemas; no tool is executed by this API probe.
        tools = native_tools()
        probes = {
            "bash": {"command": "printf hello", "timeout": 30},
            "read": {"path": "/testbed/sample.go", "offset": 2, "limit": 3},
            "edit": {"path": "/testbed/sample.py", "oldText": "return 1", "newText": "return 2"},
            "write": {"path": "/testbed/sample.php", "content": "<?php echo 'hello';\n"},
        }
        for name, arguments in probes.items():
            custom = dict(payload, tools=tools, max_tokens=2048)
            custom["messages"] = [{"role": "user", "content": (
                f"Call {name} exactly once with these exact JSON arguments, without additional arguments: "
                + json.dumps(arguments, ensure_ascii=False) + ". Do not answer in plain text."
            )}]
            custom_data, _ = await request(custom)
            tool_message(custom_data, name, arguments)
        completion_payload = dict(payload, tools=tools)
        completion_payload["messages"] = [
            *payload["messages"], assistant,
            {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "name": "bash", "content": "/testbed\n"},
            {"role": "user", "content": "The tool operation is complete. Reply with exactly protocol checked. Do not call another tool."},
        ]
        final, _ = await request(completion_payload)
        final_choice = final["choices"][0]
        if (final_choice.get("finish_reason") != "stop" or final_choice["message"].get("tool_calls")
                or str(final_choice["message"].get("content") or "").strip() != "protocol checked"):
            raise ValueError("Native Pi normal-text completion protocol failed")

        workload = dict(payload, tools=tools, max_tokens=8192, temperature=0.8, top_p=0.95, top_k=20)
        start = time.monotonic()
        replies = await asyncio.gather(*(request(workload) for _ in range(concurrency)))
        duration = time.monotonic() - start
        for data, _ in replies:
            tool_message(data)
        output_tokens = sum((data.get("usage") or {}).get("completion_tokens", 0) for data, _ in replies)
        latencies = sorted(latency for _, latency in replies)
        return {
            "model": model, "contract_check": "passed", "multi_turn_check": "passed", "custom_tool_check": "passed",
            "pi_service_check": "passed" if require_pi_service else "skipped", "native_tool_names": ["read", "bash", "edit", "write"],
            "completion_check": "passed", "thinking": False,
            "workload": "short-prompt native Pi API load smoke check; not a full-context benchmark",
            "contract_latency_sec": round(contract_latency, 3), "multi_turn_latency_sec": round(multi_latency, 3),
            "concurrency": concurrency, "wall_time_sec": round(duration, 3),
            "completion_tokens": output_tokens, "aggregate_output_tokens_per_second": round(output_tokens / duration, 2),
            "p50_latency_sec": round(latencies[(len(latencies) - 1) // 2], 3),
            "max_latency_sec": round(latencies[-1], 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen38-27b-pi-mera")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-pi-health", action="store_true", help="Check only the model API, without requiring the native Pi service")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    result = asyncio.run(check(args.base_url, args.model, os.environ.get("REACT_AGENT_API_KEY", "dummy"), args.concurrency,
                               require_pi_service=not args.skip_pi_health))
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
