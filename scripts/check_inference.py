#!/usr/bin/env python3
"""Check the real tool-call protocol and concurrent request throughput."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
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


def tool_message(response: dict) -> dict:
    if response.get("object") != "chat.completion":
        raise ValueError("not a chat.completion")
    choice = response["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
        raise ValueError(f"expected one tool call, got {choice.get('finish_reason')}, {len(calls)}")
    call = calls[0]
    if not isinstance(call.get("id"), str) or not call["id"] or call.get("type") != "function":
        raise ValueError("tool call requires function type and a non-empty ID")
    if call["function"]["name"] != "bash":
        raise ValueError("wrong tool name")
    arguments = json.loads(call["function"]["arguments"])
    if arguments != {"command": "pwd"}:
        raise ValueError(f"wrong tool arguments: {arguments!r}")
    message = {"role": "assistant", "content": choice["message"].get("content"), "tool_calls": calls}
    reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
    if reasoning:
        message["reasoning_content"] = reasoning
        message["reasoning"] = reasoning
    return message


async def check(base: str, model: str, key: str, concurrency: int) -> dict:
    async with httpx.AsyncClient(
        base_url=base.rstrip("/") + "/", headers={"Authorization": f"Bearer {key}"}, timeout=300,
        limits=httpx.Limits(max_connections=concurrency + 2), trust_env=False,
    ) as client:
        response = await client.get("models")
        response.raise_for_status()
        ids = [entry["id"] for entry in response.json()["data"]]
        if model not in ids:
            raise ValueError(f"model alias missing: {model}; served: {ids}")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Call the bash tool exactly once with the command pwd. Do not answer in plain text."}],
            "tools": [TOOL], "tool_choice": "auto", "temperature": 0, "max_tokens": 256,
        }

        async def request(body: dict) -> tuple[dict, float]:
            start = time.monotonic()
            reply = await client.post("chat/completions", json=body)
            reply.raise_for_status()
            data = reply.json()
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

        # Check the submitted schemas in addition to the organizer's minimal probe.
        sys.path.insert(0, str(ROOT / "submission_stage2"))
        from custom_agent.react_bash_agent import TOOLS

        custom = dict(payload, tools=TOOLS)
        custom["messages"] = [{"role": "user", "content": 'Call bash exactly once with command "printf hello" and timeout_sec 30.'}]
        custom_data, _ = await request(custom)
        custom_calls = custom_data["choices"][0]["message"].get("tool_calls") or []
        if len(custom_calls) != 1 or custom_calls[0]["function"]["name"] != "bash" or json.loads(custom_calls[0]["function"]["arguments"]) != {"command": "printf hello", "timeout_sec": 30}:
            raise ValueError("actual bash schema failed")
        finish_payload = dict(payload, tools=TOOLS)
        finish_payload["messages"] = [{"role": "user", "content": 'Call finish exactly once with summary "protocol checked". Do not call bash.'}]
        finish_data, _ = await request(finish_payload)
        finish_calls = finish_data["choices"][0]["message"].get("tool_calls") or []
        if len(finish_calls) != 1 or finish_calls[0]["function"]["name"] != "finish" or json.loads(finish_calls[0]["function"]["arguments"]) != {"summary": "protocol checked"}:
            raise ValueError("finish tool schema failed")

        thinking = dict(payload, max_tokens=4096, temperature=1.0, top_p=0.95,
                        reasoning_effort="medium", top_k=20,
                        thinking_token_budget=3072,
                        chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True})
        start = time.monotonic()
        replies = await asyncio.gather(*(request(thinking) for _ in range(concurrency)))
        duration = time.monotonic() - start
        for data, _ in replies:
            tool_message(data)
        output_tokens = sum((data.get("usage") or {}).get("completion_tokens", 0) for data, _ in replies)
        latencies = sorted(latency for _, latency in replies)
        return {
            "model": model, "contract_check": "passed", "multi_turn_check": "passed", "custom_tool_check": "passed",
            "workload": "short-prompt API load smoke check; not a full-context benchmark",
            "contract_latency_sec": round(contract_latency, 3), "multi_turn_latency_sec": round(multi_latency, 3),
            "concurrency": concurrency, "wall_time_sec": round(duration, 3),
            "completion_tokens": output_tokens, "aggregate_output_tokens_per_second": round(output_tokens / duration, 2),
            "p50_latency_sec": round(latencies[(len(latencies) - 1) // 2], 3),
            "max_latency_sec": round(latencies[-1], 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen38-27b-mera")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    result = asyncio.run(check(args.base_url, args.model, os.environ.get("REACT_AGENT_API_KEY", "dummy"), args.concurrency))
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
