"""OpenAI-compatible async model port for the vendored mini-swe-agent core.

The full native history remains in the core. Compaction changes only the model's
request view, while the audit retains the original requests and responses.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Any
import uuid

import httpx

from ._vendor.mini_swe_agent.exceptions import FormatError
from .policy import BudgetExpired, RunBudget
from .tools import TOOLS

MEMORY_PREFIX = "Working checkpoint from earlier observed steps:\n"
REVIEW_PROMPT = (
    "The final-review reserve has started. Preserve the current fix. Run the most useful "
    "focused check, inspect the patch, and call finish with the actual results. Do not start "
    "a broad rewrite or lengthy dependency installation."
)


class ContextOverflow(Exception):
    """The inference server rejected the current request context length."""


class RequestBudgetExceeded(Exception):
    """All retries of a query share one bounded request deadline."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = limit * 2 // 3
    return value[:head] + f"\n... [{len(value) - limit} characters omitted] ...\n" + value[-(limit - head):]


class ForkModel:
    def __init__(
        self, base_url: str, api_key: str, model_name: str, logs_dir: Path,
        budget: RunBudget, session_id: str | None = None, max_tokens: int = 8192,
        thinking_budget: int = 3072, request_timeout_sec: float = 240,
        context_budget_tokens: int = 42000, enable_compaction: bool = True,
        reasoning_effort: str = "medium", temperature: float = 1.0,
        *, client: httpx.AsyncClient | None = None,
    ) -> None:
        for name, value in (("max_tokens", max_tokens), ("thinking_budget", thinking_budget),
                            ("context_budget_tokens", context_budget_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(request_timeout_sec) or request_timeout_sec <= 0:
            raise ValueError("request_timeout_sec must be positive and finite")
        if reasoning_effort not in {"low", "medium", "xhigh"}:
            raise ValueError("reasoning_effort must be low, medium, or xhigh")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between zero and two")
        parsed = httpx.URL(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("base_url must be an absolute HTTP(S) /v1 endpoint")
        if not api_key or not model_name:
            raise ValueError("api_key and model_name must be nonempty")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key, self.model_name = api_key, model_name.removeprefix("openai/")
        self.logs_dir, self.budget = Path(logs_dir), budget
        self.session_id = session_id
        self.max_tokens, self.thinking_budget = max_tokens, thinking_budget
        self.request_timeout_sec = float(request_timeout_sec)
        self.context_budget_tokens, self.enable_compaction = context_budget_tokens, enable_compaction
        self.reasoning_effort, self.temperature = reasoning_effort, temperature
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        self._client, self._owns_client = client, client is None
        self._memory = ""
        self._memory_boundary = 2
        self._output_limit = 6000
        self._clip_recent_outputs = False
        self._query_number = 0
        self._review_announced = False
        self._recover_next_query = False
        self._audit: dict[str, Any] = {
            "model": self.model_name, "started_at": _now(), "requests": [],
            "checkpoints": [], "usage": self.usage,
        }

    async def __aenter__(self):
        if self._client is None:
            self._client = httpx.AsyncClient(trust_env=False)
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()

    async def close(self) -> None:
        self._audit["finished_at"] = _now()
        self._save()
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _save(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._audit["state"] = {"queries": self._query_number, "memory_boundary": self._memory_boundary,
                                "phase": self.budget.phase}
        path = self.logs_dir / "model-audit.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._audit, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _record_usage(self, response: dict) -> None:
        usage = response.get("usage") or {}
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self.usage["cached_tokens"] += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)

    @staticmethod
    def _wire(message: dict) -> dict:
        allowed = {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content", "reasoning"}
        return deepcopy({key: value for key, value in message.items() if key in allowed})

    def _view(self, messages: list[dict]) -> list[dict]:
        if self._memory_boundary > len(messages):
            self._memory, self._memory_boundary = "", 2
        result = [self._wire(message) for message in messages[:2]]
        if self._memory:
            result.append({"role": "user", "content": MEMORY_PREFIX + self._memory})
        result.extend(self._wire(message) for message in messages[self._memory_boundary:])
        tool_indexes = [i for i, message in enumerate(result) if message.get("role") == "tool"]
        for index in tool_indexes if self._clip_recent_outputs else tool_indexes[:-6]:
            result[index]["content"] = _bounded(str(result[index].get("content") or ""), self._output_limit)
        return result

    @staticmethod
    def _estimate(messages: list[dict]) -> int:
        return len(json.dumps(messages, ensure_ascii=False)) // 3 + len(messages) * 8

    async def _post(self, payload: dict, deadline: float, *, kind: str) -> dict:
        if self._client is None:
            self._client = httpx.AsyncClient(trust_env=False)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.session_id:
            headers["X-Session-ID"] = self.session_id
        last_error = "request budget exhausted"
        for attempt in range(4):
            remaining = min(deadline - time.monotonic(), self.budget.remaining())
            if remaining <= 0:
                raise RequestBudgetExceeded(last_error)
            record = {"at": _now(), "kind": kind, "query": self._query_number,
                      "attempt": attempt + 1, "request": payload, "status": "pending"}
            self._audit["requests"].append(record)
            self._save()
            started = time.monotonic()
            try:
                async with asyncio.timeout(remaining):
                    response = await self._client.post(
                        self.endpoint, headers=headers, json=payload,
                        timeout=httpx.Timeout(remaining, connect=min(20.0, remaining)),
                    )
                record["http_status"] = response.status_code
                record["status"] = "http_error" if response.status_code >= 400 else "invalid_response"
                try:
                    data = response.json()
                except ValueError:
                    data = None
                record["response"] = data if data is not None else _bounded(response.text, 6000)
                if response.status_code < 400:
                    if not isinstance(data, dict):
                        raise RuntimeError("Model response is not a JSON object")
                    self._record_usage(data)
                    record["status"] = "success"
                    return data
                detail = _bounded(response.text, 3000)
                record["status"] = "http_error"
                if response.status_code == 400 and any(token in response.text.lower() for token in (
                    "context length", "maximum context", "context window", "too many tokens",
                )):
                    raise ContextOverflow(detail)
                if response.status_code not in {408, 409, 429} and not 500 <= response.status_code < 600:
                    raise RuntimeError(f"Model HTTP {response.status_code}: {detail}")
                last_error = f"Model HTTP {response.status_code}: {detail}"
            except asyncio.CancelledError:
                record["status"] = "cancelled"
                raise
            except (httpx.TimeoutException, httpx.NetworkError, TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                record.update({"status": "transport_error", "error": last_error})
            finally:
                record["elapsed_sec"] = round(time.monotonic() - started, 3)
                self._save()
            if attempt == 3:
                raise RequestBudgetExceeded(f"Four transient request attempts failed: {last_error}")
            available = min(deadline - time.monotonic(), self.budget.remaining())
            if available <= 0:
                raise RequestBudgetExceeded(last_error)
            await asyncio.sleep(min(2**attempt + random.random(), available))
        raise AssertionError("unreachable")

    async def _compact(self, messages: list[dict], deadline: float, *, emergency: bool = False) -> bool:
        body = messages[self._memory_boundary:]
        starts = [index for index, message in enumerate(body) if message.get("role") == "assistant"]
        keep = min(12, max(1, len(starts) // 2)) if emergency else 12
        if len(starts) <= keep:
            # Shrink observations as whole strings, leaving original tool arguments intact.
            old_limit = self._output_limit
            self._output_limit = min(self._output_limit, 1500)
            changed = old_limit != self._output_limit or not self._clip_recent_outputs
            self._clip_recent_outputs = True
            return changed
        boundary = self._memory_boundary + starts[-keep]
        older = [self._wire(message) for message in messages[self._memory_boundary:boundary]]
        for message in older:
            if message.get("role") == "tool":
                message["content"] = _bounded(str(message.get("content") or ""), 4000)
        source = {"prior_checkpoint": self._memory, "observed_steps": older}
        summary_messages = [
            {"role": "system", "content": (
                "Write a factual software-engineering checkpoint, at most 1200 words. Record the objective, "
                "relevant file paths/symbols, confirmed cause, exact edits made, test commands and observed "
                "results, unresolved failures and next actions. Distinguish facts from hypotheses. Do not "
                "invent passing tests or propose a new solution."
            )},
            {"role": "user", "content": "Original issue:\n" + str(messages[1].get("content") or "") +
             "\n\nEarlier evidence:\n" + json.dumps(source, ensure_ascii=False)},
        ]
        record = {"at": _now(), "boundary": boundary, "source": source}
        self._audit["checkpoints"].append(record)
        try:
            response = await self._post({
                "model": self.model_name, "messages": summary_messages,
                "temperature": 0, "max_tokens": min(2048, self.max_tokens),
                "chat_template_kwargs": {"enable_thinking": False},
            }, deadline, kind="checkpoint")
            summary = str(response["choices"][0]["message"].get("content") or "").strip()
            if not summary:
                raise RuntimeError("Checkpoint response was empty")
        except (RuntimeError, ContextOverflow, KeyError, IndexError, TypeError) as exc:
            record["error"] = str(exc)
            # The full native history/audit is retained; visible omissions prevent fabricated state.
            summary = "Checkpoint generation failed. Bounded original observations follow; inspect the repository where details are omitted.\n" + _bounded(json.dumps(source, ensure_ascii=False), 18000)
        self._memory, self._memory_boundary = summary, boundary
        record["summary"] = summary
        self._save()
        return True

    def _normalize(self, response: dict) -> dict:
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError("Model response contains no choices")
        choice = choices[0]
        raw = choice.get("message") or {}
        if not isinstance(raw, dict):
            raise RuntimeError("Model response contains no assistant object")
        message = {"role": "assistant", "content": raw.get("content")}
        reasoning = raw.get("reasoning_content") or raw.get("reasoning")
        if reasoning:
            message.update(reasoning_content=reasoning, reasoning=reasoning)
        calls, actions, seen = [], [], set()
        for index, call in enumerate(raw.get("tool_calls") or []):
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                continue
            function = dict(call["function"])
            identifier = call.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                identifier = f"call_{self._query_number}_{index}_{uuid.uuid4().hex[:10]}"
            seen.add(identifier)
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            function["arguments"] = arguments
            calls.append({"id": identifier, "type": "function", "function": function})
            actions.append({"name": str(function.get("name") or "unknown"), "arguments": arguments,
                            "tool_call_id": identifier})
        for action in actions:
            action["only_call"] = len(actions) == 1
        if calls:
            message["tool_calls"] = calls
        message["extra"] = {"actions": actions, "cost": 0.0, "finish_reason": choice.get("finish_reason"),
                            "timestamp": time.time(), "phase": self.budget.phase}
        return message

    async def query(self, messages: list[dict], **kwargs) -> dict:
        self._query_number += 1
        initial_phase = self.budget.phase
        recovering = self._recover_next_query
        self._recover_next_query = False
        cap = min(45.0, self.request_timeout_sec) if initial_phase == "review" else self.request_timeout_sec
        if recovering:
            cap = min(cap, 120.0)
        deadline = time.monotonic() + self.budget.timeout(cap)
        try:
            if self.enable_compaction and self._estimate(self._view(messages)) > self.context_budget_tokens:
                await self._compact(messages, deadline)
            for attempt in range(4):
                phase = self.budget.phase
                urgent = phase == "review" or self.budget.remaining() < 180
                effort = "low" if urgent or recovering else self.reasoning_effort
                view = self._view(messages)
                if phase == "review" and not self._review_announced:
                    view.append({"role": "user", "content": REVIEW_PROMPT})
                payload = {
                    "model": self.model_name, "messages": view, "tools": TOOLS, "tool_choice": "auto",
                    "temperature": self.temperature, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                    "max_tokens": min(self.max_tokens, 4096) if urgent or recovering else self.max_tokens,
                    "reasoning_effort": effort,
                    "thinking_token_budget": min(768 if urgent else (min(1024, self.thinking_budget) if recovering else self.thinking_budget), self.max_tokens),
                    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": effort},
                }
                try:
                    response = await self._post(payload, deadline, kind="agent")
                    if phase == "review":
                        self._review_announced = True
                    message = self._normalize(response)
                    if not message["extra"]["actions"]:
                        correction = {"role": "user", "content": (
                            f"No complete tool action was returned (finish_reason={message['extra']['finish_reason']}). "
                            "Use the available tools to continue, or call finish alone after reviewing the patch. "
                            "Keep reasoning concise enough for a tool call to fit."
                        )}
                        raise FormatError(message, correction)
                    return message
                except ContextOverflow:
                    if not self.enable_compaction or attempt == 3 or not await self._compact(messages, deadline, emergency=True):
                        raise
        except RequestBudgetExceeded as exc:
            self.budget.remaining()
            self._recover_next_query = True
            self._audit.setdefault("recoveries", []).append({"query": self._query_number,
                                                             "reason": str(exc), "at": _now()})
            if initial_phase == "solve" and self.budget.phase == "review":
                raise FormatError({"role": "user", "content": REVIEW_PROMPT, "extra": {"cost": 0.0}}) from exc
            raise FormatError({"role": "user", "content": (
                f"The model query failed within its shared {cap:g}s retry budget: {_bounded(str(exc), 1200)}. "
                "No tool actions were dispatched from that query. Existing repository edits were preserved. "
                "Continue with a concise tool call; the next request uses a smaller reasoning and output budget."
            ), "extra": {"cost": 0.0}}) from exc
        finally:
            self._save()
        raise AssertionError("unreachable")

    @staticmethod
    def format_message(**kwargs) -> dict:
        return kwargs

    @staticmethod
    def format_observation_messages(message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        if len(actions) != len(outputs):
            raise ValueError("Every tool call requires exactly one observation")
        results, submitted = [], None
        for action, output in zip(actions, outputs, strict=True):
            results.append({"role": "tool", "tool_call_id": action["tool_call_id"],
                            "name": action["name"], "content": str(output.get("output") or "")})
            if output.get("submitted") and action.get("only_call") and len(actions) == 1:
                submitted = str(output.get("summary") or "")
        if submitted is not None:
            results.append({"role": "exit", "content": submitted,
                            "extra": {"exit_status": "Submitted", "submission": submitted}})
        return results

    def get_template_vars(self, **kwargs) -> dict:
        return {"model_name": self.model_name}

    def serialize(self) -> dict:
        # Credentials, headers and host environment must never enter native logs.
        return {"info": {"config": {"model": {
            "model_name": self.model_name, "max_tokens": self.max_tokens,
            "thinking_budget": self.thinking_budget, "context_budget_tokens": self.context_budget_tokens,
            "enable_compaction": self.enable_compaction,
        }}, "model_usage": dict(self.usage)}}
