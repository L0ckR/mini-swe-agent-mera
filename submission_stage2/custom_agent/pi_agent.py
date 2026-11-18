"""Harbor adapter for a real Pi SDK session running beside the model server."""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import PurePosixPath

import httpx
from harbor.agents.base import BaseAgent
from harbor.agents.model_connection import ModelConnectionSpec

from .evidence import EvidenceTracker
from .pi_operations import PiOperations
from .policy import BudgetExpired, RunBudget
from .task_contract import TaskContract, validate_changes, write_requested_patch


GUIDANCE = """Competition constraints:
The repository is already populated at /testbed. Work directly on its implementation.
Preserve its Git history; do not clone placeholder URLs or erase/reinitialize the repository.
Read the entire issue and submission requirements. Trace the responsible implementation and
callers: a reporter's suspected cause is a hypothesis, and example snippets may not be repository files.
Use focused existing tests and read their output. For unittest use module discovery rather than
executing a file that only defines test classes. With shell search use -- before patterns starting
with a dash, for example grep -nF -- '--help' src. Keep all temporary files inside /testbed.
Do not inspect hidden evaluation files or external ready-made fixes. Leave the final source patch
on disk and report observed checks and remaining limitations. The adapter exports any required diff.
"""


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


class PiAgent(BaseAgent):
    MODEL_CONNECTION = ModelConnectionSpec(
        api_key_envs=("REACT_AGENT_API_KEY",), base_url_envs=("REACT_AGENT_BASE_URL",), passthrough=True,
    )

    def __init__(self, *args, max_steps=100, max_tokens=8192, time_budget_sec=1680,
                 context_window=65536, context_budget_tokens=None, workdir="/testbed", **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = _integer(max_steps, "max_steps", 1, 200)
        self.max_tokens = _integer(max_tokens, "max_tokens", 256, 32768)
        if isinstance(time_budget_sec, bool) or not math.isfinite(float(time_budget_sec)) or not 10 <= float(time_budget_sec) <= 1740:
            raise ValueError("time_budget_sec must be from 10 to 1740 seconds")
        self.time_budget_sec = float(time_budget_sec)
        self.context_window = _integer(context_window if context_budget_tokens is None else context_budget_tokens,
                                       "context_window", 4096, 131072)
        if self.max_tokens >= self.context_window:
            raise ValueError('max_tokens must be smaller than context_window')
        root = PurePosixPath(workdir)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("workdir must be an absolute task path")
        self.workdir = str(root)

    @staticmethod
    def name():
        return "mera-pi"

    def version(self):
        return "pi-1.5.2+mera.1"

    async def setup(self, environment):
        return None

    async def run(self, instruction, environment, context):
        connection = self.model_connection
        if not connection.base_url or not connection.api_key or not self.model_name:
            raise ValueError("Pi requires the organizer's model connection")
        origin = connection.base_url.rstrip("/").removesuffix("/v1")
        budget = RunBudget(self.time_budget_sec, 30)
        contract = TaskContract.from_instruction(instruction, self.workdir)
        operations = PiOperations(environment, budget, workdir=self.workdir)
        ignored = () if not contract.patch_path else (str(PurePosixPath(contract.patch_path).relative_to(self.workdir)),)
        evidence = EvidenceTracker(environment, self.workdir, self.logs_dir, ignored_paths=ignored)
        trajectory, completed, replies = [], {}, {}
        observed_usage = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
        session_id, token = None, ""
        stop_reason, final_export, final_validation = "started", None, None
        cancelled, repairs = False, 0

        def save():
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            records = (("pi-trajectory.json", {"harness": "native-pi", "version": self.version(), "events": trajectory,
                        "stop_reason": stop_reason, "completion": completed}),
                       ("tool-audit.json", {"operations": operations.records, "final_validation": final_validation,
                                             "patch_export": final_export}))
            for filename, value in records:
                target = self.logs_dir / filename
                temp = target.with_suffix(".json.tmp")
                temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
                temp.replace(target)

        async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(30, connect=10)) as client:
            async def request(method, path, value=None, retry=False):
                attempts = 3 if retry else 1
                for attempt in range(attempts):
                    try:
                        headers = {"Authorization": f"Bearer {connection.api_key}"}
                        if token:
                            headers["X-Mera-Session-Token"] = token
                        response = await client.request(method, origin + "/mera/pi" + path, json=value,
                                    headers=headers, timeout=min(25, budget.remaining()))
                        if response.status_code >= 400:
                            raise RuntimeError(f"Pi service HTTP {response.status_code}: {response.text[:2000]}")
                        return response.json() if response.content else {}
                    except (httpx.TransportError, TimeoutError):
                        if attempt + 1 == attempts:
                            raise
                        await asyncio.sleep(min(0.5 * (attempt + 1), budget.remaining()))

            try:
                started = await request("POST", "/sessions", {
                    "instruction": instruction, "append_system_prompt": GUIDANCE + "\n" + contract.guidance(),
                    "workdir": "/testbed", "model": self.model_name.removeprefix("openai/"),
                    "max_turns": self.max_steps, "time_budget_sec": math.ceil(self.time_budget_sec),
                    "max_output_tokens": self.max_tokens, "context_window": self.context_window,
                })
                if started.get('model') != self.model_name.removeprefix('openai/'):
                    raise RuntimeError('Pi service selected a different model than requested')
                session_id, token = started["session_id"], started["token"]
                while budget.remaining() > 15:
                    payload = await request("GET", f"/sessions/{session_id}/events", retry=True)
                    for event in payload.get("events", []):
                        kind = event.get("type")
                        if kind == "operation":
                            identifier = event.get("id")
                            if not isinstance(identifier, str) or not identifier:
                                raise RuntimeError("Invalid Pi operation identifier")
                            if identifier not in replies:
                                name, arguments = event.get("operation"), event.get("args", {})
                                is_check, before = False, None
                                if name == "exec":
                                    from .mini_agent import _observed_validation_action
                                    is_check, _ = _observed_validation_action({"name": "bash", "arguments": arguments})
                                if is_check:
                                    before = await evidence.snapshot(timeout_sec=min(8, budget.remaining()), include_diff=False)
                                reply = {"id": identifier}
                                try:
                                    result = await operations.execute(name, arguments)
                                    reply["result"] = result
                                    if before is not None:
                                        after = await evidence.snapshot(timeout_sec=min(8, budget.remaining()), include_diff=False)
                                        evidence.record_check(arguments.get("command", ""), {
                                            **result, "raw_output": str(result.get("stdout", "")) + "\n" + str(result.get("stderr", "")),
                                        }, before, after)
                                except (BudgetExpired, asyncio.CancelledError):
                                    raise
                                except Exception as exc:
                                    reply["error"] = f"{type(exc).__name__}: {exc}"
                                replies[identifier] = reply
                            await request("POST", f"/sessions/{session_id}/result", replies[identifier], retry=True)
                        elif kind == "complete":
                            completed = event
                            stop_reason = event.get("stop_reason", "completed")
                            final_validation = await validate_changes(environment, contract, timeout_sec=min(15, budget.remaining()))
                            if final_validation.get("returncode") and repairs < 2 and budget.remaining() > 45 and stop_reason == "completed":
                                repairs += 1
                                await request("POST", f"/sessions/{session_id}/continue", {"message":
                                    "Repair these observed violations of the explicit task file constraints, preserving your implementation:\n" +
                                    final_validation.get("output", "") + "\n" + json.dumps(final_validation.get("rejected_test_paths", []))})
                            else:
                                return
                        elif kind == "error":
                            raise RuntimeError("Pi worker: " + str(event.get("error", "unknown failure")))
                        else:
                            trajectory.append(event)
                            native = event.get("event", {})
                            if isinstance(native, dict) and native.get('type') == 'usage_totals':
                                for key in observed_usage:
                                    observed_usage[key] = max(observed_usage[key], int((native.get('usage') or {}).get(key, 0)))
                            if isinstance(native, dict) and native.get("type") == "message_end":
                                message = native.get("message", {})
                                if message.get("role") == "assistant":
                                    for key in observed_usage:
                                        observed_usage[key] += int((message.get("usage") or {}).get(key, 0))
                    save()
                stop_reason = "time_budget"
            except BudgetExpired:
                stop_reason = "time_budget"
            except asyncio.CancelledError:
                cancelled, stop_reason = True, "cancelled"
                raise
            except Exception:
                stop_reason = "error"
                raise
            finally:
                try:
                    if session_id:
                        try:
                            await client.delete(origin + f"/mera/pi/sessions/{session_id}", headers={"X-Mera-Session-Token": token}, timeout=8)
                        except (httpx.HTTPError, TimeoutError):
                            pass
                    if not cancelled:
                        final_validation = await validate_changes(environment, contract, timeout_sec=15)
                        final_export = await write_requested_patch(environment, contract, timeout_sec=15)
                        state = await evidence.snapshot(timeout_sec=10, include_diff=True)
                        trajectory.append({"type": "final_repository", **state,
                                           "observed_checks": evidence.summary(state.get("fingerprint"))})
                except asyncio.CancelledError:
                    cancelled, stop_reason = True, "cancelled"
                    raise
                finally:
                    save()
                    usage = {key: max(observed_usage[key], int((completed.get("usage") or {}).get(key, 0))) for key in observed_usage}
                    context.n_input_tokens = usage["input"] + usage["cacheRead"] + usage["cacheWrite"]
                    context.n_output_tokens = usage["output"]
                    context.n_cache_tokens = usage["cacheRead"]
                    context.cost_usd = 0.0
                    context.metadata = {"harness": "native-pi", "agent_version": self.version(), "stop_reason": stop_reason,
                        "model": self.model_name, "thinking": False, "requested_patch": contract.patch_path,
                        "patch_export": final_export, "file_constraints_valid": final_validation.get("valid") if final_validation else None,
                        "operations": len(operations.records), "contract_repairs": repairs}
