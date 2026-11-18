"""Harbor entry point for the pinned asynchronous mini-swe-agent fork.

The vendored DefaultAgent owns the agent loop. This module adapts its model and
environment ports to the organizer's inference endpoint and task container.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import time
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.model_connection import ModelConnectionSpec
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ._vendor.mini_swe_agent.agents.default import DefaultAgent
from .model_client import ForkModel
from .policy import BudgetExpired, RunBudget
from .tools import TOOLS, TaskTools


SYSTEM_PROMPT = """You are an autonomous software engineer running mini-swe-agent for a real repository issue.
Implement the requested behavior in /testbed and leave a minimal, correct patch on disk.
Read repository instructions and relevant source before editing. Trace callers and edge cases.
Reproduce the issue when practical, fix its cause, then run focused repository tests. Distinguish
actual behavior failures from missing dependencies or infrastructure errors. Read test output
and revise your hypothesis; never claim a test passed without tool evidence.

You have bash, read_file, edit_file, create_file and finish tools. Prefer read_file for targeted
source ranges and edit_file for exact replacement; old_text must match exactly once. Use bash
for search, project-specific tools and tests. Commands run in fresh shells rooted at /testbed.
Preserve indentation and existing conventions. Use narrow searches and avoid dumping whole trees.
Independent reads may share a response. Execute changes and their dependent tests sequentially.

Keep track of confirmed findings, rejected hypotheses, changed files and remaining checks.
Keep reasoning concise and take an action. If an edit fails, read the exact current text before
retrying. If tests fail, investigate the cause rather than repeating an unchanged command loop.
Do not spend the task budget on optional installs or an unrelated rewrite. Preserve working fixes.

Use ordinary repository tests. Do not inspect hidden evaluation tests, reference/gold patches,
organizer files or external ready-made solutions. Do not weaken checks or hardcode task examples.
Do not commit or reset the repository. Preserve pre-existing work and avoid unrelated artifacts.
New implementation files must be visible to git diff (git add -N -- <new-files> if needed).
Before finishing, inspect the full relevant diff and run focused checks. Call finish alone with
a concise summary of edits, observed tests and unresolved limitations. Finish submits the current
patch; a text-only answer does not finish the task. Time reminders are authoritative: use the
reserved review phase for useful validation and cleanup, then finish with the best current patch.
"""

INSTANCE_TEMPLATE = "Please solve this issue:\n\n{{ task }}"
SNAPSHOT_COMMAND = 'git diff --check HEAD\nmera_diff_status=$?\ngit status --short\ngit diff --stat HEAD\nexit "$mera_diff_status"'


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number, not boolean")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return parsed


def _positive_integer(value, name):
    parsed = _positive(value, name)
    if not parsed.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(parsed)


class HarborTools:
    """mini-swe-agent environment port; all task operations use Harbor exec."""

    def __init__(self, environment, budget: RunBudget, logs_dir: Path, *, workdir="/testbed",
                 review_gate=True, max_output_chars=20000):
        self.environment = environment
        self.budget = budget
        self.logs_dir = logs_dir
        self.workdir = workdir
        self.tools = TaskTools(environment, workdir=workdir, max_output_chars=max_output_chars)
        self.review_gate = review_gate
        self.actions: list[dict] = []
        self.snapshots: list[dict] = []
        self._save()

    def _save(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        target = self.logs_dir / "tool-audit.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"actions": self.actions, "snapshots": self.snapshots}, ensure_ascii=False, indent=2))
        temporary.replace(target)

    async def final_snapshot(self) -> tuple[bool, str]:
        record = {"command": SNAPSHOT_COMMAND, "passed": None}
        self.snapshots.append(record)
        self._save()
        try:
            timeout = min(20, self.budget.remaining())
            async with asyncio.timeout(timeout):
                result = await self.environment.exec(command=SNAPSHOT_COMMAND, cwd=self.workdir, timeout_sec=timeout)
            output = f"exit_code: {result.return_code}\n{result.stdout or ''}\n{result.stderr or ''}"
            passed = result.return_code == 0
        except asyncio.CancelledError:
            record["output"] = "Final check was cancelled; completion is unconfirmed"
            self._save()
            raise
        except Exception as exc:
            output, passed = f"Final check unconfirmed: {type(exc).__name__}: {exc}", False
        record.update(passed=passed, output=output)
        self._save()
        return passed, output

    async def execute(self, action: dict) -> dict:
        record = {"action": action, "started_monotonic": time.monotonic(), "phase": self.budget.phase}
        self.actions.append(record)
        self._save()
        try:
            name = action.get("name")
            if name == "finish" and not action.get("only_call", True):
                result = {"output": "Call finish alone after completing other tool calls.", "returncode": 1}
            else:
                timeout = self.budget.timeout(300)
                async with asyncio.timeout(timeout):
                    result = await self.tools.execute(action, timeout_sec=timeout)
                if result.get("submitted") and self.review_gate:
                    passed, evidence = await self.final_snapshot()
                    if not passed and len(self.snapshots) == 1:
                        result = {"output": "Final diff check needs attention. Inspect its output, correct relevant problems, then finish again. Edits preserved.\n" + evidence, "returncode": 1}
                    else:
                        result["output"] = result.get("output", "") + "\nFinal snapshot:\n" + evidence
                        result["review_passed"] = passed
            record["result"] = result
            return result
        except TimeoutError:
            result = {"output": "Tool reached its time limit. Completion is unconfirmed; inspect state before continuing. Use the remaining time for focused review and finish.", "returncode": 124}
            record["result"] = result
            return result
        except asyncio.CancelledError:
            record["cancelled"] = True
            raise
        finally:
            record["elapsed_sec"] = round(time.monotonic() - record["started_monotonic"], 3)
            self._save()

    def get_template_vars(self, **kwargs):
        return {"system": "Linux", "release": "", "version": "", "machine": "x86_64", **kwargs}

    def serialize(self):
        return {"info": {"environment": "Harbor environment.exec", "workdir": self.workdir}}


class ReactBashAgent(BaseAgent):
    MODEL_CONNECTION = ModelConnectionSpec(
        api_key_envs=("REACT_AGENT_API_KEY",), base_url_envs=("REACT_AGENT_BASE_URL",), passthrough=True,
    )

    def __init__(self, *args: Any, max_steps=100, max_tokens=8192, time_budget_sec=1680,
                 final_reserve_sec=90, request_timeout_sec=240, thinking_budget_tokens=3072,
                 context_budget_tokens=42000, profile="mera", workdir="/testbed", **kwargs: Any):
        super().__init__(*args, **kwargs)
        if profile not in {"mera", "ported"}:
            raise ValueError("profile must be mera or ported")
        self.profile = profile
        self.max_steps = _positive_integer(max_steps, "max_steps")
        self.max_tokens = _positive_integer(max_tokens, "max_tokens")
        self.time_budget_sec = _positive(time_budget_sec, "time_budget_sec")
        self.final_reserve_sec = float(final_reserve_sec)
        if not math.isfinite(self.final_reserve_sec) or self.final_reserve_sec < 0:
            raise ValueError("final_reserve_sec must be nonnegative and finite")
        self.request_timeout_sec = _positive(request_timeout_sec, "request_timeout_sec")
        self.thinking_budget_tokens = _positive_integer(thinking_budget_tokens, "thinking_budget_tokens")
        self.context_budget_tokens = _positive_integer(context_budget_tokens, "context_budget_tokens")
        self.workdir = workdir

    @staticmethod
    def name():
        return "mera-mini-swe-agent"

    def version(self):
        return "2.4.6+mera.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext):
        access = self.model_connection
        if not access.base_url or not access.api_key:
            raise ValueError("Configure REACT_AGENT_BASE_URL and REACT_AGENT_API_KEY for local inference")
        if not self.model_name:
            raise ValueError("A model name is required")
        improved = self.profile == "mera"
        budget = RunBudget(self.time_budget_sec, self.final_reserve_sec if improved else 0)
        env = HarborTools(environment, budget, self.logs_dir, workdir=self.workdir, review_gate=improved)
        model = ForkModel(
            base_url=access.base_url, api_key=access.api_key, model_name=self.model_name.removeprefix("openai/"),
            logs_dir=self.logs_dir, budget=budget, session_id=self.session_id,
            max_tokens=self.max_tokens, thinking_budget=self.thinking_budget_tokens,
            request_timeout_sec=self.request_timeout_sec, context_budget_tokens=self.context_budget_tokens,
            enable_compaction=improved,
        )
        core = DefaultAgent(
            model, env, system_template=SYSTEM_PROMPT, instance_template=INSTANCE_TEMPLATE,
            step_limit=self.max_steps, cost_limit=0, wall_time_limit_seconds=math.ceil(self.time_budget_sec),
            max_consecutive_format_errors=4, output_path=self.logs_dir / "mini-swe-agent.json",
        )
        stop_reason, summary = "started", ""
        try:
            async with model:
                result = await core.run(instruction)
            stop_reason = result.get("exit_status", "unknown")
            summary = result.get("submission", "")
        except BudgetExpired:
            stop_reason = "time_budget"
        except asyncio.CancelledError:
            stop_reason = "cancelled"
            raise
        except Exception:
            stop_reason = "error"
            raise
        finally:
            # Also leave a final state snapshot when the core reaches a step or
            # format-error limit. Never pretend this is a semantic test result.
            try:
                if stop_reason != "cancelled" and not env.snapshots and budget.deadline - time.monotonic() > 0.1:
                    await env.final_snapshot()
            except asyncio.CancelledError:
                stop_reason = "cancelled"
                raise
            finally:
                core.save(self.logs_dir / "mini-swe-agent.json", {"info": {"mera_stop_reason": stop_reason}})
                usage = model.usage
                context.n_input_tokens = usage["prompt_tokens"]
                context.n_output_tokens = usage["completion_tokens"]
                context.n_cache_tokens = usage["cached_tokens"]
                context.cost_usd = 0.0
                context.metadata = {"steps": core.n_calls, "stop_reason": stop_reason, "final_answer": summary,
                                    "agent_version": self.version(), "profile": self.profile,
                                    "upstream_commit": "04d809ceab9df28f9adaed044884180159172930",
                                    "final_check_passed": env.snapshots[-1]["passed"] if env.snapshots else None}
