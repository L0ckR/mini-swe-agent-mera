#!/usr/bin/env python3
"""One live-agent attempt on the public Click statement, without its hidden verifier.

The independent checks below follow the public issue text. They are not the SWE
MERA evaluation tests and the reported result is not competition pass@1.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.local_environment import IsolationUnavailable, LocalEnvironment

ARCHIVE_URL = "https://codeload.github.com/pallets/click/tar.gz/d73083e"
PUBLIC_TEST_COMMAND = "python3 -B -m pytest -q tests/test_options.py -k help"

ACCEPTANCE_SCRIPT = r'''
import json
import re
import shlex
import click
from click.testing import CliRunner

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    pass

@click.command()
@click.argument("required_arg")
@click.option("--host", "-h", "host", default="localhost")
def foo(required_arg, host):
    click.echo(f"Foo command required_arg={required_arg} host={host}")

@click.command()
@click.argument("required_arg")
def bar(required_arg):
    click.echo(required_arg)

cli.add_command(foo)
cli.add_command(bar)
runner = CliRunner()
missing = runner.invoke(cli, ["foo"], prog_name="click-test-cli")
help_page = runner.invoke(cli, ["foo", "--help"], prog_name="click-test-cli")
host = runner.invoke(cli, ["foo", "value", "-h", "example.test"], prog_name="click-test-cli")
bar_missing = runner.invoke(cli, ["bar"], prog_name="click-test-cli")
root_help = runner.invoke(cli, ["-h"], prog_name="click-test-cli")

def suggested_help_works(result):
    match = re.search(r"Try '([^']+)' for help\.", result.output)
    if match is None:
        return False
    invocation = shlex.split(match.group(1))
    return bool(invocation and runner.invoke(cli, invocation[1:], prog_name="click-test-cli").exit_code == 0)

checks = [
    ("missing_argument_hint_uses_unshadowed_long_help", missing.exit_code == 2 and "Try 'click-test-cli foo --help' for help." in missing.output),
    ("suggested_foo_help_is_executable", suggested_help_works(missing)),
    ("explicit_long_help_still_works", help_page.exit_code == 0 and "--help" in help_page.output and "--host" in help_page.output),
    ("short_host_option_keeps_its_meaning", host.exit_code == 0 and "host=example.test" in host.output),
    ("unshadowed_command_keeps_executable_help_hint", bar_missing.exit_code == 2 and suggested_help_works(bar_missing)),
    ("group_short_help_alias_is_preserved", root_help.exit_code == 0 and "foo" in root_help.output),
]
result = {"passed": sum(bool(value) for _, value in checks), "total": len(checks),
          "checks": [{"criterion": name, "passed": bool(value)} for name, value in checks],
          "observations": {"missing_argument": missing.output, "help": help_page.output,
                           "host_option": host.output, "unshadowed_missing_argument": bar_missing.output}}
print(json.dumps(result))
raise SystemExit(0 if result["passed"] == result["total"] else 1)
'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_repo_file(repo: Path, name: str) -> bool:
    path = repo / name
    if path.is_symlink():
        return False
    try:
        path.resolve().relative_to(repo.resolve())
    except (ValueError, OSError):
        return False
    return path.is_file()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True).stdout


def test_hashes(repo: Path) -> dict[str, str]:
    if (repo / "tests").is_symlink():
        return {}
    return {path.relative_to(repo).as_posix(): digest(path) for path in sorted((repo / "tests").rglob("*"))
            if not path.is_symlink() and path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def test_modes(repo: Path) -> dict[str, int]:
    if (repo / "tests").is_symlink():
        return {}
    return {path.relative_to(repo).as_posix(): stat.S_IMODE(path.stat().st_mode)
            for path in sorted((repo / "tests").rglob("*")) if not path.is_symlink() and path.is_file()
            and "__pycache__" not in path.parts and path.suffix != ".pyc"}


class ClickEnvironment(LocalEnvironment):
    """Use the prepared repo-local pytest environment and its src checkout."""

    def __init__(self, repo: Path):
        super().__init__(repo, python_bin=repo / ".venv/bin/python3")

    async def exec(self, command, cwd="/testbed", timeout_sec=120, env=None, **kwargs):
        task_env = {"PYTHONPATH": "/testbed/src", **(env or {})}
        return await super().exec(command, cwd=cwd, timeout_sec=timeout_sec, env=task_env, **kwargs)


async def inspect_public_acceptance(environment: ClickEnvironment) -> dict:
    encoded = base64.b64encode(ACCEPTANCE_SCRIPT.encode()).decode()
    command = "python3 -B -c \"import base64;exec(base64.b64decode('" + encoded + "'))\""
    try:
        result = await environment.exec(command, timeout_sec=30)
        try:
            details = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            details = {"passed": 0, "total": 6, "error": "public reproducer did not emit its result JSON"}
        return {"returncode": result.return_code, "stdout": result.stdout, "stderr": result.stderr, **details}
    except Exception as exc:
        return {"returncode": None, "passed": 0, "total": 6, "error": type(exc).__name__ + ": " + str(exc)}


async def inspect_visible_tests(environment: ClickEnvironment) -> dict:
    try:
        result = await environment.exec(PUBLIC_TEST_COMMAND, timeout_sec=120)
        match = re.search(r"(\d+) passed", result.stdout + result.stderr)
        return {"command": PUBLIC_TEST_COMMAND, "returncode": result.return_code,
                "passed_count": int(match.group(1)) if match else 0,
                "stdout": result.stdout, "stderr": result.stderr}
    except Exception as exc:
        return {"command": PUBLIC_TEST_COMMAND, "returncode": None, "passed_count": 0,
                "error": type(exc).__name__ + ": " + str(exc)}


def check_patch_paths(patch: str, numstat: str) -> list[str]:
    violations = []
    for line in numstat.splitlines():
        fields = line.split("\t", 2)
        if len(fields) != 3:
            violations.append("unrecognized patch numstat")
            continue
        added, removed, name = fields
        parts = Path(name).parts
        if added == "-" or removed == "-":
            violations.append("binary patch: " + name)
        if "tests" in parts or "test" in parts or Path(name).name == "conftest.py":
            violations.append("test file in delivered patch: " + name)
        if name == "change.diff" or any(part in {".venv", "__pycache__", ".pytest_cache"} for part in parts):
            violations.append("generated artifact in delivered patch: " + name)
    if "GIT binary patch" in patch:
        violations.append("binary patch body")
    return violations


async def inspect_delivered_patch(base: Path, repo: Path, output: Path) -> dict:
    delivered = repo / "change.diff"
    if delivered.is_symlink() or not delivered.is_file() or not delivered.stat().st_size:
        return {"valid": False, "error": "task requires a nonempty /testbed/change.diff; only this file is collected"}
    try:
        patch = delivered.read_text()
    except (OSError, UnicodeError) as exc:
        return {"valid": False, "error": "cannot read delivered change.diff: " + str(exc)}
    (output / "delivered_change.diff").write_text(patch)
    target = output / "delivered_patch_repo"
    shutil.copytree(base, target, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    (target / "change.diff").write_text(patch)
    environment = ClickEnvironment(target)
    try:
        stat_result = await environment.exec("git apply --numstat change.diff", timeout_sec=20)
        check = await environment.exec("git apply --check change.diff", timeout_sec=20)
        violations = check_patch_paths(patch, stat_result.stdout)
        valid = stat_result.return_code == 0 and check.return_code == 0 and bool(stat_result.stdout.strip()) and not violations
        report = {"valid": valid, "sha256": digest(delivered), "numstat": stat_result.stdout,
                  "apply_check_returncode": check.return_code, "apply_check_stderr": check.stderr,
                  "violations": violations}
        if valid:
            applied = await environment.exec("git apply change.diff", timeout_sec=20)
            report["apply_returncode"] = applied.return_code
            if applied.return_code == 0:
                report["public_acceptance"] = await inspect_public_acceptance(environment)
                report["visible_tests"] = await inspect_visible_tests(environment)
            else:
                report.update(valid=False, apply_stderr=applied.stderr)
        return report
    except Exception as exc:
        return {"valid": False, "error": type(exc).__name__ + ": " + str(exc)}


async def run(args) -> dict:
    source = args.source.resolve()
    base = args.base_repo.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output directory already exists; use a fresh name to preserve the original attempt")
    if not (base / ".venv/bin/python3").is_file() or not (base / "src/click/core.py").is_file():
        raise ValueError("prepare the declared immutable Click checkout and repo-local pytest .venv first")
    statement = args.statement.read_text()
    output.mkdir(parents=True)
    repo = output / "repo"
    shutil.copytree(base, repo, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    source_snapshot = output / "source"
    shutil.copytree(source, source_snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sys.path.insert(0, str(source_snapshot))
    from custom_agent.react_bash_agent import ReactBashAgent
    from harbor.models.agent.context import AgentContext

    environment = ClickEnvironment(repo)
    original_tests = test_hashes(repo)
    original_test_modes = test_modes(repo)
    initial_head = git(repo, "rev-parse", "HEAD").strip()
    (output / "statement.txt").write_text(statement)
    report = {
        "kind": "NON_HARBOR_PUBLIC_CLICK_DIAGNOSTIC", "official_verifier": "NOT_RUN", "pass_at_1": None,
        "public_statement": "py-009", "base_archive": ARCHIVE_URL, "base_local_head": initial_head,
        "source_hashes": {path.relative_to(source_snapshot).as_posix(): digest(path)
                          for path in sorted(source_snapshot.rglob("*.py"))},
        "model_alias": args.model, "base_url": args.base_url, "attempts": 1,
        "settings": {name: getattr(args, name) for name in ("time_budget_sec", "max_steps", "max_tokens", "request_timeout_sec", "context_budget_tokens")},
        "notes": ["Checks reproduce only behavior stated in the public issue and ordinary repository tests.",
                  "No gold patch, official verifier or hidden evaluation tests were accessed.",
                  "The production model/server/hardware differ from this local diagnostic."],
    }
    try:
        import httpx
        with httpx.Client(timeout=5, trust_env=False) as client:
            info = client.get(args.base_url.rstrip("/").removesuffix("/v1") + "/local-info")
            info.raise_for_status()
            report["local_inference"] = info.json()
    except Exception as exc:
        report["local_inference"] = {"metadata_unavailable": type(exc).__name__}
    report["before"] = {"public_acceptance": await inspect_public_acceptance(environment),
                        "visible_tests": await inspect_visible_tests(environment)}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    context = AgentContext()
    agent = ReactBashAgent(
        logs_dir=output / "agent", model_name="openai/" + args.model, workdir="/testbed",
        extra_env={"REACT_AGENT_BASE_URL": args.base_url, "REACT_AGENT_API_KEY": os.environ.get("REACT_AGENT_API_KEY", "dummy")},
        time_budget_sec=args.time_budget_sec, max_steps=args.max_steps, max_tokens=args.max_tokens,
        thinking_budget_tokens=512, request_timeout_sec=args.request_timeout_sec,
        context_budget_tokens=args.context_budget_tokens,
    )
    print("START public Click py-009: one live-model attempt", flush=True)
    start = time.monotonic()
    error = None
    try:
        await asyncio.wait_for(agent.run(statement, environment, context), timeout=args.time_budget_sec + 20)
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    report["elapsed_sec"] = round(time.monotonic() - start, 3)
    report["agent_error"] = error
    report["context"] = context.model_dump(mode="json")
    report["after"] = {"public_acceptance": await inspect_public_acceptance(environment),
                       "visible_tests": await inspect_visible_tests(environment)}
    report["modified_original_test_files"] = [path for path, sha in original_tests.items()
                                               if not safe_repo_file(repo, path) or digest(repo / path) != sha]
    report["original_test_mode_changes"] = [path for path, mode in original_test_modes.items()
                                              if not safe_repo_file(repo, path)
                                              or stat.S_IMODE((repo / path).stat().st_mode) != mode]
    report["added_test_files"] = sorted(set(test_hashes(repo)) - set(original_tests))
    report["test_symlinks"] = ["tests"] if (repo / "tests").is_symlink() else [path.relative_to(repo).as_posix() for path in (repo / "tests").rglob("*") if path.is_symlink()]
    try:
        # Post-agent Git config and filters are untrusted; execute even grading
        # Git commands inside the same task namespace rather than on the host.
        head = await environment.exec("git rev-parse HEAD", timeout_sec=20)
        patch_result = await environment.exec("git diff --binary HEAD --", timeout_sec=20)
        test_diff = await environment.exec("git diff --name-only HEAD -- tests", timeout_sec=20)
        if head.return_code or patch_result.return_code or test_diff.return_code:
            raise ValueError("repository metadata or diff unavailable after the attempt")
        report["head_changed"] = head.stdout.strip() != initial_head
        working_patch = patch_result.stdout
        (output / "working_tree.diff").write_text(working_patch)
        report["working_patch_nonempty"] = bool(working_patch.strip())
        report["test_paths_changed_in_git"] = test_diff.stdout.splitlines()
    except Exception as exc:
        report.update(head_changed=True, working_patch_nonempty=False, repository_error=str(exc))
    report["delivered_patch"] = await inspect_delivered_patch(base, repo, output)
    delivered = report["delivered_patch"]
    delivered_acceptance = delivered.get("public_acceptance", {})
    delivered_tests = delivered.get("visible_tests", {})
    report["resolved_public_diagnostic"] = bool(
        error is None and not report["head_changed"] and not report["modified_original_test_files"]
        and not report["original_test_mode_changes"] and not report["test_symlinks"]
        and not report.get("test_paths_changed_in_git") and not report["added_test_files"] and delivered.get("valid")
        and delivered_acceptance.get("passed") == delivered_acceptance.get("total") == 6
        and delivered_tests.get("returncode") == 0 and delivered_tests.get("passed_count", 0) > 0
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("kind", "elapsed_sec", "agent_error", "resolved_public_diagnostic")}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-repo", type=Path, default=ROOT / "local_runs/click-public-base")
    parser.add_argument("--statement", type=Path, default=ROOT / "data/public_statements/py-009.md")
    parser.add_argument("--source", type=Path, default=ROOT / "submission_stage2")
    parser.add_argument("--output", type=Path, default=ROOT / "jobs/public-click-live-001")
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="qwen35-local")
    parser.add_argument("--time-budget-sec", type=float, default=600)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout-sec", type=float, default=120)
    parser.add_argument("--context-budget-tokens", type=int, default=8000)
    args = parser.parse_args()
    if min(args.time_budget_sec, args.max_steps, args.max_tokens, args.request_timeout_sec, args.context_budget_tokens) <= 0:
        parser.error("budgets must be positive")
    try:
        asyncio.run(run(args))
    except (ValueError, OSError, IsolationUnavailable) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
