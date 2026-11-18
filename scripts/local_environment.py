"""Isolated local task execution for diagnostic runs without Docker.

Only the selected repository is writable and visible at /testbed. The host's
home directories, credentials, model service and network are not exposed.
This is a diagnostic backend, not the organizer's Docker/Harbor evaluation.
"""
from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
from types import SimpleNamespace
from typing import Mapping


class IsolationUnavailable(RuntimeError):
    """The required bubblewrap namespace could not be established."""


class LocalEnvironment:
    isolation = "bubblewrap"

    def __init__(self, repo: Path, python_bin: str | Path = "/usr/bin/python3"):
        self.repo = Path(repo).resolve(strict=True)
        if not self.repo.is_dir():
            raise ValueError("repo must name a directory")
        self.bwrap = shutil.which("bwrap")
        if self.bwrap is None:
            raise IsolationUnavailable("bubblewrap is required; unsafe host-shell fallback is disabled")
        source_python = Path(python_bin)
        if source_python.is_absolute():
            try:
                relative_python = source_python.relative_to(self.repo)
            except ValueError:
                if not str(source_python).startswith("/usr/") and not str(source_python).startswith("/testbed/"):
                    raise ValueError("python_bin must be in /usr or the selected task repository") from None
                self.python_bin = str(source_python)
            else:
                self.python_bin = "/testbed/" + relative_python.as_posix()
        else:
            if ".." in source_python.parts:
                raise ValueError("python_bin must remain inside the task repository")
            self.python_bin = "/testbed/" + source_python.as_posix()

    def _command(self, command: str, cwd: str, env: Mapping[str, str] | None) -> list[str]:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be nonempty text")
        target = PurePosixPath(cwd)
        if not target.is_absolute() or ".." in target.parts or not (target == PurePosixPath("/testbed") or PurePosixPath("/testbed") in target.parents):
            raise ValueError("cwd must remain inside /testbed")
        argv = [
            self.bwrap, "--unshare-user", "--unshare-pid", "--unshare-net",
            "--unshare-ipc", "--unshare-uts", "--die-with-parent",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--dir", "/testbed",
            "--bind", str(self.repo), "/testbed", "--clearenv",
            # /testbed is a separate mount and remains writable after the
            # otherwise-empty private root becomes read-only.
            "--remount-ro", "/",
        ]
        task_env = {
            "PATH": str(PurePosixPath(self.python_bin).parent) + ":/usr/bin:/bin",
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": "/testbed", "PAGER": "cat", "GIT_PAGER": "cat",
        }
        for key, value in (env or {}).items():
            if not isinstance(key, str) or not isinstance(value, str) or not key or "=" in key or "\0" in key + value:
                raise ValueError("environment entries require valid string names and values")
            if key in {"HOME", "CODEX_HOME"}:
                raise ValueError("the local diagnostic backend does not set home-directory variables")
            task_env[key] = value
        for key, value in task_env.items():
            argv.extend(["--setenv", key, value])
        argv.extend(["--chdir", str(target), "/bin/bash", "--noprofile", "--norc", "-c", command])
        return argv

    @staticmethod
    def _kill_group(process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def exec(self, command: str, cwd: str = "/testbed", timeout_sec: float = 120,
                   env: Mapping[str, str] | None = None, **kwargs) -> SimpleNamespace:
        if kwargs:
            raise ValueError("unsupported local execution options: " + ", ".join(sorted(kwargs)))
        if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)) or not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive and finite")
        argv = self._command(command, cwd, env)
        # Starting a process group lets cancellation kill the launcher and its
        # descendants. Ending the PID namespace also kills detached descendants.
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communication = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout=timeout_sec)
        except TimeoutError:
            timed_out = True
            self._kill_group(process)
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout=5)
        except asyncio.CancelledError:
            self._kill_group(process)
            try:
                await asyncio.wait_for(asyncio.shield(communication), timeout=5)
            except (TimeoutError, asyncio.CancelledError):
                communication.cancel()
            raise
        out, err = stdout.decode(errors="replace"), stderr.decode(errors="replace")
        if not timed_out and process.returncode and err.startswith("bwrap:"):
            raise IsolationUnavailable(
                "Local task isolation failed; unsafe host-shell fallback is disabled. " + err.strip()
            )
        if timed_out:
            err += f"\nLocal command timed out after {timeout_sec:g}s; its process group was killed. Completion is unconfirmed.\n"
        return SimpleNamespace(stdout=out, stderr=err, return_code=124 if timed_out else process.returncode,
                               timed_out=timed_out)
