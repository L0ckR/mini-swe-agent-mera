"""Typed repository tools whose filesystem operations stay inside Harbor's task.

The host builds a command containing only base64-encoded source and JSON. The
helper resolves and operates on files in the task container, never on the host.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import shlex
from typing import Any


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False},
    }}


TOOLS = [
    _tool("bash", "Run a shell command in the task repository. Each call starts a fresh shell.", {
        "command": {"type": "string"},
        "timeout_sec": {"type": "integer", "minimum": 1, "maximum": 300,
                        "description": "Optional command timeout, default 120 seconds."},
    }, ["command"]),
    _tool("read_file", "Read UTF-8 repository text with line numbers; at most 400 lines per call.", {
        "path": {"type": "string"},
        "start_line": {"type": "integer", "minimum": 1, "description": "First line, default 1."},
        "end_line": {"type": "integer", "minimum": 1,
                     "description": "Inclusive final line, default start_line + 199; window at most 400 lines."},
    }, ["path"]),
    _tool("edit_file", "Replace exactly one occurrence of old_text in an existing UTF-8 file. Whitespace is exact; ambiguous or missing text leaves the file unchanged.", {
        "path": {"type": "string"}, "old_text": {"type": "string"},
        "new_text": {"type": "string"},
    }, ["path", "old_text", "new_text"]),
    _tool("create_file", "Create a new UTF-8 repository file, including parent directories. Fails if the path exists. Makes only this new file visible in git diff using git add -N.", {
        "path": {"type": "string"}, "content": {"type": "string"},
    }, ["path", "content"]),
    _tool("finish", "Submit the current patch after focused tests and diff review. Call this tool alone.", {
        "summary": {"type": "string", "description": "Changes, actual checks and observed results, remaining limitations."},
    }, ["summary"]),
]


_FILE_HELPER = r'''
import base64
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


def run(payload):
    root = Path(payload["workdir"]).resolve(strict=True)
    args = payload["arguments"]
    requested = Path(args["path"])
    if ".." in requested.parts:
        raise ValueError("parent traversal is not allowed; use a path inside the repository")
    candidate = requested if requested.is_absolute() else root / requested
    path = candidate.resolve(strict=False)
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError("path resolves outside the task repository") from None
    if path == root:
        raise ValueError("path must name a file, not the repository directory")
    operation = payload["operation"]
    limit = payload["max_file_bytes"]
    if operation in {"read_file", "edit_file"}:
        if not path.is_file():
            raise ValueError("file does not exist or is not a regular file")
        if path.stat().st_size > limit:
            raise ValueError("file exceeds the tool byte limit; use a focused bash command")
        raw = path.read_bytes()
        if b"\0" in raw:
            raise ValueError("binary files are unsupported; this tool handles UTF-8 text")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("file is not valid UTF-8 text") from None
    if operation == "read_file":
        lines = content.splitlines(keepends=True)
        start = args.get("start_line", 1)
        end = args.get("end_line", start + 199)
        if not lines:
            return {"output": str(relative) + ": empty file (0 lines).", "returncode": 0, "total_lines": 0}
        if start > len(lines):
            return {"output": str(relative) + ": requested start line " + str(start) + " is beyond EOF (" + str(len(lines)) + " lines); no lines returned.", "returncode": 0, "total_lines": len(lines)}
        last = min(end, len(lines))
        numbered = []
        for index in range(start - 1, last):
            line = lines[index].rstrip("\r\n")
            numbered.append(str(index + 1) + " | " + line)
        return {"output": str(relative) + ": lines " + str(start) + "-" + str(last) + " of " + str(len(lines)) + "\n" + "\n".join(numbered), "returncode": 0, "total_lines": len(lines)}
    if operation == "edit_file":
        old, new = args["old_text"], args["new_text"]
        first = content.find(old)
        if first < 0:
            raise ValueError("old_text was not found; read the current file and use an exact match")
        if content.find(old, first + 1) >= 0:
            raise ValueError("old_text matches more than once; include more surrounding context")
        if old == new:
            return {"output": str(relative) + ": old_text equals new_text; file unchanged.", "returncode": 0, "changed": False}
        replacement = content[:first] + new + content[first + len(old):]
        encoded = replacement.encode("utf-8")
        if len(encoded) > limit:
            raise ValueError("edited file exceeds the tool byte limit")
        mode = stat.S_IMODE(path.stat().st_mode)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".mera-edit-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(mode)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        line = content.count("\n", 0, first) + 1
        return {"output": str(relative) + ": replaced one exact occurrence at line " + str(line) + ".", "returncode": 0, "changed": True}
    if operation == "create_file":
        if os.path.lexists(candidate):
            raise ValueError("path already exists; use edit_file for an existing file")
        encoded = args["content"].encode("utf-8")
        if len(encoded) > limit:
            raise ValueError("new file exceeds the tool byte limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".mera-create-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o644)
            # A same-directory link creates the complete file atomically and
            # refuses to overwrite a path created concurrently.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        result = {"output": str(relative) + ": created UTF-8 file.", "returncode": 0, "changed": True, "git_intent_to_add": False}
        try:
            tracked = subprocess.run(["git", "add", "-N", "--", str(relative)], cwd=root, capture_output=True, text=True, timeout=5)
            if tracked.returncode == 0:
                result["git_intent_to_add"] = True
                result["output"] += " Added intent-to-add for this file only."
            else:
                result["output"] += " Warning: git add -N failed; file exists but may be absent from the patch. " + (tracked.stderr or tracked.stdout).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["output"] += " Warning: git add -N was not confirmed: " + str(exc)
        return result
    raise ValueError("unsupported file operation")


try:
    payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    result = run(payload)
except Exception as exc:
    result = {"output": "File tool error: " + type(exc).__name__ + ": " + str(exc), "returncode": 1, "changed": False}
print(json.dumps(result, ensure_ascii=False))
raise SystemExit(result["returncode"])
'''


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + f"\n... [{len(text) - limit} characters omitted] ...\n" + text[-tail:]


class TaskTools:
    def __init__(self, environment: Any, workdir: str = "/testbed", *, max_output_chars: int = 20000,
                 max_file_bytes: int = 4 * 1024 * 1024, command_timeout_sec: int = 120) -> None:
        if not isinstance(workdir, str) or not workdir.startswith("/"):
            raise ValueError("workdir must be an absolute path in the task container")
        for name, value in (("max_output_chars", max_output_chars), ("max_file_bytes", max_file_bytes),
                            ("command_timeout_sec", command_timeout_sec)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.environment, self.workdir = environment, workdir
        self.max_output_chars, self.max_file_bytes = max_output_chars, max_file_bytes
        self.command_timeout_sec = min(command_timeout_sec, 300)

    def _result(self, output: str, returncode: int, **extra: Any) -> dict[str, Any]:
        return {**extra, "output": _bounded(output, self.max_output_chars), "raw_output": output, "returncode": returncode}

    @staticmethod
    def _action(action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if not isinstance(action, dict):
            raise ValueError("action must be an object")
        function = action.get("function") if isinstance(action.get("function"), dict) else action
        name = function.get("name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError("action needs a tool name and JSON object arguments")
        definition = next((tool["function"] for tool in TOOLS if tool["function"]["name"] == name), None)
        if definition is None:
            raise ValueError(f"unknown tool {name!r}")
        schema = definition["parameters"]
        missing = set(schema["required"]) - set(arguments)
        extra = set(arguments) - set(schema["properties"])
        if missing or extra:
            raise ValueError(f"invalid {name} arguments: missing={sorted(missing)}, unexpected={sorted(extra)}")
        for field, value in arguments.items():
            expected = schema["properties"][field]
            if expected["type"] == "string" and not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
            if expected["type"] == "integer" and (isinstance(value, bool) or not isinstance(value, int)
                    or value < expected.get("minimum", value) or value > expected.get("maximum", value)):
                raise ValueError(f"{field} must be an integer within the tool's range")
        if name == "read_file":
            start, end = arguments.get("start_line", 1), arguments.get("end_line", arguments.get("start_line", 1) + 199)
            if end < start or end - start + 1 > 400:
                raise ValueError("read_file requires end_line >= start_line and at most 400 lines")
        if name == "edit_file" and not arguments["old_text"]:
            raise ValueError("old_text must be nonempty")
        if name == "edit_file" and "\0" in arguments["new_text"]:
            raise ValueError("new_text must be text without NUL characters")
        if name == "create_file" and "\0" in arguments["content"]:
            raise ValueError("content must be text without NUL characters")
        if name == "bash" and not arguments["command"].strip():
            raise ValueError("command must be nonempty")
        if name == "finish" and not arguments["summary"].strip():
            raise ValueError("summary must be nonempty")
        if name in {"read_file", "edit_file", "create_file"} and not arguments["path"].strip():
            raise ValueError("path must be nonempty")
        return name, arguments

    async def execute(self, action: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
        """Execute one normalized/OpenAI action within the caller's remaining budget."""
        try:
            name, arguments = self._action(action)
            if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)) or not math.isfinite(timeout_sec) or timeout_sec <= 0:
                return self._result("Tool deadline expired; no command was started.", 124)
            if name == "finish":
                return self._result("Finish requested; final review remains the harness's responsibility.", 0,
                                    submitted=True, summary=arguments["summary"], changed=False)
            timeout = min(float(timeout_sec), 300.0)
            if name == "bash":
                timeout = min(timeout, arguments.get("timeout_sec", self.command_timeout_sec))
                command = arguments["command"]
            else:
                payload = {"operation": name, "arguments": arguments, "workdir": self.workdir,
                           "max_file_bytes": self.max_file_bytes}
                encoded_source = base64.b64encode(_FILE_HELPER.encode()).decode("ascii")
                encoded_payload = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode("ascii")
                bootstrap = "import base64;exec(base64.b64decode(" + repr(encoded_source) + "))"
                command = "command -v python3 >/dev/null 2>&1 || { printf '%s\\n' 'File tool requires python3 in the task container; use bash tools available in this repository.'; exit 127; }; python3 -c " + shlex.quote(bootstrap) + " " + shlex.quote(encoded_payload)
            async with asyncio.timeout(timeout):
                result = await self.environment.exec(command=command, cwd=self.workdir, timeout_sec=timeout)
            stdout, stderr = result.stdout or "", result.stderr or ""
            code = int(result.return_code)
            if name != "bash":
                try:
                    parsed = json.loads(stdout)
                    if isinstance(parsed, dict) and isinstance(parsed.get("output"), str):
                        extra = {key: value for key, value in parsed.items() if key not in {"output", "returncode", "raw_output"}}
                        output = parsed["output"] + ("\nstderr:\n" + stderr if stderr else "")
                        return self._result(output, code, **extra)
                except (ValueError, TypeError):
                    pass
            output = f"exit_code: {code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            return self._result(output, code)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._result(f"Tool argument error: {exc}", 1, changed=False)
        except TimeoutError:
            return self._result("Tool command timed out; completion is unconfirmed. Inspect the current repository before retrying.", 124)
        except Exception as exc:
            return self._result(f"Tool execution error: {type(exc).__name__}: {exc}", 1)
