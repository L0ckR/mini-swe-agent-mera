"""Typed repository tools whose filesystem operations stay inside Harbor's task.

The host builds a command containing only base64-encoded source and JSON. The
helper resolves and operates on files in the task container, never on the host.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import posixpath
import re
import shlex
from typing import Any


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False},
    }}


TOOLS = [
    _tool("run_tests", "Run a focused existing test or reproduction command and record its exit code against the current patch. Choose the command from this repository's configuration; success is observed, not assumed.", {
        "command": {"type": "string"},
        "timeout_sec": {"type": "integer", "minimum": 1, "maximum": 300},
    }, ["command"]),
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
    _tool("search", "Search literal, case-sensitive text in repository UTF-8 files. Leading dashes such as -h and --help are ordinary text. Returns bounded filename:line matches with context; narrow path/file_glob if truncated.", {
        "pattern": {"type": "string", "description": "Nonempty literal text on one line; not a regex."},
        "path": {"type": "string", "description": "Repository file or directory, default ."},
        "file_glob": {"type": "string", "description": "Optional filename or repository-relative glob, e.g. *.py or src/*.py."},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Maximum matching lines, default 40."},
        "context_lines": {"type": "integer", "minimum": 0, "maximum": 3, "description": "Surrounding lines per match, default 1."},
    }, ["pattern"]),
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
import fnmatch
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


def search(root, path, args, limit):
    pattern, maximum, context = args["pattern"], args.get("max_results", 40), args.get("context_lines", 1)
    excluded = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    pending, files, scanned_paths, scanned_bytes = [path], 0, 0, 0
    matches, output, output_size, skipped, truncated = 0, [], 0, 0, False
    while pending and not truncated:
        target = pending.pop()
        if target.is_dir():
            entries = []
            with os.scandir(target) as iterator:
                for entry in iterator:
                    scanned_paths += 1
                    if scanned_paths > 4000:
                        truncated = True
                        break
                    if not entry.is_symlink() and entry.name not in excluded:
                        entries.append(Path(entry.path))
            pending.extend(sorted(entries, reverse=True))
            continue
        try:
            relative = str(target.relative_to(root))
            if target.is_symlink() or not target.is_file():
                continue
            glob = args.get("file_glob")
            if glob and not (fnmatch.fnmatchcase(relative, glob) or fnmatch.fnmatchcase(target.name, glob)):
                continue
            files += 1
            if files > 2000:
                truncated = True
                break
            size = target.stat().st_size
            if size > limit:
                skipped += 1
                continue
            if scanned_bytes + size > 8 * 1024 * 1024:
                truncated = True
                break
            with target.open("rb") as handle:
                raw = handle.read(limit + 1)
            scanned_bytes += len(raw)
            if len(raw) > limit or b"\0" in raw:
                skipped += 1
                continue
            lines = raw.decode("utf-8").splitlines()
        except (OSError, UnicodeError):
            skipped += 1
            continue
        last_shown = -1
        for index, line in enumerate(lines):
            if pattern not in line:
                continue
            if matches >= maximum:
                truncated = True
                break
            block = []
            for number in range(max(0, index - context, last_shown + 1), min(len(lines), index + context + 1)):
                text = lines[number]
                start = max(0, text.find(pattern) - 120) if pattern in text else 0
                snippet = ("..." if start else "") + text[start:start + 600] + ("..." if len(text) > start + 600 else "")
                marker = ":" if pattern in text else "-"
                block.append(relative + ":" + str(number + 1) + marker + " " + snippet)
            rendered = "\n".join(block)
            if output_size + len(rendered) > 24000:
                truncated = True
                break
            if rendered:
                output.append(rendered)
                output_size += len(rendered) + 1
            matches += 1
            last_shown = max(last_shown, min(len(lines) - 1, index + context))
    header = "Literal search: " + str(matches) + " matching lines; " + str(min(files, 2000)) + " files scanned."
    if skipped:
        header += " Skipped " + str(skipped) + " binary, non-UTF-8, unreadable or oversized files."
    if truncated:
        header += " Results truncated by the search limits; narrow path or file_glob."
    if not matches:
        header += " No matches in the scanned files."
    return {"output": header + ("\n" + "\n".join(output) if output else ""), "returncode": 0,
            "matches": matches, "scanned_files": min(files, 2000), "scanned_paths": min(scanned_paths, 4000),
            "scanned_bytes": scanned_bytes, "skipped_files": skipped, "truncated": truncated, "changed": False}


def run(payload):
    root = Path(payload["workdir"]).resolve(strict=True)
    args = payload["arguments"]
    requested = Path(args.get("path", "."))
    if ".." in requested.parts:
        raise ValueError("parent traversal is not allowed; use a path inside the repository")
    candidate = requested if requested.is_absolute() else root / requested
    path = candidate.resolve(strict=False)
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError("path resolves outside the task repository") from None
    operation = payload["operation"]
    limit = payload["max_file_bytes"]
    if operation == "search":
        if not path.exists():
            raise ValueError("search path does not exist")
        return search(root, path, args, limit)
    if path == root:
        raise ValueError("path must name a file, not the repository directory")
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


def _shell_without_heredoc_bodies(command: str) -> str:
    """Keep shell commands while ignoring literal source passed through heredocs."""
    visible, pending, quote = [], [], None
    for line in command.splitlines(keepends=True):
        if pending:
            delimiter, strip_tabs = pending[0]
            candidate = line.rstrip("\r\n")
            if (candidate.lstrip("\t") if strip_tabs else candidate) == delimiter:
                pending.pop(0)
            visible.append("\n")
            continue
        index, kept = 0, line
        while index < len(line):
            char = line[index]
            if char == "\\" and quote != "'":
                index += 2
                continue
            if quote:
                if char == quote:
                    quote = None
            elif char in {"'", '"'}:
                quote = char
            elif char == "#" and (index == 0 or line[index-1].isspace() or line[index-1] in ";&|()"):
                kept = line[:index] + "\n"
                break
            elif line.startswith("<<", index) and not line.startswith("<<<", index):
                match = re.match(r'''<<(-?)\s*(?:'([^']+)'|"([^"]+)"|([A-Za-z0-9_.-]+))''', line[index:])
                if match:
                    pending.append((next(value for value in match.groups()[1:] if value is not None), bool(match.group(1))))
                    index += len(match.group(0))
                    continue
            index += 1
        visible.append(kept)
    return "".join(visible)


def repository_command_error(command: str, workdir: str = "/testbed", _depth: int = 0) -> str | None:
    """Catch common accidental repo erasure; this is not a security boundary."""
    lexer = shlex.shlex(_shell_without_heredoc_bodies(command), posix=True, punctuation_chars=";&|()\n")
    lexer.whitespace, lexer.whitespace_split, lexer.commenters = " \t\r", True, ""
    try:
        tokens = list(lexer)
    except ValueError:
        return "Correct the shell quoting before running the command."
    segments, current = [], []
    for token in tokens + [";"]:
        if token and all(char in ";&|()\n" for char in token):
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    root, cwd = posixpath.normpath(workdir), posixpath.normpath(workdir)

    def protected(target):
        target = target.replace("${PWD}", cwd).replace("$PWD", cwd)
        path = posixpath.normpath(target if target.startswith("/") else posixpath.join(cwd, target))
        return (path in {root, "/"} or path == root + "/.git" or path.startswith(root + "/.git/")
                or (posixpath.dirname(path) == root and posixpath.basename(path) in {"*", "**", ".*"}))

    for args in segments:
        while args and (args[0] in {"{", "}"} or re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", args[0])):
            args = args[1:]
        while args and posixpath.basename(args[0]) in {"command", "builtin", "exec", "env", "sudo", "time", "nohup"}:
            wrapper, args = posixpath.basename(args[0]), args[1:]
            while args and (args[0].startswith("-") or re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", args[0])):
                flag, args = args[0], args[1:]
                if wrapper == "sudo" and flag in {"-u", "-g"} and args:
                    args = args[1:]
        if not args:
            continue
        executable = posixpath.basename(args[0])
        if executable == "cd" and len(args) == 2 and "$" not in args[1]:
            cwd = posixpath.normpath(args[1] if args[1].startswith("/") else posixpath.join(cwd, args[1]))
        elif executable in {"bash", "sh", "dash"} and _depth < 3:
            for index, arg in enumerate(args[1:], 1):
                if arg.startswith("-") and "c" in arg and index + 1 < len(args):
                    error = repository_command_error(args[index+1], cwd, _depth+1)
                    if error:
                        return error
                    break
        elif executable == "rm":
            if any(protected(arg) for arg in args[1:] if arg != "--" and not arg.startswith("-")):
                return "Do not erase the uploaded repository, all root files, or its .git metadata."
        elif executable == "git":
            rest = args[1:]
            while rest and rest[0].startswith("-"):
                flag, rest = rest[0], rest[1:]
                if flag in {"-c", "-C", "--git-dir", "--work-tree"} and rest:
                    value, rest = rest[0], rest[1:]
                    if flag == "-C":
                        cwd = posixpath.normpath(value if value.startswith("/") else posixpath.join(cwd, value))
            if not rest:
                continue
            action, flags = rest[0], rest[1:]
            if action == "reset" and any(flag == "--hard" or flag.startswith("--hard=") for flag in flags):
                return "git reset --hard can discard the existing task patch; inspect and edit the current files instead."
            short = "".join(flag[1:] for flag in flags if flag.startswith("-") and not flag.startswith("--"))
            if action == "clean" and ("f" in short or "--force" in flags) and ("d" in short or "--dirs" in flags) and "n" not in short and "--dry-run" not in flags:
                return "git clean -fd can remove task files; remove only explicitly identified temporary files."
            if action in {"checkout", "restore"} and any(protected(arg) for arg in flags if arg != "--" and not arg.startswith("-")):
                return "Do not restore or checkout the whole working tree; preserve edits and inspect individual files."
            if action == "clone" and flags and protected(flags[-1]):
                return "The repository already exists at the task root; inspect it instead of cloning over it."
    return None


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
        if name == "search" and (not arguments["pattern"] or len(arguments["pattern"]) > 1000
                                  or any(char in arguments["pattern"] for char in "\r\n\0")):
            raise ValueError("search pattern must be nonempty single-line literal text, at most 1000 characters")
        if name == "search" and len(arguments.get("file_glob", "")) > 256:
            raise ValueError("file_glob must be at most 256 characters")
        if name == "edit_file" and "\0" in arguments["new_text"]:
            raise ValueError("new_text must be text without NUL characters")
        if name == "create_file" and "\0" in arguments["content"]:
            raise ValueError("content must be text without NUL characters")
        if name in {"bash", "run_tests"} and not arguments["command"].strip():
            raise ValueError("command must be nonempty")
        if name == "finish" and not arguments["summary"].strip():
            raise ValueError("summary must be nonempty")
        if name in {"read_file", "edit_file", "create_file", "search"} and not arguments.get("path", ".").strip():
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
            if name in {"bash", "run_tests"}:
                timeout = min(timeout, arguments.get("timeout_sec", self.command_timeout_sec))
                command = arguments["command"]
                error = repository_command_error(command, self.workdir)
                if error:
                    return self._result("Repository-preservation guard: " + error + " The uploaded repository is already present. Continue by reading its files; no command was executed.", 1,
                                        changed=False, guard_blocked=True)
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
            if name not in {"bash", "run_tests"}:
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
