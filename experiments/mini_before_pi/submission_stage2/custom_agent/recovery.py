"""Bounded, observational advice for unsuccessful repository-tool loops.

This module never executes tools, changes their results, or refuses an action.
In particular, historical edit failures are not evidence of current file bytes:
an intervening shell command may have changed the repository.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import posixpath
import re


@dataclass(frozen=True)
class _FailedEdit:
    path: str
    target: str


class LoopRecovery:
    """Call ``observe(action, result)`` once after each completed tool call.

    The returned string is empty unless observed results justify recovery
    advice. Append nonempty advice to the tool observation shown to the model.
    Interleaved reads do not erase failures; successful file writes clear that
    path. All retained state is bounded by ``max_history`` tool calls.
    """

    def __init__(self, *, max_history: int = 128, workdir: str = "/testbed"):
        if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 2:
            raise ValueError("max_history must be an integer of at least 2")
        self._history: deque[_FailedEdit | None] = deque(maxlen=max_history)
        self._workdir = posixpath.normpath(workdir)

    def _path(self, path: str) -> str:
        normalized = posixpath.normpath(path)
        if normalized.startswith(self._workdir + "/"):
            normalized = normalized[len(self._workdir) + 1:]
        return normalized

    def _clear_path(self, path: str) -> None:
        self._history = deque(
            (None if item is not None and item.path == path else item for item in self._history),
            maxlen=self._history.maxlen,
        )

    @staticmethod
    def _action(action: Mapping) -> tuple[str, Mapping]:
        function = action.get("function", action)
        if not isinstance(function, Mapping):
            return "", {}
        name, arguments = function.get("name", ""), function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, RecursionError):
                arguments = {}
        return name if isinstance(name, str) else "", arguments if isinstance(arguments, Mapping) else {}

    def observe(self, action: Mapping, result: Mapping) -> str:
        """Return advisory feedback based on completed, unsuccessful actions."""
        self._history.append(None)
        if not isinstance(action, Mapping) or not isinstance(result, Mapping):
            return ""
        name, arguments = self._action(action)
        path = arguments.get("path")
        succeeded = result.get("returncode") == 0
        if name in {"edit_file", "create_file"} and isinstance(path, str) and succeeded:
            self._clear_path(self._path(path))
            return ""

        output = result.get("raw_output") or result.get("output") or ""
        if not isinstance(output, str):
            return ""
        # A successful --help invocation is not a successful source search.
        if name == "bash" and isinstance(arguments.get("command"), str):
            command = arguments["command"]
            if (re.search(r"(?:^|[\s;/|&(])(?:[^\s;|&()]*/)?grep(?:\s|$)", command)
                    and re.search(r"(?im)^\s*Usage:\s+(?:\S*/)?grep\b", output)):
                return (
                    "Search observation: this command printed grep usage, which does not establish "
                    "whether repository text matches. For a literal pattern, especially one beginning "
                    "with '-', use grep -nF -- '<pattern>' <paths> or rg --fixed-strings -- '<pattern>' <paths>."
                )

        old_text = arguments.get("old_text")
        if (name != "edit_file" or not isinstance(path, str) or not path
                or not isinstance(old_text, str) or not old_text or succeeded
                or result.get("changed") is True):
            return ""
        lower = output.lower()
        missing = "old_text" in lower and ("not found" in lower or "does not occur" in lower)
        ambiguous = "old_text" in lower and ("more than once" in lower or "ambiguous" in lower)
        if not (missing or ambiguous):
            return ""
        path = self._path(path)
        target = hashlib.sha256(old_text.encode("utf-8", errors="surrogatepass")).hexdigest()
        self._history[-1] = _FailedEdit(path, target)
        path_count = sum(item is not None and item.path == path for item in self._history)
        same_count = sum(item is not None and item.path == path and item.target == target for item in self._history)
        if same_count < 2 and path_count < 3:
            return ""

        snippet = json.dumps(" ".join(old_text.split())[:100], ensure_ascii=False)
        displayed_path = json.dumps(path[:200], ensure_ascii=False)
        advice = (
            f"Edit-loop recovery: recent tool history contains {path_count} failed exact-text edits "
            f"for {displayed_path}; {same_count} used this same old_text: {snippet}. "
            "These are observed past failures, not a claim that the file is still unchanged. "
        )
        if same_count >= 3 or path_count >= 3:
            advice += (
                "Break the repeated read/edit loop and reconsider the target file and root-cause hypothesis. "
                "An issue's illustrative caller snippet may not exist in the repository; locate the "
                "responsible implementation by searching its symbols and tracing callers. "
            )
        else:
            advice += "Do not retry this replacement merely after another identical read. "
        if ambiguous:
            advice += "Use exact surrounding context from the current file to identify one unique occurrence."
        else:
            advice += (
                "Choose old_text directly from current source, preserving whitespace, or search for the "
                "actual implementation before choosing a new edit. For literal search use grep -nF -- "
                "'<pattern>' <paths> or rg --fixed-strings -- '<pattern>' <paths>."
            )
        return advice
