"""Async core fork of mini-swe-agent 2.4.6, without global configuration I/O.

Upstream: SWE-agent/mini-swe-agent at 04d809ceab9df28f9adaed044884180159172930.
The original MIT license and exact source provenance accompany this package.
"""

from pathlib import Path
from typing import Any, Protocol

__version__ = "2.4.6"


class Model(Protocol):
    """The native-tool model adapter consumed by the async agent core."""

    config: Any

    async def query(self, messages: list[dict], **kwargs) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Environment(Protocol):
    """Tool execution is delegated to Harbor's existing task environment."""

    config: Any

    async def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Agent(Protocol):
    config: Any

    async def run(self, task: str, **kwargs) -> dict: ...

    def save(self, path: Path | None, *extra_dicts) -> dict: ...


__all__ = ["Agent", "Environment", "Model", "__version__"]
