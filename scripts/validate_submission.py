#!/usr/bin/env python3
"""Validate the Stage II upload contract without loading an inference model.

Static validation cannot establish GPU compatibility or benchmark quality. The
organizer's inference test and a scored Harbor run are still required.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import stat
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_FILES = 1000
MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ckpt", ".onnx"}
AGENT_SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".jinja", ".j2", ".md", ".txt"}
AGENT_NOTICE_NAMES = {"license", "notice", "copying"}
BANNED_PARTS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".cache", "jobs", "tasks", "verifier",
    "solutions", "checkpoints", "hf_cache", "models",
}
SECRET_FILENAMES = {
    "id_rsa", "id_ed25519", "credentials.json", "service_account.json",
    "service-account.json", ".netrc", ".npmrc", ".pypirc",
}
SECRET_PATTERN = re.compile(
    rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
    rb"|\bhf_[A-Za-z0-9]{30,}\b"
    rb"|\bgh[pousr]_[A-Za-z0-9]{30,}\b"
    rb"|\bgithub_pat_[A-Za-z0-9_]{50,}\b"
    rb"|\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b"
)


class ValidationError(ValueError):
    """A submission violates the upload contract."""


@dataclass(frozen=True)
class SubmissionFile:
    name: str
    data: bytes
    executable: bool = False


@dataclass
class ValidationReport:
    agent: str
    model: str
    files: list[str]
    total_bytes: int
    model_lock: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    validation: str = "static_only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValidationError("submission.yaml keys must be strings")
        if key in mapping:
            raise ValidationError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _safe_name(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise ValidationError(f"unsafe archive path: {name!r}")
    parts = name.split("/")
    if name.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValidationError(f"unsafe archive path: {name!r}")
    if ":" in parts[0]:
        raise ValidationError(f"unsafe archive path: {name!r}")
    return PurePosixPath(name)


def _check_name(name: str) -> None:
    path = _safe_name(name)
    if name == "submission.yaml":
        return
    if path.parts[0] not in {"custom_agent", "inference"} or len(path.parts) < 2:
        raise ValidationError(f"unexpected upload path (archive must have no wrapper directory): {name}")
    if name == "inference/test.sh":
        raise ValidationError("inference/test.sh is trusted organizer code and must not be uploaded")
    for part in path.parts[1:]:
        lower = part.lower()
        if lower in BANNED_PARTS or lower.startswith(".env"):
            raise ValidationError(f"cache, task data, credentials, or local results must not be uploaded: {name}")
        if lower in SECRET_FILENAMES:
            raise ValidationError(f"credential file must not be uploaded: {name}")
    if path.suffix.lower() in WEIGHT_SUFFIXES | {".pyc", ".pyo", ".pem", ".key", ".p12", ".pfx", ".zip", ".tar", ".gz"}:
        raise ValidationError(f"weights, credentials, caches, or nested archives must not be uploaded: {name}")
    if path.parts[0] == "custom_agent":
        if path.suffix.lower() not in AGENT_SOURCE_SUFFIXES and path.name.lower() not in AGENT_NOTICE_NAMES:
            raise ValidationError(f"custom_agent may contain only source, template, provenance, or license files: {name}")


def collect_directory(source: Path) -> tuple[list[SubmissionFile], list[str]]:
    """Collect upload sources; omit only known generated files and trusted tests."""
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise ValidationError(f"submission directory is missing or is a symlink: {source}")
    files: list[SubmissionFile] = []
    warnings: list[str] = []
    total = 0
    for root, dirs, names in os.walk(source, followlinks=False):
        root_path = Path(root)
        for dirname in list(dirs):
            entry = root_path / dirname
            relative = entry.relative_to(source).as_posix()
            if entry.is_symlink():
                raise ValidationError(f"symlinks must not be uploaded: {relative}")
            if dirname == "__pycache__":
                dirs.remove(dirname)
                warnings.append(f"excluded generated directory: {relative}/")
            elif dirname.lower() in BANNED_PARTS:
                raise ValidationError(f"cache, task data, or local results directory in submission: {relative}")
            elif dirname.startswith(".env"):
                raise ValidationError(f"credential directory in submission: {relative}")
        for filename in sorted(names):
            entry = root_path / filename
            relative = entry.relative_to(source).as_posix()
            if entry.is_symlink() or not entry.is_file():
                raise ValidationError(f"only regular files may be uploaded: {relative}")
            if relative == "inference/test.sh" or entry.suffix in {".pyc", ".pyo"}:
                warnings.append(f"excluded trusted or generated file: {relative}")
                continue
            _check_name(relative)
            size = entry.stat().st_size
            if size > MAX_FILE_BYTES:
                raise ValidationError(f"file exceeds the source-only upload limit ({MAX_FILE_BYTES} bytes): {relative}")
            total += size
            if total > MAX_TOTAL_BYTES or len(files) >= MAX_FILES:
                raise ValidationError("submission exceeds the source-only upload size or file-count limit")
            files.append(SubmissionFile(relative, entry.read_bytes(), bool(entry.stat().st_mode & 0o111)))
    return sorted(files, key=lambda item: item.name), sorted(warnings)


def _read_zip(source: Path) -> list[SubmissionFile]:
    files = []
    seen = set()
    total = 0
    try:
        with zipfile.ZipFile(source) as archive:
            if len(archive.infolist()) > MAX_FILES:
                raise ValidationError("too many archive entries")
            for item in archive.infolist():
                name = item.filename
                if name in seen:
                    raise ValidationError(f"duplicate archive entry: {name}")
                seen.add(name)
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValidationError(f"symlinks must not be uploaded: {name}")
                if item.is_dir():
                    directory = _safe_name(name.rstrip("/"))
                    if directory.parts[0] not in {"inference", "custom_agent"}:
                        raise ValidationError(f"unexpected archive directory: {name}")
                    if any(part.lower() in BANNED_PARTS or part.lower().startswith(".env") for part in directory.parts):
                        raise ValidationError(f"excluded directory in archive: {name}")
                    continue
                _check_name(name)
                if item.flag_bits & 1:
                    raise ValidationError(f"encrypted archives are unsupported: {name}")
                total += item.file_size
                if item.file_size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                    raise ValidationError("archive exceeds the source-only upload size limit")
                files.append(SubmissionFile(name, archive.read(item), bool(mode & 0o111)))
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ValidationError(f"cannot read submission archive: {exc}") from exc
    return sorted(files, key=lambda item: item.name)


def validate_files(files: list[SubmissionFile], warnings: list[str] | None = None) -> ValidationReport:
    contents = {item.name: item.data for item in files}
    if len(contents) != len(files):
        raise ValidationError("duplicate submission file names")
    if "submission.yaml" not in contents:
        raise ValidationError("submission.yaml must be at the archive root")
    if "inference/Dockerfile" not in contents:
        raise ValidationError("inference/Dockerfile is required")
    for item in files:
        _check_name(item.name)
        if SECRET_PATTERN.search(item.data):
            raise ValidationError(f"possible embedded credential in {item.name}; remove it before packaging")
    try:
        config = yaml.load(contents["submission.yaml"].decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValidationError(f"cannot parse submission.yaml: {exc}") from exc
    if not isinstance(config, dict) or set(config) != {"agent", "model"}:
        raise ValidationError("submission.yaml must contain exactly the fields agent and model")
    if not isinstance(config["agent"], str) or config["agent"] not in {"custom_agent", "mini_swe_agent"}:
        raise ValidationError("agent must be custom_agent or mini_swe_agent")
    model = config["model"]
    if not isinstance(model, str) or MODEL_PATTERN.fullmatch(model) is None:
        raise ValidationError("model must be the exact served alias using the organizer's permitted characters")
    if config["agent"] == "custom_agent":
        for filename in ("custom_agent/__init__.py", "custom_agent/react_bash_agent.py"):
            if filename not in contents:
                raise ValidationError(f"required custom-agent file is missing: {filename}")
        source = contents["custom_agent/react_bash_agent.py"]
        try:
            tree = ast.parse(source, filename="custom_agent/react_bash_agent.py")
        except (SyntaxError, UnicodeError, ValueError) as exc:
            raise ValidationError(f"custom agent has invalid Python syntax: {exc}") from exc
        exports_class = any(isinstance(node, ast.ClassDef) and node.name == "ReactBashAgent" for node in tree.body)
        exports_import = any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.asname == "ReactBashAgent" or (alias.asname is None and alias.name == "ReactBashAgent") for alias in node.names)
            for node in tree.body
        )
        if not (exports_class or exports_import):
            raise ValidationError("custom_agent/react_bash_agent.py must export ReactBashAgent")
    elif any(name.startswith("custom_agent/") for name in contents):
        raise ValidationError("custom_agent files must be omitted when agent is mini_swe_agent")
    for name, data in contents.items():
        if name.startswith("custom_agent/") and PurePosixPath(name).suffix.lower() == ".py":
            try:
                ast.parse(data, filename=name)
            except (SyntaxError, UnicodeError, ValueError) as exc:
                raise ValidationError(f"invalid Python syntax in {name}: {exc}") from exc
    report_warnings = list(warnings or [])
    lock = None
    if "inference/model.json" in contents:
        try:
            lock = json.loads(contents["inference/model.json"])
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValidationError(f"invalid inference/model.json: {exc}") from exc
        if not isinstance(lock, dict):
            raise ValidationError("inference/model.json must contain an object")
        for key in ("model_id", "revision", "served_model_name", "vllm_version"):
            if not isinstance(lock.get(key), str) or not lock[key].strip():
                raise ValidationError(f"inference/model.json requires a nonempty string {key}")
        if re.fullmatch(r"[a-fA-F0-9]{40}", lock["revision"]) is None:
            raise ValidationError("model revision must be a pinned, full 40-character commit SHA")
        if lock["served_model_name"] != model:
            raise ValidationError("model alias differs between submission.yaml and inference/model.json")
    else:
        report_warnings.append("no inference/model.json: model revision and served alias were not cross-checked")
    return ValidationReport(
        agent=config["agent"], model=model, files=sorted(contents),
        total_bytes=sum(map(len, contents.values())), model_lock=lock,
        warnings=sorted(report_warnings),
    )


def validate_submission(source: Path | str) -> ValidationReport:
    source = Path(source)
    if source.is_dir():
        files, warnings = collect_directory(source)
        return validate_files(files, warnings)
    if source.is_symlink():
        raise ValidationError("submission archives must not be symlinks")
    return validate_files(_read_zip(source))


def validate_directory(source: Path | str) -> ValidationReport:
    """Validate a submission source directory, also used by the local runner."""
    files, warnings = collect_directory(Path(source))
    return validate_files(files, warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="submission directory or ZIP archive")
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    args = parser.parse_args(argv)
    try:
        report = validate_submission(args.source)
    except ValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"PASS (static contract): agent={report.agent} model={report.model}; {len(report.files)} files, {report.total_bytes} bytes")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
        print("GPU startup, tool calling, throughput, and pass@1 require a real evaluation run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
