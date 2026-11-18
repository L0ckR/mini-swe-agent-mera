#!/usr/bin/env python3
"""Create and update the requested GitHub fork using existing Git credentials.

Credentials are read into memory only, never written to artifacts or printed.
The public fork contains only the explicitly selected submission sources.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "SWE-agent/mini-swe-agent"
FORK_NAME = "mini-swe-agent-mera"
STATE = ROOT / "forks/mini-swe-agent/FORK.json"


def collect_publish_files(root: Path = ROOT) -> list[tuple[str, bytes, bool]]:
    """Return a sorted source/fixture inventory without authentication or network."""
    try:
        from .validate_submission import SECRET_PATTERN, collect_directory, validate_files
    except ImportError:
        from validate_submission import SECRET_PATTERN, collect_directory, validate_files

    root = Path(root).resolve()
    sources, warnings = collect_directory(root / "submission_stage2")
    validate_files(sources, warnings)
    selected = {"submission_stage2/" + item.name: (item.data, item.executable) for item in sources}

    def add(source: Path, destination: str | None = None):
        if source.is_symlink():
            raise RuntimeError(f"Refusing to publish symlink: {source.relative_to(root)}")
        if not source.is_file():
            return
        data = source.read_bytes()
        if len(data) > 2 * 1024 * 1024:
            raise RuntimeError(f"Unexpectedly large publication source: {source.relative_to(root)}")
        if SECRET_PATTERN.search(data):
            raise RuntimeError(f"Possible credential in publication source: {source.relative_to(root)}")
        selected[destination or source.relative_to(root).as_posix()] = (data, bool(source.stat().st_mode & 0o111))

    for path, destination in [("README.md", "MERA_README.md"), ("docs/harness-fork.md", "MERA_FORK.md"),
                              ("requirements.txt", "mera_requirements.txt")]:
        add(root / path, destination)
    for folder, suffixes in (("scripts", {".py", ".sh"}), ("vendor/stage2", {".py", ".sh", ".json", ".txt"}),
                             ("vendor/mini_swe_agent", {".py", ".json", ".md", ".patch", ".txt"}),
                             ("vendor/pi", {".json", ".md", ".txt", ""}), ("docs", {".md"})):
        for source in sorted((root / folder).rglob("*")):
            if "__pycache__" not in source.parts and source.suffix in suffixes:
                add(source)
    for source in sorted((root / "tests").rglob("*")):
        if "__pycache__" not in source.parts and source.suffix in {".py", ".json", ".jinja", ".j2"} and source.name != "test_agent.py":
            add(source, "tests_mera/" + source.relative_to(root / "tests").as_posix())
    for name in ("cases.json", "public_statement_wrapper.json"):
        add(root / "experiments/diagnostics" / name)
    previous = root / 'experiments/mini_before_pi/submission_stage2'
    if previous.is_dir():
        previous_files, previous_warnings = collect_directory(previous)
        validate_files(previous_files, previous_warnings)
        for item in previous_files:
            selected['experiments/mini_before_pi/submission_stage2/' + item.name] = (item.data, item.executable)
        add(previous.parent / 'source_sha256.json')
    # Publish only provenance metadata. External statements stay local and can
    # be fetched separately using their pinned source URLs and SHA256 hashes.
    inventory = root / "data/public_task_inventory.json"
    if inventory.is_file():
        report = json.loads(inventory.read_text())
        rows = report.get("tasks", [])
        if len(rows) != report.get("n_public_tasks"):
            raise RuntimeError("Public statement inventory count mismatch")
        for row in rows:
            source = root / row["instruction_path"]
            try:
                source.resolve().relative_to((root / "data/public_statements").resolve())
            except ValueError:
                raise RuntimeError("Public statement inventory points outside its allowed directory") from None
        add(inventory)
    for source in sorted((root / "experiments/results").glob("*.json")):
        if source.stat().st_size > 256 * 1024:
            raise RuntimeError(f"Diagnostic report is not concise: {source.name}")
        add(source)
    return [(name, data, executable) for name, (data, executable) in sorted(selected.items())]


def prepare_diagnostic_evidence(root: Path = ROOT) -> list[Path]:
    """Copy only whitelisted diagnostic facts from our local jobs into reports."""
    root = Path(root).resolve()
    output = root / "experiments/results"
    output.mkdir(parents=True, exist_ok=True)
    written = []

    def compact_acceptance(value):
        result = {key: value[key] for key in ("returncode", "passed", "total", "passed_count") if key in value}
        result["checks"] = [{key: item[key] for key in ("criterion", "passed") if key in item} for item in value.get("checks", [])]
        if value.get("error"):
            result["check_error"] = True
        return result

    def compact_backend(value):
        fields = {"diagnostic_only", "engine", "engine_version", "model", "served_alias", "context_tokens",
                  "max_output_tokens", "thinking", "thinking_token_budget_enforced", "production_equivalent",
                  "ignored_fields_seen", "unsupported_controls", "metadata_unavailable"}
        return {key: value[key] for key in fields if key in value}

    for source in sorted((root / "jobs").glob("diagnostics-live-*/summary.json")):
        report = json.loads(source.read_text())
        if report.get("kind") != "NON_HARBOR_SYNTHETIC_DIAGNOSTIC" or not report.get("by_variant"):
            continue  # An incomplete/foreign job is not promoted to published evidence.
        saved = {key: report[key] for key in ("kind", "model", "settings", "baseline_archive_sha256", "statement_format", "notes", "by_variant", "source_snapshots") if key in report}
        saved["backend_metadata"] = compact_backend(report.get("backend_metadata", {}))
        saved.update(raw_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), raw_jobs_published=False,
                     official_verifier="NOT_RUN", pass_at_1=None)
        saved["trials"] = []
        for trial in report.get("trials", []):
            compact = {key: trial[key] for key in ("variant", "case", "resolved", "elapsed_sec", "visible_tests_modified", "patch_nonempty") if key in trial}
            compact["agent_failed"] = bool(trial.get("agent_error"))
            compact["acceptance"] = compact_acceptance(trial.get("acceptance", {}))
            saved["trials"].append(compact)
        audit = source.with_name("post-audit.json")
        if audit.is_file():
            raw = json.loads(audit.read_text())
            saved["post_audit"] = {"raw_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
                                  "trials": [{key: trial[key] for key in ("variant", "case", "audited_test_changes", "note") if key in trial} for trial in raw.get("trials", [])]}
        target = output / (source.parent.name + ".json")
        target.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
        written.append(target)
    for source in sorted((root / "jobs").glob("public-click-live-*/result.json")):
        report = json.loads(source.read_text())
        if report.get("kind") != "NON_HARBOR_PUBLIC_CLICK_DIAGNOSTIC" or "resolved_public_diagnostic" not in report:
            continue
        saved = {key: report[key] for key in ("kind", "official_verifier", "pass_at_1", "public_statement", "base_archive", "base_local_head", "source_hashes", "attempts", "settings", "notes", "elapsed_sec", "resolved_public_diagnostic", "head_changed", "modified_original_test_files", "original_test_mode_changes", "added_test_files", "test_symlinks", "test_paths_changed_in_git", "working_patch_nonempty") if key in report}
        saved.update(raw_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), raw_jobs_published=False,
                     local_inference=compact_backend(report.get("local_inference", {})), agent_failed=bool(report.get("agent_error")))
        for phase in ("before", "after"):
            saved[phase] = {key: compact_acceptance(report.get(phase, {}).get(key, {})) for key in ("public_acceptance", "visible_tests")}
        delivered = report.get("delivered_patch", {})
        saved["delivered_patch"] = {key: delivered[key] for key in ("valid", "nonempty", "bytes", "sha256", "applies", "test_paths", "binary") if key in delivered}
        for key in ("public_acceptance", "visible_tests"):
            saved["delivered_patch"][key] = compact_acceptance(delivered.get(key, {}))
        target = output / (source.parent.name + ".json")
        target.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
        written.append(target)
    return written


class GitHub:
    def __init__(self):
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not token:
            env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false")
            result = subprocess.run(
                ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                capture_output=True, text=True, env=env, timeout=20,
            )
            credentials = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            token = credentials.get("password")
        if not token:
            raise RuntimeError("No existing GitHub API credential is available to create the remote fork")
        self._token = token

    def request(self, method: str, path: str, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            "https://api.github.com" + path, data=body, method=method,
            headers={"Authorization": f"Bearer {self._token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "swe-mera-harness-fork", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not print headers, credentials, or arbitrary response bodies.
            if exc.code == 404 and method == "GET":
                return None
            raise RuntimeError(f"GitHub API {method} {path}: HTTP {exc.code}") from None


def create(api: GitHub) -> dict:
    user = api.request("GET", "/user")
    owner = user["login"]
    name = f"{owner}/{FORK_NAME}"
    existing = api.request("GET", f"/repos/{name}")
    if existing:
        if not existing.get("fork") or existing.get("parent", {}).get("full_name", "").lower() != UPSTREAM.lower():
            raise RuntimeError(f"Existing {name} is not a fork of the intended upstream")
        fork = existing
    else:
        fork = api.request("POST", f"/repos/{UPSTREAM}/forks", {"name": FORK_NAME, "default_branch_only": True})
    state = {"upstream": UPSTREAM, "repository": fork["full_name"], "url": fork["html_url"],
             "default_branch": fork["default_branch"], "source_commit": "04d809ceab9df28f9adaed044884180159172930"}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))
    return state


def publish(api: GitHub, branch: str) -> dict:
    state = json.loads(STATE.read_text())
    repo = state["repository"]
    info = api.request("GET", f"/repos/{repo}")
    if not info or not info.get("fork") or info.get("parent", {}).get("full_name", "").lower() != UPSTREAM.lower():
        raise RuntimeError("Remote target is not the expected upstream fork")
    # Source/fixture selection is deterministic and independently inspectable.
    selected = collect_publish_files(ROOT)
    current = api.request("GET", f"/repos/{repo}/git/ref/heads/{branch}")
    parent_branch = state.get('adaptation_branch', state['default_branch'])
    ref = current or api.request("GET", f"/repos/{repo}/git/ref/heads/{parent_branch}")
    parent = ref["object"]["sha"]
    base = api.request("GET", f"/repos/{repo}/git/commits/{parent}")["tree"]["sha"]
    manifest = {"format": "swe-mera-source-inventory-1", "contains_raw_jobs": False,
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data, _ in selected}}
    selected.append(("MERA_SOURCE_MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode(), False))
    tree = []
    for name, data, executable in selected:
        blob = api.request("POST", f"/repos/{repo}/git/blobs", {"content": base64.b64encode(data).decode(), "encoding": "base64"})
        tree.append({"path": name, "mode": "100755" if executable else "100644", "type": "blob", "sha": blob["sha"]})
    result_tree = api.request("POST", f"/repos/{repo}/git/trees", {"base_tree": base, "tree": tree})
    commit = api.request("POST", f"/repos/{repo}/git/commits", {
        "message": "Adapt native Pi and ready Qwen RL adapter to SWE MERA Stage II; retain mini baseline",
        "tree": result_tree["sha"], "parents": [parent],
    })
    if current:
        api.request("PATCH", f"/repos/{repo}/git/refs/heads/{branch}", {"sha": commit["sha"], "force": False})
    else:
        api.request("POST", f"/repos/{repo}/git/refs", {"ref": f"refs/heads/{branch}", "sha": commit["sha"]})
    state.update({"adaptation_branch": branch, "adaptation_commit": commit["sha"],
                  "adaptation_url": f"https://github.com/{repo}/tree/{branch}/submission_stage2"})
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "publish", "inventory", "evidence"])
    parser.add_argument("--branch", default="feature/swe-mera-stage2")
    args = parser.parse_args()
    try:
        if args.action == "inventory":
            files = collect_publish_files(ROOT)
            print(json.dumps({"files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data, _ in files}}, indent=2))
            return
        if args.action == "evidence":
            print(json.dumps([str(path.relative_to(ROOT)) for path in prepare_diagnostic_evidence(ROOT)], indent=2))
            return
        api = GitHub()
        create(api) if args.action == "create" else publish(api, args.branch)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
