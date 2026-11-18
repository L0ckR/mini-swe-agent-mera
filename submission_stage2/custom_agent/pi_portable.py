"""Python-free task-container operations; Python here runs only on Harbor's host.

All repository access still goes through environment.exec. Container helpers
use Bash, coreutils and Git, which the Go/PHP task images provide.
"""
from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import difflib
import hashlib
import os
from pathlib import PurePosixPath
import re
import shlex
import time


_PYTHON = OrderedDict()


async def has_python(environment, workdir, timeout_sec=5):
    """Probe once per environment/workdir, including concurrent callers."""
    key = (id(environment), workdir)
    existing = _PYTHON.get(key)
    if existing is not None and existing[0] is environment:
        return await asyncio.shield(existing[1])

    async def detect():
        async with asyncio.timeout(timeout_sec):
            result = await environment.exec(command="command -v python3 >/dev/null 2>&1", cwd=workdir,
                                            timeout_sec=max(0.001, min(timeout_sec, 5)))
        return result.return_code == 0

    task = asyncio.create_task(detect())
    _PYTHON[key] = (environment, task)
    while len(_PYTHON) > 256:
        _PYTHON.popitem(last=False)
    try:
        return await asyncio.shield(task)
    except BaseException:
        if _PYTHON.get(key, (None, None))[1] is task:
            _PYTHON.pop(key, None)
        raise


def _b64(value):
    return base64.b64encode(value).decode("ascii")


def _decode(value):
    return base64.b64decode(value, validate=True)


def _path(path, workdir):
    if not isinstance(path, str) or not path or "\0" in path:
        raise ValueError("path must be nonempty text")
    value = PurePosixPath(path)
    root = PurePosixPath(workdir)
    if ".." in value.parts:
        raise ValueError("parent traversal is not allowed")
    value = value if value.is_absolute() else root / value
    value.relative_to(root)
    return str(value)


def _preamble(workdir):
    return "set -euo pipefail\nmera_root=$(readlink -f -- " + shlex.quote(workdir) + ")\n" + r'''
mera_resolve() {
    mera_probe=$1
    mera_suffix=
    while [ ! -e "$mera_probe" ] && [ ! -L "$mera_probe" ]; do
        [ "$mera_probe" != / ] || return 1
        mera_suffix="/${mera_probe##*/}$mera_suffix"
        mera_probe=${mera_probe%/*}
        [ -n "$mera_probe" ] || mera_probe=/
    done
    mera_resolved="$(readlink -f -- "$mera_probe")$mera_suffix"
    case "$mera_resolved" in "$mera_root"|"$mera_root"/*) ;; *) echo 'path resolves outside task repository' >&2; return 1;; esac
}
'''


async def _run(environment, workdir, script, timeout_sec):
    if timeout_sec <= 0:
        raise TimeoutError("portable operation deadline expired")
    async with asyncio.timeout(timeout_sec):
        result = await environment.exec(command="bash -c " + shlex.quote(_preamble(workdir) + script),
                                        cwd=workdir, timeout_sec=timeout_sec)
    if result.return_code:
        raise RuntimeError((result.stderr or result.stdout or f"command exited {result.return_code}")[-4000:])
    return result.stdout or ""


async def _cleanup(environment, workdir, path):
    try:
        await _run(environment, workdir, "rm -f -- " + shlex.quote(path), 5)
    except Exception:
        pass


async def portable_file_operation(environment, workdir, operation, args, timeout_sec=20):
    requested = _path(args.get("path"), workdir)
    resolve = "mera_resolve " + shlex.quote(requested) + "\n"
    deadline = time.monotonic() + timeout_sec
    remaining = lambda: deadline - time.monotonic()
    if operation == "read_file":
        script = resolve + r'''
[ -f "$mera_resolved" ] || { echo 'not a regular file' >&2; exit 1; }
[ "$(stat -c %s -- "$mera_resolved")" -le 4194304 ] || { echo 'file exceeds 4MiB' >&2; exit 1; }
base64 -w0 -- "$mera_resolved"
'''
        return {"data_base64": (await _run(environment, workdir, script, remaining())).strip()}
    if operation == "access":
        flag = "-w" if args.get("write") else "-r"
        await _run(environment, workdir, resolve + f"[ -e \"$mera_resolved\" ] && [ {flag} \"$mera_resolved\" ]", remaining())
        return {}
    if operation == "mkdir":
        await _run(environment, workdir, resolve + 'mkdir -p -- "$mera_resolved"', remaining())
        return {}
    if operation != "write_file":
        raise ValueError("unsupported portable file operation")
    encoded = args.get("data_base64")
    if not isinstance(encoded, str) or len(encoded) > 6 * 1024 * 1024:
        raise ValueError("write requires bounded base64 file bytes")
    if len(_decode(encoded)) > 4 * 1024 * 1024:
        raise ValueError("write exceeds 4MiB")
    meta = await _run(environment, workdir, resolve + r'''
[ "$mera_resolved" != "$mera_root" ] || { echo 'path must name a file' >&2; exit 1; }
mera_mode=644
mera_existed=0
if [ -e "$mera_resolved" ]; then
    [ -f "$mera_resolved" ] || { echo 'not a regular file' >&2; exit 1; }
    mera_mode=$(stat -c %a -- "$mera_resolved")
    mera_existed=1
fi
mera_temporary=$(mktemp -- "${mera_resolved%/*}/.mera-pi-write-XXXXXX")
printf '%s\n' "$mera_mode" "$mera_existed"
printf '%s' "$mera_temporary" | base64 -w0
''', remaining())
    mode, existed, temporary_b64 = meta.strip().split("\n", 2)
    temporary = _decode(temporary_b64).decode()
    if not re.fullmatch(r"[0-7]{3,4}", mode):
        raise ValueError("invalid file mode from portable helper")
    try:
        for offset in range(0, len(encoded), 24000):
            await _run(environment, workdir,
                       "printf '%s' " + shlex.quote(encoded[offset:offset+24000]) +
                       " | base64 --decode >> " + shlex.quote(temporary), remaining())
        script = resolve + "chmod " + mode + " -- " + shlex.quote(temporary) + "\n"
        script += "mv -f -- " + shlex.quote(temporary) + ' "$mera_resolved"\n'
        if existed == "0":
            script += r'''git add -N -- "${mera_resolved#"$mera_root"/}" || printf '%s\n' 'New file written, but git add -N failed.'
'''
        output = await _run(environment, workdir, script, remaining())
        return {"warning": output.strip()} if output.strip() else {}
    finally:
        await _cleanup(environment, workdir, temporary)


def is_test_path(name):
    path = PurePosixPath(name)
    if any(part.lower() in {"test", "tests", "testing", "__tests__", "spec", "specs"} for part in path.parts[:-1]):
        return True
    return bool(re.search(r"(^test_.*\.py$|^test.*\.php$|_test\.go$|(?:Test|Spec)\.php$|\.(?:test|spec)\.[cm]?[jt]sx?$|^conftest\.py$)", path.name, re.I))


async def portable_contract(environment, contract, timeout_sec=20, validate_only=False):
    deadline = time.monotonic() + timeout_sec
    remaining = lambda: deadline - time.monotonic()
    workdir, target = contract.workdir, contract.patch_path
    relative = None
    if target:
        target = _path(target, workdir)
        relative = str(PurePosixPath(target).relative_to(workdir))
        if not relative or relative == "." or PurePosixPath(relative).parts[0] == ".git":
            raise ValueError("patch output must be an ordinary repository file")
    exclusions = [":(exclude,glob).mera-export-*/**", ":(exclude,glob).mera-snapshot-*/**",
                  ":(exclude,glob).mera-pi-write-*", ":(exclude,glob).mera-pi-transfer-*",
                  ":(exclude,glob)**/junit.xml",
                  ":(exclude,glob)**/task_tests.json", ":(exclude,glob)**/*.orig", ":(exclude,glob)**/*.rej"]
    if relative:
        exclusions.append(":(exclude)" + relative)
    script = r'''
mera_directory=$(mktemp -d -- "$mera_root/.mera-export-XXXXXX")
trap 'rm -rf -- "$mera_directory"' EXIT
GIT_INDEX_FILE="$mera_directory/index" git read-tree HEAD
GIT_INDEX_FILE="$mera_directory/index" git add -A -- . ''' + " ".join(map(shlex.quote, exclusions)) + r'''
GIT_INDEX_FILE="$mera_directory/index" git diff --cached --no-renames --name-only -z HEAD > "$mera_directory/names"
GIT_INDEX_FILE="$mera_directory/index" git diff --cached --no-renames --numstat -z HEAD > "$mera_directory/numstat"
GIT_INDEX_FILE="$mera_directory/index" git diff --cached --no-renames --full-index --no-color HEAD > "$mera_directory/change.patch"
printf '%s\n' "${mera_directory##*/}"
base64 -w0 -- "$mera_directory/names"; printf '\n'
base64 -w0 -- "$mera_directory/numstat"; printf '\n'
wc -c < "$mera_directory/change.patch"
trap - EXIT
'''
    directory = None
    try:
        text = await _run(environment, workdir, script, remaining())
        directory, encoded_names, encoded_numstat, size = text.split("\n", 3)
        if not re.fullmatch(r"\.mera-export-[A-Za-z0-9]+", directory):
            raise ValueError("invalid temporary export directory")
        names = [os.fsdecode(name) for name in _decode(encoded_names).split(b"\0") if name]
        rejected = [name for name in names if contract.non_test_only and is_test_path(name)]
        result = {"valid": False, "written": False, "path": target, "returncode": 1}
        if rejected:
            return {**result, "output": "Test changes are forbidden; inspect and repair without reverting pre-existing work.",
                    "rejected_test_paths": rejected}
        binary = [item for item in _decode(encoded_numstat).split(b"\0") if item.startswith(b"-\t-\t")]
        if binary and target:
            return {**result, "output": "Patch not exported: binary changes are forbidden.",
                    "binary_entries": [os.fsdecode(item) for item in binary]}
        if validate_only:
            return {"valid": True, "written": False, "path": target, "returncode": 0,
                    "output": "Current changes satisfy the detected file constraints.", "changed_paths": names}
        if not int(size.strip()):
            return {**result, "output": "Patch not exported: no implementation changes were found."}
        output = str(PurePosixPath(workdir) / directory)
        script = "mera_directory=" + shlex.quote(output) + "\n" + r'''
GIT_INDEX_FILE="$mera_directory/validation-index" git read-tree HEAD
GIT_INDEX_FILE="$mera_directory/validation-index" git apply --cached --check "$mera_directory/change.patch"
'''
        script += '[ ! -L ' + shlex.quote(target) + " ] || { echo 'patch output must not be a symlink' >&2; exit 1; }\n"
        script += "mera_resolve " + shlex.quote(target) + '\nmkdir -p -- "${mera_resolved%/*}"\n'
        script += 'mv -f -- "$mera_directory/change.patch" "$mera_resolved"\n'
        await _run(environment, workdir, script, remaining())
        return {"valid": True, "written": True, "path": target, "returncode": 0,
                "output": "Exported unified diff and verified it applies to the original HEAD index.",
                "bytes": int(size.strip()), "changed_paths": names, "apply_check_passed": True}
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return {"valid": False, "written": False, "path": target, "returncode": 124,
                "output": "Patch validation/export timed out; success is unconfirmed."}
    except Exception as exc:
        return {"valid": False, "written": False, "path": target, "returncode": 1,
                "output": f"Patch validation/export failed: {type(exc).__name__}: {exc}"}
    finally:
        if directory and re.fullmatch(r"\.mera-export-[A-Za-z0-9]+", directory):
            try:
                await _run(environment, workdir, "rm -rf -- " + shlex.quote(str(PurePosixPath(workdir) / directory)), 5)
            except Exception:
                pass


def _bounded(value, limit=24000):
    if len(value) <= limit:
        return value
    head = limit * 2 // 3
    return value[:head] + f"\n... [{len(value)-limit} characters omitted; inspect targeted files] ...\n" + value[-(limit-head):]


async def portable_snapshot(environment, workdir, timeout_sec=20, include_diff=True, ignored_paths=()):
    """Bounded observed patch/fingerprint without a task-container interpreter."""
    deadline = time.monotonic() + timeout_sec
    remaining = lambda: deadline - time.monotonic()
    exclusions = [":(exclude,glob).mera-snapshot-*/**", *[":(exclude)" + name for name in ignored_paths]]
    quoted = " ".join(shlex.quote(name) for name in exclusions)
    script = r'''
mera_dir=$(mktemp -d "$mera_root/.mera-snapshot-XXXXXX")
trap 'rm -rf -- "$mera_dir"' EXIT
'''
    script += "git --no-optional-locks diff --no-ext-diff --no-textconv --binary HEAD -- . " + quoted + ' > "$mera_dir/patch"\n'
    script += 'wc -c < "$mera_dir/patch"\n'
    script += 'mera_hash=$(sha256sum "$mera_dir/patch"); printf "%s\\n" "${mera_hash%% *}"\n'
    if include_diff:
        script += r'''if [ "$(wc -c < "$mera_dir/patch")" -gt 24000 ]; then
  { head -c 16000 "$mera_dir/patch"; printf '\n... [patch bytes omitted] ...\n'; tail -c 8000 "$mera_dir/patch"; } | base64 -w0
else base64 -w0 "$mera_dir/patch"; fi
'''
    script += "printf '\\n'\n"
    script += "git --no-optional-locks diff --no-ext-diff --no-textconv --stat HEAD -- . " + quoted + ' > "$mera_dir/stat"\n'
    script += 'head -c 8000 "$mera_dir/stat" | base64 -w0\nprintf \'\\n\'\n'
    script += "git --no-optional-locks status --short --untracked-files=all -- . " + quoted + ' > "$mera_dir/status"\n'
    script += 'head -c 8000 "$mera_dir/status" | base64 -w0\nprintf \'\\n\'\n'
    script += "git --no-optional-locks ls-files --others --exclude-standard -z -- . " + quoted + " | base64 -w0\nprintf '\\n'\n"
    try:
        output = await _run(environment, workdir, script, remaining())
        lines = output.splitlines()
        if len(lines) != 6:
            raise ValueError("portable snapshot returned malformed fields")
        size = int(lines[0].strip())
        if len(lines[1]) != 64:
            raise ValueError("portable snapshot returned invalid patch digest")
        rendered = base64.b64decode(lines[2]).decode("utf-8", "replace")
        stats = base64.b64decode(lines[3]).decode("utf-8", "replace")
        status = base64.b64decode(lines[4]).decode("utf-8", "replace")
        names = [name for name in base64.b64decode(lines[5]).split(b"\0") if name]
        digest = hashlib.sha256(b"mera-portable-evidence-v1\0" + lines[0].encode() + b"\0" + lines[1].encode())
        volatile = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "node_modules", "coverage", "logs"}
        complete, skipped, count, total = True, [], 0, 0
        for raw_name in sorted(names):
            name = raw_name.decode("utf-8", "surrogateescape")
            path = PurePosixPath(name)
            if name in ignored_paths or any(part in volatile for part in path.parts) or path.suffix in {".pyc", ".pyo", ".log"}:
                continue
            if count >= 128:
                complete = False
                skipped.append({"path": name, "reason": "untracked content exceeds bounded snapshot limit"})
                continue
            candidate = str(PurePosixPath(workdir) / path)
            command = "mera_candidate=" + shlex.quote(candidate) + "\n" + r'''
if [ -L "$mera_candidate" ]; then
  printf 'L\n0\n0\n'; readlink -- "$mera_candidate" | base64 -w0
else
  mera_resolve "$mera_candidate"
  [ -f "$mera_resolved" ] || { echo 'not a regular file' >&2; exit 1; }
  mera_size=$(stat -c %s "$mera_resolved")
  [ "$mera_size" -le 4194304 ] || { echo 'untracked file exceeds 4MiB' >&2; exit 1; }
  printf 'F\n'; stat -c %a "$mera_resolved"; printf '%s\n' "$mera_size"
  base64 -w0 "$mera_resolved"
fi
printf '\n'
'''
            try:
                detail = (await _run(environment, workdir, command, remaining())).splitlines()
                if len(detail) != 4:
                    raise ValueError("untracked file inspection returned malformed fields")
                data = base64.b64decode(detail[3])
                if detail[0] == "L":
                    digest.update(b"\0symlink\0" + raw_name + b"\0" + data)
                    skipped.append({"path": name, "reason": "symlink target was not read"})
                    count += 1
                    continue
                mode = int(detail[1], 8)
                if total + len(data) > 4 * 1024 * 1024:
                    raise ValueError("untracked content exceeds bounded snapshot limit")
                total += len(data)
                count += 1
                digest.update(b"\0file\0" + raw_name + b"\0" + str(mode).encode() + b"\0" + data)
                if include_diff:
                    if b"\0" in data:
                        rendered += f"\nNew untracked binary file {name!r} ({len(data)} bytes)\n"
                    else:
                        rendered += "\nNew untracked file " + repr(name) + "\n" + "".join(difflib.unified_diff(
                            [], data.decode("utf-8", "replace").splitlines(True), fromfile="/dev/null", tofile=name))
                        rendered = _bounded(rendered)
            except (RuntimeError, ValueError) as exc:
                complete = False
                skipped.append({"path": name, "reason": str(exc)})
        return {"returncode": 0, "fingerprint": digest.hexdigest() if complete else None,
                "complete_fingerprint": complete, "has_changes": bool(size or count or skipped),
                "diff": _bounded(rendered), "stat": stats, "status": status,
                "untracked_count": count, "skipped_untracked": skipped}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {"fingerprint": None, "has_changes": False, "diff": "", "stat": "", "status": "",
                "returncode": 1, "complete_fingerprint": False,
                "error": f"Repository snapshot unavailable: {type(exc).__name__}: {exc}"}
