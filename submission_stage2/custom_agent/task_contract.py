"""Read task-specific deliverables and export requested patches through Harbor."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
import shlex


@dataclass(frozen=True)
class TaskContract:
    issue_text: str
    non_test_only: bool = False
    dependencies_preinstalled: bool = False
    patch_path: str | None = None
    workdir: str = "/testbed"

    @classmethod
    def from_instruction(cls, instruction: str, workdir: str = "/testbed") -> "TaskContract":
        root = PurePosixPath(workdir)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("workdir must be an absolute repository path")
        match = re.search(r"<issue_description>([\s\S]*?)</issue_description>", instruction, re.I)
        issue = match.group(1).strip() if match else instruction.strip()
        constraints = instruction[:match.start()] + instruction[match.end():] if match else instruction
        non_test = bool(re.search(r"non[- ]test files|do not (?:modify or add|modify|change) test files", constraints, re.I))
        installed = bool(re.search(r"dependencies (?:are )?(?:already installed|pre[- ]?installed)", constraints, re.I))
        section = re.search(r"(?im)^#{1,6}\s*(?:submission|deliverables?|output)(?:\s*:?\s*)$", constraints)
        deliverable = constraints[section.end():] if section else constraints
        target = re.search(
            r"\b(?:write|save|output|generate)\b[\s\S]{0,200}?\b(?:unified\s+diff|patch)\b"
            r"[\s\S]{0,80}?\b(?:to|at|in)\b\s*:?\s*[`'\"]?([^\s`'\"<>]+?\.(?:diff|patch))",
            deliverable, re.I,
        )
        patch = None
        if target:
            requested = PurePosixPath(target.group(1))
            if ".." in requested.parts:
                raise ValueError("patch path must remain inside the task repository")
            requested = requested if requested.is_absolute() else root / requested
            try:
                relative = requested.relative_to(root)
            except ValueError:
                raise ValueError("patch path must remain inside the task repository") from None
            if not relative.parts or relative.parts[0] == ".git":
                raise ValueError("patch path must be an ordinary repository file")
            patch = str(requested)
        return cls(issue, non_test, installed, patch, str(root))

    def guidance(self) -> str:
        items = ["Task contract: implement every requested behavior; verify examples and relevant edge cases.",
                 "Follow repository conventions and use focused existing tests before broad suites."]
        if self.non_test_only:
            items.append("Change non-test files only. Read/run existing tests; do not modify or add tests. Remove only your own temporary reproduction artifacts before finishing.")
        if self.dependencies_preinstalled:
            items.append("Dependencies are already installed. Discover the existing project test command before considering installs or upgrades.")
        if self.patch_path:
            items.append(f"Required deliverable: a complete applicable unified diff at {self.patch_path}. The harness exports and checks it when finishing; keep binary/generated artifacts out of the patch.")
        items.append(f"Keep repository operations and your temporary files within {self.workdir}; preserve pre-existing work.")
        return "\n".join(items)


_EXPORT = r'''
import base64,json,os,re,subprocess,sys,tempfile
from pathlib import Path

def export(payload):
    root=Path(payload['workdir']).resolve(strict=True)
    target=relative=None
    if payload['path']:
        requested=Path(payload['path'])
        if '..' in requested.parts:
            raise ValueError('patch path contains parent traversal')
        candidate=requested if requested.is_absolute() else root/requested
        if candidate.is_symlink():
            raise ValueError('patch output must not be a symlink')
        target=candidate.resolve(strict=False)
        relative=target.relative_to(root)
        if not relative.parts or relative.parts[0]=='.git':
            raise ValueError('patch output must be an ordinary repository file')
    target_name=str(target) if target is not None else None
    def git(args,env):
        result=subprocess.run(['git',*args],cwd=root,env=env,capture_output=True)
        if result.returncode:
            raise ValueError('git '+args[0]+' failed: '+result.stderr.decode(errors='replace')[:3000])
        return result.stdout
    def is_test(name):
        path=Path(name)
        if any(part.lower() in {'test','tests','testing','__tests__','spec','specs'} for part in path.parts[:-1]):
            return True
        file=path.name
        return bool(re.search(r'(^test_.*\.py$|^test.*\.php$|_test\.go$|(?:Test|Spec)\.php$|\.(?:test|spec)\.[cm]?[jt]sx?$|^conftest\.py$)',file,re.I))
    with tempfile.TemporaryDirectory(prefix='.mera-export-',dir=root) as directory:
        temporary=Path(directory)
        env=dict(os.environ,GIT_INDEX_FILE=str(temporary/'index'))
        git(['read-tree','HEAD'],env)
        exclusions=[':(exclude,glob).mera-export-*/**',
                    ':(exclude,glob)**/junit.xml',':(exclude,glob)**/task_tests.json',
                    ':(exclude,glob)**/*.orig',':(exclude,glob)**/*.rej']
        if relative is not None:
            exclusions.append(':(exclude)'+str(relative))
        git(['add','-A','--','.',*exclusions],env)
        names=[os.fsdecode(name) for name in git(['diff','--cached','--no-renames','--name-only','-z','HEAD'],env).split(b'\0') if name]
        rejected=[name for name in names if payload['non_test_only'] and is_test(name)]
        if rejected:
            return {'valid':False,'written':False,'path':target_name,'returncode':1,'output':'Test changes are forbidden; inspect and repair without reverting pre-existing work.','rejected_test_paths':rejected}
        binary=[entry for entry in git(['diff','--cached','--no-renames','--numstat','-z','HEAD'],env).split(b'\0') if entry.startswith(b'-\t-\t')]
        if binary and target is not None:
            return {'valid':False,'written':False,'path':target_name,'returncode':1,'output':'Patch not exported: binary changes are forbidden.','binary_entries':[os.fsdecode(item) for item in binary]}
        if payload['validate_only']:
            return {'valid':True,'written':False,'path':target_name,'returncode':0,'output':'Current changes satisfy the detected file constraints.','changed_paths':names}
        patch=git(['diff','--cached','--no-renames','--full-index','--no-color','HEAD'],env)
        if not patch:
            return {'valid':False,'written':False,'path':target_name,'returncode':1,'output':'Patch not exported: no implementation changes were found.'}
        source=temporary/'change.patch'
        source.write_bytes(patch)
        validation=dict(os.environ,GIT_INDEX_FILE=str(temporary/'validation-index'))
        git(['read-tree','HEAD'],validation)
        git(['apply','--cached','--check',str(source)],validation)
        target.parent.mkdir(parents=True,exist_ok=True)
        os.replace(source,target)
        return {'valid':True,'written':True,'path':target_name,'returncode':0,'output':'Exported unified diff and verified it applies to the original HEAD index.','bytes':len(patch),'changed_paths':names,'apply_check_passed':True}

try:
    result=export(json.loads(base64.b64decode(sys.argv[1]).decode()))
except Exception as exc:
    result={'valid':False,'written':False,'returncode':1,'output':'Patch validation/export failed: '+type(exc).__name__+': '+str(exc)}
print(json.dumps(result,ensure_ascii=True))
raise SystemExit(result['returncode'])
'''


async def _execute(environment, contract: TaskContract, timeout_sec: float, validate_only: bool) -> dict:
    if timeout_sec <= 0:
        return {"valid": False, "written": False, "path": contract.patch_path, "returncode": 124, "output": "Patch validation/export deadline expired; no command started."}
    source = base64.b64encode(_EXPORT.encode()).decode("ascii")
    payload = base64.b64encode(json.dumps({"workdir": contract.workdir, "path": contract.patch_path,
                                         "non_test_only": contract.non_test_only, "validate_only": validate_only}).encode()).decode("ascii")
    program = "import base64;exec(base64.b64decode(" + repr(source) + "))"
    command = "python3 -c " + shlex.quote(program) + " " + shlex.quote(payload)
    try:
        async with asyncio.timeout(timeout_sec):
            response = await environment.exec(command=command, cwd=contract.workdir, timeout_sec=timeout_sec)
        value = json.loads(response.stdout or "")
        if not isinstance(value, dict):
            raise ValueError("helper result is not an object")
        value.setdefault("path", contract.patch_path)
        return value
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return {"valid": False, "written": False, "path": contract.patch_path, "returncode": 124, "output": "Patch validation/export timed out; success is unconfirmed."}
    except Exception as exc:
        detail = (getattr(locals().get("response"), "stderr", "") or "")[:3000]
        return {"valid": False, "written": False, "path": contract.patch_path, "returncode": 1, "output": f"Patch validation/export failed: {type(exc).__name__}: {exc}. {detail}"}


async def validate_changes(environment, contract: TaskContract, timeout_sec: float = 20) -> dict:
    """Check detected file constraints even when no separate diff file is requested."""
    if not contract.non_test_only and not contract.patch_path:
        return {"valid": True, "written": False, "path": None, "returncode": 0, "output": "No special file constraints detected.", "skipped": True}
    return await _execute(environment, contract, timeout_sec, validate_only=True)


async def write_requested_patch(environment, contract: TaskContract, timeout_sec: float = 20) -> dict:
    if contract.patch_path is None:
        return {"written": False, "path": None, "returncode": 0, "output": "No separate patch artifact requested.", "skipped": True}
    return await _execute(environment, contract, timeout_sec, validate_only=False)
