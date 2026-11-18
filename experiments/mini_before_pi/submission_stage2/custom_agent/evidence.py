"""Observed checks tied to repository contents; all inspection runs via Harbor."""
from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex


_SNAPSHOT_HELPER = r'''
import base64,difflib,hashlib,json,os,stat,subprocess,sys,tempfile,time
from pathlib import Path
p=json.loads(base64.b64decode(sys.argv[1]))
root=Path(p['workdir']).resolve(strict=True)
deadline=time.monotonic()+p['timeout']
limit=4*1024*1024
volatile={'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.venv','node_modules','coverage','logs'}
exclusions=[':(exclude)'+name for name in p.get('ignored_paths',[])]
def remaining():
    value=deadline-time.monotonic()
    if value<=0: raise TimeoutError('repository snapshot deadline expired')
    return value
def git(*args,stdout=None):
    return subprocess.run(['git','--no-optional-locks',*args],cwd=root,stdout=stdout or subprocess.PIPE,
                          stderr=subprocess.PIPE,timeout=remaining())
def bounded(text,n=24000):
    if len(text)<=n: return text
    head=n*2//3
    return text[:head]+'\n... ['+str(len(text)-n)+' characters omitted; inspect relevant file ranges] ...\n'+text[-(n-head):]
def run():
    with tempfile.TemporaryFile(dir=root) as patch:
        result=git('diff','--no-ext-diff','--no-textconv','--binary','HEAD','--','.',*exclusions,stdout=patch)
        if result.returncode:
            return {'returncode':result.returncode,'fingerprint':None,'has_changes':False,'diff':'',
                    'stat':'','status':'','error':result.stderr.decode('utf-8','replace')[:2000],
                    'complete_fingerprint':False}
        size=patch.tell()
        digest=hashlib.sha256(b'mera-evidence-v1\0'+str(size).encode()+b'\0')
        patch.seek(0)
        while True:
            block=patch.read(1024*1024)
            if not block: break
            digest.update(block)
            remaining()
        rendered=''
        if p['include_diff']:
            patch.seek(0)
            head=patch.read(24000)
            if size>24000:
                patch.seek(max(0,size-8000))
                head+=b'\n... [patch bytes omitted; inspect targeted files] ...\n'+patch.read(8000)
            rendered=head.decode('utf-8','replace')
    status=git('status','--short','--untracked-files=all')
    stats=git('diff','--no-ext-diff','--no-textconv','--stat','HEAD','--','.',*exclusions)
    others=git('ls-files','--others','--exclude-standard','-z')
    if any(item.returncode for item in (status,stats,others)):
        raise RuntimeError('git status/stat/untracked inspection failed')
    complete=True
    skipped=[]
    untracked_count=0
    total=0
    for raw_name in sorted(name for name in others.stdout.split(b'\0') if name):
        name=os.fsdecode(raw_name)
        relative=Path(name)
        if name in p.get('ignored_paths',[]):
            continue
        if any(part in volatile for part in relative.parts) or relative.suffix in {'.pyc','.pyo','.log'}:
            continue
        file=root/relative
        if file.is_symlink():
            skipped.append({'path':name,'reason':'symlink target was not read'})
            digest.update(b'\0symlink\0'+raw_name+b'\0'+os.fsencode(os.readlink(file)))
            untracked_count+=1
            continue
        try:
            file.resolve(strict=True).relative_to(root)
            if not file.is_file():
                raise ValueError('not a regular repository file')
            info=file.stat()
            if untracked_count>=128 or info.st_size>limit or total+info.st_size>limit:
                raise ValueError('untracked content exceeds bounded snapshot limit')
            content=file.read_bytes()
        except (OSError,ValueError) as exc:
            complete=False
            skipped.append({'path':name,'reason':str(exc)})
            continue
        total+=len(content)
        untracked_count+=1
        digest.update(b'\0untracked\0'+raw_name+b'\0'+str(stat.S_IMODE(info.st_mode)).encode()+b'\0'+str(len(content)).encode()+b'\0'+content)
        if p['include_diff']:
            quoted=json.dumps(name,ensure_ascii=False)
            if b'\0' in content:
                rendered+='\nNew untracked binary file '+quoted+' ('+str(len(content))+' bytes)\n'
            else:
                text=content.decode('utf-8','replace')
                rendered+='\nNew untracked file '+quoted+'\n'+''.join(difflib.unified_diff([],text.splitlines(True),fromfile='/dev/null',tofile=name))
                rendered=bounded(rendered)
        remaining()
    return {'returncode':0,'fingerprint':digest.hexdigest() if complete else None,
            'complete_fingerprint':complete,'has_changes':bool(size or untracked_count or skipped),
            'diff':bounded(rendered),'stat':bounded(stats.stdout.decode('utf-8','replace'),8000),
            'status':bounded(status.stdout.decode('utf-8','replace'),8000),
            'untracked_count':untracked_count,'skipped_untracked':skipped}
try:
    data=run()
except Exception as exc:
    data={'returncode':1,'fingerprint':None,'complete_fingerprint':False,'has_changes':False,
          'diff':'','stat':'','status':'','error':type(exc).__name__+': '+str(exc)}
print(json.dumps(data,ensure_ascii=True))
raise SystemExit(data['returncode'])
'''


def _bounded(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    head = limit * 2 // 3
    return value[:head] + f"\n... [{len(value)-limit} characters omitted] ...\n" + value[-(limit-head):]


class EvidenceTracker:
    def __init__(self, environment, workdir: str = "/testbed", logs_dir: Path | None = None, ignored_paths=()):
        if not isinstance(workdir, str) or not workdir.startswith("/"):
            raise ValueError("workdir must be an absolute task-container path")
        self.environment, self.workdir = environment, workdir
        self.ignored_paths = tuple(ignored_paths)
        if any(not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts for name in self.ignored_paths):
            raise ValueError("ignored paths must be relative repository paths")
        self.logs_dir = None if logs_dir is None else Path(logs_dir)
        self.checks: list[dict] = []

    def _save(self):
        if self.logs_dir is None:
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        target = self.logs_dir / "evidence.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"checks": self.checks}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    async def snapshot(self, timeout_sec: float = 20, include_diff: bool = True) -> dict:
        if isinstance(timeout_sec, bool) or not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("snapshot timeout must be positive and finite")
        payload = {"workdir": self.workdir, "timeout": timeout_sec * 0.9, "include_diff": include_diff, "ignored_paths": self.ignored_paths}
        source = base64.b64encode(_SNAPSHOT_HELPER.encode()).decode()
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        bootstrap = "import base64;exec(base64.b64decode(" + repr(source) + "))"
        command = "python3 -c " + shlex.quote(bootstrap) + " " + shlex.quote(encoded)
        try:
            async with asyncio.timeout(timeout_sec):
                result = await self.environment.exec(command=command, cwd=self.workdir, timeout_sec=timeout_sec)
            try:
                data = json.loads(result.stdout or "")
            except ValueError:
                raise ValueError(f"snapshot helper exit {result.return_code}: {_bounded(result.stderr or result.stdout or 'no JSON output', 2000)}") from None
            if not isinstance(data, dict) or "fingerprint" not in data:
                raise ValueError("snapshot helper returned invalid JSON")
            if result.return_code and not data.get("error"):
                data["error"] = result.stderr or f"snapshot helper exit {result.return_code}"
                data.update(returncode=result.return_code, fingerprint=None, complete_fingerprint=False)
            return data
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {"fingerprint": None, "has_changes": False, "diff": "", "stat": "", "status": "",
                    "returncode": 1, "complete_fingerprint": False,
                    "error": f"Repository snapshot unavailable: {type(exc).__name__}: {exc}"}

    def record_check(self, command: str, result: dict, before: dict, after: dict) -> dict:
        code = result.get("returncode")
        if isinstance(code, bool) or not isinstance(code, int):
            code = None
        record = {"command": command, "returncode": code,
                  "observed_at": datetime.now(timezone.utc).isoformat(),
                  "before_fingerprint": before.get("fingerprint"), "after_fingerprint": after.get("fingerprint"),
                  "output": _bounded(str(result.get("raw_output") or result.get("output") or ""))}
        record["changed_during_check"] = bool(record["before_fingerprint"] and record["after_fingerprint"]
                                               and record["before_fingerprint"] != record["after_fingerprint"])
        record["advisories"] = []
        if re.search(r"\bRan\s+0\s+tests?\b|\bcollected\s+0\s+items?\b|\bno tests (?:ran|executed|found)\b", record["output"], re.I):
            record["advisories"].append("Output reports zero tests. This command does not show that the intended test cases executed.")
        elif code == 0 and not re.search(r"\bRan\s+[1-9]\d*\s+tests?\b|\b[1-9]\d*\s+passed\b", record["output"]):
            try:
                tokens = shlex.split(command)
            except ValueError:
                tokens = []
            if tokens and re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", Path(tokens[0]).name):
                script = next((token for token in tokens[1:] if not token.startswith("-")), "")
                if not any(flag in tokens[1:] for flag in ("-m", "-c")) and (
                        Path(script).name.startswith("test") or Path(script).name.endswith("_test.py")) and script.endswith(".py"):
                    record["advisories"].append("A direct test-file invocation returned 0 without a test count. It may only define unittest classes. For unittest, use python3 -m unittest discovery or the repository's configured runner; a silent assert reproducer needs its actual assertions reviewed.")
        self.checks.append(record)
        self._save()
        return record

    @staticmethod
    def _current(record: dict, fingerprint: str | None) -> bool:
        return bool(fingerprint and record["before_fingerprint"] == record["after_fingerprint"] == fingerprint)

    def has_current_check(self, fingerprint: str | None, successful_only: bool = True) -> bool:
        return any(self._current(record, fingerprint) and (not successful_only or record["returncode"] == 0)
                   for record in self.checks)

    def summary(self, currentfingerprint: str | None) -> str:
        lines = ["Observed validation commands (exit codes do not prove semantic correctness):"]
        if not self.checks:
            return lines[0] + "\nNo validation commands have been recorded. A text claim of passing tests is not evidence."
        if currentfingerprint is None:
            lines.append("Current repository fingerprint unavailable; validation freshness is unknown.")
        for record in self.checks[-8:]:
            freshness = "current" if self._current(record, currentfingerprint) else (
                "unknown freshness" if not currentfingerprint or not record["after_fingerprint"] else "stale")
            code = record["returncode"]
            status = "completion unconfirmed (timeout)" if code == 124 else (
                "command unavailable (127)" if code == 127 else f"command returned {code}" if code is not None else "exit code unknown")
            lines.append(f"- {freshness}: {status}: {_bounded(str(record['command']), 300)}")
            if record["changed_during_check"]:
                lines.append("  Repository changed during this command; run a focused check again after reviewing edits.")
            for advisory in record.get("advisories", []):
                lines.append("  Advisory: " + advisory)
            if code != 0:
                lines.append("  Observed output: " + _bounded(record["output"], 700))
        older_failures = [record for record in self.checks[:-8] if record["returncode"] not in {0, None}]
        if older_failures:
            lines.append("Earlier failed commands (historical; later checks may supersede them):")
            for record in older_failures[-3:]:
                lines.append(f"- command returned {record['returncode']}: {_bounded(str(record['command']), 300)}")
        return "\n".join(lines)
