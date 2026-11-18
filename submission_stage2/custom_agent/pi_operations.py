"""Remote operations used by Pi's native tools, executed only through Harbor."""
from __future__ import annotations

import asyncio
import base64
import json
import math
from pathlib import PurePosixPath
import shlex
import secrets

from .tools import repository_command_error
from .pi_portable import has_python, portable_file_operation


FILE_HELPER = r'''
import base64,json,os,stat,subprocess,sys,tempfile
from pathlib import Path
p=json.loads(base64.b64decode(sys.argv[1]))
root=Path(p['workdir']).resolve(strict=True)
args=p['args'];requested=Path(args['path'])
if '..' in requested.parts: raise ValueError('path must remain inside the task repository')
path=(requested if requested.is_absolute() else root/requested).resolve(strict=False)
path.relative_to(root)
operation=p['operation']
if operation=='read_file':
    if not path.is_file(): raise ValueError('file does not exist or is not a regular file')
    if path.stat().st_size>4*1024*1024: raise ValueError('file exceeds 4MiB; use a focused shell read')
    result={'data_base64':base64.b64encode(path.read_bytes()).decode()}
elif operation=='access':
    if not path.exists(): raise FileNotFoundError(str(requested))
    mode=os.W_OK if args.get('write') else os.R_OK
    if not os.access(path,mode): raise PermissionError(str(requested))
    result={}
elif operation=='mkdir':
    path.mkdir(parents=True,exist_ok=True);result={}
elif operation=='write_file':
    if path==root: raise ValueError('path must name a file')
    if args.get('data_path'):
        transfer=(root/args['data_path']).resolve(strict=True)
        transfer.relative_to(root)
        if not transfer.name.startswith('.mera-pi-transfer-'): raise ValueError('invalid transfer')
        data=transfer.read_bytes()
    else:
        data=base64.b64decode(args['data_base64'],validate=True)
    if len(data)>4*1024*1024: raise ValueError('write exceeds 4MiB')
    existed=path.exists()
    if existed and not path.is_file(): raise ValueError('target is not a regular file')
    mode=stat.S_IMODE(path.stat().st_mode) if existed else 0o644
    descriptor,name=tempfile.mkstemp(prefix='.mera-pi-write-',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as out: out.write(data)
        os.chmod(name,mode);os.replace(name,path)
    finally:
        try: os.unlink(name)
        except FileNotFoundError: pass
    result={}
    if not existed:
        added=subprocess.run(['git','add','-N','--',str(path.relative_to(root))],cwd=root,capture_output=True,timeout=10)
        if added.returncode: result['warning']='New file written, but git add -N failed: '+added.stderr.decode('utf-8','replace')[:1000]
else: raise ValueError('unsupported remote file operation')
print(json.dumps(result))
'''


def _path(path, workdir):
    if not isinstance(path, str) or not path or "\0" in path:
        raise ValueError("operation path must be nonempty text")
    value = PurePosixPath(path)
    if ".." in value.parts:
        raise ValueError("parent traversal outside the task is not allowed")
    if value.is_absolute() and value != PurePosixPath(workdir) and PurePosixPath(workdir) not in value.parents:
        raise ValueError("path must remain inside the task repository")
    return str(value)


class PiOperations:
    def __init__(self, environment, budget, *, workdir="/testbed", max_output_chars=1024 * 1024):
        self.environment, self.budget = environment, budget
        self.workdir, self.max_output_chars = workdir, max_output_chars
        self.records = []

    @staticmethod
    def _clip(text, maximum):
        value = str(text or "")
        if len(value) <= maximum:
            return value
        return value[:maximum * 2 // 3] + "\n[remote output truncated; inspect a focused range]\n" + value[-maximum // 3:]

    async def execute(self, operation, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("operation arguments must be an object")
        record = {"operation": operation, "arguments": {key: (f'<{len(value)} base64 characters>' if key == 'data_base64' else value)
                  for key, value in arguments.items()}}
        self.records.append(record)
        transfer = None
        try:
            if operation == "exec":
                command = arguments.get("command")
                if not isinstance(command, str) or not command.strip():
                    raise ValueError("command must be nonempty text")
                cwd = _path(arguments.get("cwd", self.workdir), self.workdir)
                if not PurePosixPath(cwd).is_absolute():
                    cwd = str(PurePosixPath(self.workdir) / cwd)
                guard = repository_command_error(command, self.workdir)
                if guard:
                    result = {"stdout": "", "stderr": guard, "returncode": 1}
                else:
                    cap = arguments.get("timeout_sec", 120)
                    if isinstance(cap, bool) or not isinstance(cap, (float, int)) or not math.isfinite(cap) or cap <= 0:
                        raise ValueError("timeout must be positive and finite")
                    timeout = self.budget.timeout(min(cap, 300))
                    async with asyncio.timeout(timeout):
                        output = await self.environment.exec(command=command, cwd=cwd, timeout_sec=timeout)
                    result = {"stdout": self._clip(output.stdout, self.max_output_chars),
                              "stderr": self._clip(output.stderr, self.max_output_chars),
                              "returncode": output.return_code}
            elif operation in {"read_file", "access", "write_file", "mkdir"}:
                arguments = dict(arguments, path=_path(arguments.get("path"), self.workdir))
                if operation == "write_file":
                    data = arguments.get("data_base64")
                    if not isinstance(data, str) or len(data) > 6 * 1024 * 1024:
                        raise ValueError("write requires bounded base64 file bytes")
                    if len(base64.b64decode(data, validate=True)) > 4 * 1024 * 1024:
                        raise ValueError("write exceeds 4MiB")
                if not await has_python(self.environment, self.workdir):
                    result = await portable_file_operation(self.environment, self.workdir, operation, arguments,
                                                          timeout_sec=self.budget.timeout(60))
                    record["result"] = {key: (f"<{len(value)} base64 characters>" if key == "data_base64" else value)
                                        for key, value in result.items()}
                    return result
                if operation == "write_file":
                    data = arguments.get("data_base64")
                    if not isinstance(data, str) or len(data) > 6 * 1024 * 1024:
                        raise ValueError("write requires bounded base64 file bytes")
                    decoded = base64.b64decode(data, validate=True)
                    if len(decoded) > 4 * 1024 * 1024:
                        raise ValueError('write exceeds 4MiB')
                    # Native edit writes the whole source file; keep each exec
                    # argument below the platform's limit using task-local chunks.
                    if len(data) > 24000:
                        transfer = '.mera-pi-transfer-' + secrets.token_hex(12)
                        for offset in range(0, len(data), 24000):
                            mode = 'xb' if offset == 0 else 'ab'
                            script = 'import base64;from pathlib import Path;p=Path(' + repr(transfer) + ');assert not p.is_symlink();p.open(' + repr(mode) + ').write(base64.b64decode(' + repr(data[offset:offset+24000]) + '))'
                            timeout = self.budget.timeout(15)
                            response = await self.environment.exec(command='python3 -c ' + shlex.quote(script), cwd=self.workdir, timeout_sec=timeout)
                            if response.return_code:
                                raise RuntimeError('Source transfer failed: ' + (response.stderr or '')[:1000])
                        arguments.pop('data_base64')
                        arguments['data_path'] = transfer
                payload = base64.b64encode(json.dumps({"operation": operation, "args": arguments,
                                                       "workdir": self.workdir}).encode()).decode()
                source = base64.b64encode(FILE_HELPER.encode()).decode()
                command = "python3 -c " + shlex.quote("import base64;exec(base64.b64decode(" + repr(source) + "))") + " " + shlex.quote(payload)
                timeout = self.budget.timeout(20)
                async with asyncio.timeout(timeout):
                    output = await self.environment.exec(command=command, cwd=self.workdir, timeout_sec=timeout)
                if output.return_code:
                    raise RuntimeError(self._clip(output.stderr or output.stdout, 3000))
                result = json.loads(output.stdout)
            else:
                raise ValueError("Unknown Pi operation: " + str(operation))
            # Large file bytes stay out of human-readable audit records.
            record["result"] = {key: (f"<{len(value)} base64 characters>" if key == "data_base64" else value)
                                for key, value in result.items()}
            return result
        except TimeoutError:
            result = {"stdout": "", "stderr": "Operation timed out; completion is unconfirmed.",
                      "returncode": 124, "timed_out": True}
            record["result"] = result
            if operation == "exec":
                return result
            raise RuntimeError("Remote file operation timed out") from None
        except asyncio.CancelledError:
            record["cancelled"] = True
            raise
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if transfer:
                cleanup = 'from pathlib import Path;Path(' + repr(transfer) + ').unlink(missing_ok=True)'
                try:
                    await self.environment.exec(command='python3 -c ' + shlex.quote(cleanup), cwd=self.workdir, timeout_sec=5)
                except Exception:
                    record['cleanup_unconfirmed'] = True
