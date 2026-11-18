#!/usr/bin/env python3
"""Run live-model synthetic repository diagnostics; this is NOT a Harbor score."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "experiments/diagnostics/cases.json"
WRAPPER = ROOT / "experiments/diagnostics/public_statement_wrapper.json"


def render_statement(case):
    """Use the public task's common framing without its original issue content."""
    if WRAPPER.is_file():
        wrapper=json.loads(WRAPPER.read_text())
        return wrapper["prefix"]+case["statement"]+wrapper["suffix"]
    return ("<uploaded_files>\n/testbed\n</uploaded_files>\n\nThe repository is already uploaded at /testbed.\n"
            "<issue_description>\n"+case["statement"]+"\n</issue_description>\n"
            "All dependencies are already installed. Make minimal changes to non-test files only. "
            "Do not modify existing tests, clone or reset the repository, or navigate outside /testbed. "
            "Read and clarify the issue, run existing tests, explore relevant code, reproduce the issue, "
            "analyze the cause, implement the fix, verify it and review the actual diff against every requirement.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend_metadata(base_url):
    import httpx
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.get(base_url.rstrip('/').removesuffix('/v1') + '/local-info')
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        return {'metadata_unavailable': type(exc).__name__}


def source_snapshot(source):
    files = {path.relative_to(source).as_posix(): digest(path) for path in sorted(source.rglob('*.py'))}
    return {'files': files, 'bundle_sha256': hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def safe_test_state(repo, name, expected):
    """Inspect a stopped trial's test without following model-created symlinks."""
    path=repo/name
    try:
        if path.is_symlink():
            raise ValueError("test path became a symlink")
        path.resolve(strict=True).relative_to(repo.resolve(strict=True))
        descriptor=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
        with os.fdopen(descriptor,"rb") as source:
            info=os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size>4*1024*1024:
                raise ValueError("test is not a bounded regular file")
            actual=hashlib.sha256(source.read()).hexdigest()
        expected_hash=expected["sha256"] if isinstance(expected,dict) else expected
        expected_mode=expected.get("mode") if isinstance(expected,dict) else None
        mode=stat.S_IMODE(info.st_mode)
        return {"modified":actual!=expected_hash or (expected_mode is not None and mode!=expected_mode),
                "sha256":actual,"mode":mode}
    except (OSError,ValueError) as exc:
        return {"modified":True,"sha256":None,"mode":None,"error":str(exc)}


async def isolated_repository_state(repo, test_names):
    """Model-writable Git configuration must never run helpers on the host."""
    sys.path.insert(0,str(ROOT))
    from scripts.local_environment import LocalEnvironment
    environment=LocalEnvironment(repo)
    head=await environment.exec(command="git rev-parse HEAD",cwd="/testbed",timeout_sec=10)
    patch=await environment.exec(command="git diff --no-ext-diff --no-textconv --binary HEAD --",cwd="/testbed",timeout_sec=20)
    # Names are seeded by this script, not supplied by the model.
    import shlex
    command="git diff --no-ext-diff --no-textconv --name-only HEAD -- "+" ".join(shlex.quote(name) for name in test_names)
    tests=await environment.exec(command=command,cwd="/testbed",timeout_sec=10)
    errors=[item.stderr or item.stdout for item in (head,patch,tests) if item.return_code]
    return {"head":head.stdout.strip() if head.return_code==0 else None,
            "patch":patch.stdout if patch.return_code==0 else "",
            "changed_test_paths":tests.stdout.splitlines() if tests.return_code==0 else [],
            "error":"\n".join(errors) if errors else None}


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


def seed(case, repo):
    repo.mkdir(parents=True)
    for name, content in case["files"].items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Diagnostic", "-c", "user.email=diagnostic@example.invalid", "commit", "-qm", "synthetic initial state")
    return git(repo, "rev-parse", "HEAD").strip()


def grade_script(case):
    return "import json,sys\nsys.path.insert(0,'/testbed')\n" + case["grader"] + """
results=[]
for name,check in checks:
    try:
        passed=bool(check())
        results.append({'criterion':name,'passed':passed})
    except Exception as exc:
        results.append({'criterion':name,'passed':False,'error':type(exc).__name__+': '+str(exc)})
passed=sum(item['passed'] for item in results)
print(json.dumps({'passed':passed,'total':len(results),'checks':results}))
raise SystemExit(0 if passed==len(results) else 1)
"""


async def worker(request):
    # This process imports only one source variant. Tool subprocesses are isolated
    # by LocalEnvironment; inference HTTP stays in this worker outside that jail.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, request["source"])
    from scripts.local_environment import LocalEnvironment
    from custom_agent.react_bash_agent import ReactBashAgent
    from harbor.models.agent.context import AgentContext

    repo, logs = Path(request["repo"]), Path(request["logs"])
    environment = LocalEnvironment(repo)
    context = AgentContext()
    agent = ReactBashAgent(
        logs_dir=logs, model_name="openai/"+request["model"], workdir="/testbed",
        extra_env={"REACT_AGENT_BASE_URL":request["base_url"], "REACT_AGENT_API_KEY":os.environ.get("REACT_AGENT_API_KEY","dummy")},
        time_budget_sec=request["time_budget_sec"], max_steps=request["max_steps"],
        max_tokens=request["max_tokens"], thinking_budget_tokens=request["thinking_budget_tokens"],
        request_timeout_sec=request["request_timeout_sec"],
        context_budget_tokens=request["context_budget_tokens"],
    )
    start = time.monotonic()
    error = None
    try:
        await asyncio.wait_for(agent.run(request["statement"], environment, context), timeout=request["time_budget_sec"]+20)
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    elapsed = time.monotonic()-start
    visible = await environment.exec(command="python3 -B -m unittest -q test_solution", cwd="/testbed", timeout_sec=20)
    original_script = ("import sys,unittest\nsys.path.insert(0,'/testbed')\n" + request["original_tests"] +
                       "\nresult=unittest.main(argv=['original-visible-tests'],verbosity=1,exit=False).result\nraise SystemExit(0 if result.wasSuccessful() and result.testsRun else 1)\n")
    encoded_original=base64.b64encode(original_script.encode()).decode()
    original_visible=await environment.exec(command="python3 -B -c \"import base64;exec(base64.b64decode('"+encoded_original+"'))\"",cwd="/testbed",timeout_sec=20)
    encoded = base64.b64encode(request["grader"].encode()).decode()
    command = "python3 -B -c \"import base64;exec(base64.b64decode('"+encoded+"'))\""
    acceptance = await environment.exec(command=command, cwd="/testbed", timeout_sec=20)
    try:
        graded = json.loads(acceptance.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        graded = {"passed":0,"total":None,"error":"grader did not return JSON","stdout":acceptance.stdout,"stderr":acceptance.stderr}
    return {
        "elapsed_sec":round(elapsed,3), "agent_error":error,
        "context":context.model_dump(mode="json"),
        "visible_tests":{"returncode":visible.return_code,"stdout":visible.stdout,"stderr":visible.stderr},
        "original_visible_tests":{"returncode":original_visible.return_code,"stdout":original_visible.stdout,"stderr":original_visible.stderr},
        "acceptance":{"returncode":acceptance.return_code,**graded},
        "isolation":"LocalEnvironment bubblewrap; NON_HARBOR diagnostic",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--base-url",default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model",default="qwen35-local")
    parser.add_argument("--before-zip",type=Path,default=ROOT/"dist/mera-mini-stage2.zip")
    parser.add_argument("--after-source",type=Path,default=ROOT/"submission_stage2")
    parser.add_argument("--output",type=Path,default=ROOT/"jobs/local_diagnostics")
    parser.add_argument("--cases",nargs="+",choices=list(json.loads(CASES.read_text())))
    parser.add_argument("--variants",nargs="+",choices=["before","after"],default=["before","after"])
    parser.add_argument("--time-budget-sec",type=float,default=240)
    parser.add_argument("--max-steps",type=int,default=25)
    parser.add_argument("--max-tokens",type=int,default=2048)
    parser.add_argument("--thinking-budget-tokens",type=int,default=512)
    parser.add_argument("--request-timeout-sec",type=float,default=120)
    parser.add_argument("--context-budget-tokens",type=int,default=8000)
    args=parser.parse_args()
    if args.worker:
        request=json.loads(args.worker.read_text())
        result=asyncio.run(worker(request))
        Path(request["result"]).write_text(json.dumps(result,indent=2)+"\n")
        return 0
    if min(args.time_budget_sec,args.max_steps,args.max_tokens,args.thinking_budget_tokens,args.request_timeout_sec,args.context_budget_tokens)<=0:
        parser.error("budgets must be positive")
    output=args.output.resolve()
    if (output/"summary.json").exists() or (output/"sources").exists():
        parser.error("output already contains a run; use a fresh output directory")
    output.mkdir(parents=True,exist_ok=True)
    all_cases=json.loads(CASES.read_text())
    selected=args.cases or list(all_cases)
    sources=output/"sources"
    sources.mkdir()
    if "before" in args.variants:
        before=sources/"before"
        before.mkdir()
        with zipfile.ZipFile(args.before_zip) as archive:
            for item in archive.infolist():
                target=(before/item.filename).resolve()
                target.relative_to(before.resolve())
            archive.extractall(before)
    if "after" in args.variants:
        shutil.copytree(args.after_source,sources/"after",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    summary={"kind":"NON_HARBOR_SYNTHETIC_DIAGNOSTIC","model":args.model,"base_url":args.base_url,
             "attempts_per_case_per_variant":1,"settings":{key:getattr(args,key) for key in (
                 "time_budget_sec","max_steps","max_tokens","thinking_budget_tokens","request_timeout_sec","context_budget_tokens")},
             "baseline_archive_sha256":digest(args.before_zip) if "before" in args.variants else None,
             "statement_format":"public_task_common_wrapper_v2_preinstalled_non_test_eight_phases",
             "backend_metadata":backend_metadata(args.base_url),
             "source_snapshots":{variant:source_snapshot(sources/variant) for variant in args.variants},
             "notes":["Synthetic tasks are not SWE MERA tasks or a competition score.",
                      "This experiment uses the public common task wrapper; it is a different condition from v001 plain issue text.",
                      "Visible tests are available to the agent; independent acceptance checks run only after the attempt.",
                      "Same local model and configured budgets are used for both variants."],"trials":[]}
    for variant in args.variants:
        for name in selected:
            case=all_cases[name]
            statement=render_statement(case)
            trial=output/variant/name
            repo=trial/"repo"
            initial_head=seed(case,repo)
            original_tests={path:{"sha256":digest(repo/path),"mode":stat.S_IMODE((repo/path).stat().st_mode)}
                            for path in case["files"] if path.startswith("test_")}
            (trial/"statement.txt").write_text(statement)
            request={"source":str(sources/variant),"repo":str(repo),"logs":str(trial/"agent"),
                     "result":str(trial/"result.json"),"statement":statement,"grader":grade_script(case),
                     "model":args.model,"base_url":args.base_url,"original_tests":case["files"]["test_solution.py"],
                     **summary["settings"]}
            # Local key is normally dummy; keep it out of the persisted request.
            request_path=trial/"request.json"
            request_path.write_text(json.dumps(request,indent=2)+"\n")
            request_path.chmod(0o600)
            print(f"START {variant}/{name}",flush=True)
            started=time.monotonic()
            try:
                process=subprocess.run([sys.executable,"-I",str(Path(__file__).resolve()),"--worker",str(request_path)],
                                       capture_output=True,text=True,timeout=args.time_budget_sec+80)
            except subprocess.TimeoutExpired as exc:
                process=SimpleNamespace(returncode=124,stdout=str(exc.stdout or ""),stderr="Diagnostic worker exceeded total timeout")
            (trial/"worker.stdout").write_text(process.stdout)
            (trial/"worker.stderr").write_text(process.stderr)
            result_path=trial/"result.json"
            result=json.loads(result_path.read_text()) if result_path.exists() else {
                "agent_error":f"worker exited {process.returncode}","elapsed_sec":round(time.monotonic()-started,3),
                "acceptance":{"returncode":None},"visible_tests":{"returncode":None},"original_visible_tests":{"returncode":None}}
            sanitized=dict(request)
            sanitized.pop("api_key",None)
            request_path.write_text(json.dumps(sanitized,indent=2)+"\n")
            test_states={path:safe_test_state(repo,path,old) for path,old in original_tests.items()}
            modified=[path for path,state in test_states.items() if state["modified"]]
            repository_error=None
            try:
                isolated=asyncio.run(isolated_repository_state(repo,original_tests))
                patch=isolated["patch"]
                head_changed=isolated["head"]!=initial_head
                repository_error=isolated["error"]
                modified=sorted(set(modified)|set(isolated["changed_test_paths"]))
            except Exception as exc:
                patch=""
                head_changed=True
                repository_error="Repository metadata unavailable after agent: "+str(exc)
            (trial/"model_patch.diff").write_text(patch)
            result.update({"variant":variant,"case":name,"visible_tests_modified":modified,"visible_test_states":test_states,
                           "head_changed":head_changed,"repository_error":repository_error,
                           "patch_nonempty":bool(patch.strip()),"source_sha256":digest(sources/variant/"custom_agent/react_bash_agent.py")})
            acceptance_ok=(result["acceptance"].get("returncode")==0
                           and isinstance(result["acceptance"].get("total"),int)
                           and result["acceptance"]["total"]>0
                           and result["acceptance"].get("passed")==result["acceptance"]["total"]
                           and not result["acceptance"].get("error"))
            result["resolved"]=bool(not result.get("agent_error") and acceptance_ok
                                    and result["visible_tests"]["returncode"]==0 and result["original_visible_tests"]["returncode"]==0 and not modified
                                    and not result["head_changed"] and result["patch_nonempty"])
            result_path.write_text(json.dumps(result,indent=2)+"\n")
            summary["trials"].append({key:result.get(key) for key in (
                "variant","case","resolved","elapsed_sec","agent_error","acceptance","visible_tests_modified","patch_nonempty")})
            (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            print(f"END {variant}/{name}: resolved={result['resolved']} time={result['elapsed_sec']}s",flush=True)
    summary["by_variant"]={variant:{"resolved":sum(trial["resolved"] for trial in summary["trials"] if trial["variant"]==variant),
                                   "total":sum(trial["variant"]==variant for trial in summary["trials"])} for variant in args.variants}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary["by_variant"],indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
