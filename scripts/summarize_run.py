#!/usr/bin/env python3
"""Summarize one Harbor run using original per-trial reward files, read-only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def summarize_run(run_dir: Path | str, expected_tasks: int | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if not (run_dir / "config.json").is_file() or not (run_dir / "result.json").is_file():
        raise ValueError("provide one timestamped Harbor run containing config.json and result.json, not its parent jobs directory")
    config = _load_object(run_dir / "config.json")
    run_result = _load_object(run_dir / "result.json")
    attempts = config.get("n_attempts", 1)
    warnings = []
    if attempts != 1:
        warnings.append(f"n_attempts={attempts!r}; this is not an eligible pass@1 competition run")
    trials = []
    for trial_dir in sorted(run_dir.iterdir()):
        if not trial_dir.is_dir() or trial_dir.name.startswith("."):
            continue
        if not ((trial_dir / "result.json").exists() or (trial_dir / "verifier").is_dir() or "__" in trial_dir.name):
            continue
        trial: dict[str, Any] = {
            "trial": trial_dir.name, "task": trial_dir.name.split("__", 1)[0],
            "reward": None, "finished": False, "error": None, "timeout": False,
        }
        try:
            result = _load_object(trial_dir / "result.json")
            trial["task"] = result.get("task_name") or trial["task"]
            trial["finished"] = bool(result.get("finished_at"))
            exception = result.get("exception_info")
            if exception:
                kind = str(exception.get("exception_type", "UnknownError")) if isinstance(exception, dict) else "InvalidExceptionInfo"
                trial["error"] = kind
                trial["timeout"] = "timeout" in kind.lower()
        except ValueError as exc:
            trial["error"] = str(exc)
        reward_path = trial_dir / "verifier" / "reward.txt"
        if reward_path.is_file():
            try:
                value = float(reward_path.read_text(encoding="utf-8").strip())
                if not math.isfinite(value) or value not in (0.0, 1.0):
                    raise ValueError("reward must be 0 or 1")
                trial["reward"] = int(value)
            except (OSError, UnicodeError, ValueError) as exc:
                trial["reward_error"] = str(exc)
        trials.append(trial)
    counts = Counter(trial["task"] for trial in trials)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        warnings.append("multiple trials for tasks: " + ", ".join(duplicates))
    declared = run_result.get("n_total_trials")
    if expected_tasks is not None:
        if expected_tasks <= 0:
            raise ValueError("expected_tasks must be positive")
        denominator = expected_tasks
        if isinstance(declared, int) and declared != expected_tasks:
            warnings.append(f"run declares {declared} trials but expected task count is {expected_tasks}")
    elif isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
        denominator = declared
    else:
        denominator = len(trials)
        warnings.append("run has no positive n_total_trials; denominator inferred from observed trial directories")
    if len(trials) > denominator:
        warnings.append(f"observed {len(trials)} trials exceeds expected count {denominator}")
    solved = sum(trial["reward"] == 1 for trial in trials)
    failed = sum(trial["reward"] == 0 for trial in trials)
    missing_rewards = sum(trial["reward"] is None and "reward_error" not in trial for trial in trials)
    invalid_rewards = sum("reward_error" in trial for trial in trials)
    errors = sum(trial["error"] is not None for trial in trials)
    unfinished = sum(not trial["finished"] for trial in trials)
    missing_dirs = max(denominator - len(trials), 0)
    valid_attempts = attempts == 1 and not duplicates and len(trials) <= denominator
    run_finished = bool(run_result.get("finished_at"))
    complete = bool(
        run_finished and valid_attempts and denominator > 0
        and not (missing_rewards or invalid_rewards or errors or unfinished or missing_dirs)
    )
    score = solved / denominator if denominator and valid_attempts else None
    reported_means = {}
    stats = run_result.get("stats") or {}
    for name, evaluation in (stats.get("evals") or {}).items():
        for metric in evaluation.get("metrics") or []:
            if isinstance(metric, dict) and "mean" in metric:
                reported_means[name] = metric["mean"]
    return {
        "run_dir": str(run_dir.resolve()), "run_finished": run_finished,
        "complete": complete, "score_is_provisional": not complete,
        "n_attempts": attempts, "n_expected_tasks": denominator,
        "n_trial_directories": len(trials), "n_missing_trial_directories": missing_dirs,
        "n_scored_trials": solved + failed, "n_solved": solved, "n_failed": failed,
        "n_missing_rewards": missing_rewards, "n_invalid_rewards": invalid_rewards,
        "n_errors": errors, "n_timeouts": sum(trial["timeout"] for trial in trials),
        "n_unfinished_trials": unfinished, "duplicate_tasks": duplicates,
        "pass_at_1": score, "harbor_reported_means": reported_means,
        "warnings": warnings, "trials": trials,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="single timestamped Harbor job directory")
    parser.add_argument("--expected-tasks", type=int, help="full dataset size; missing trials remain in the denominator")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = summarize_run(args.run_dir, args.expected_tasks)
    except (ValueError, OSError) as exc:
        print(f"Cannot summarize run: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        score = summary["pass_at_1"]
        label = "provisional pass@1" if summary["score_is_provisional"] else "pass@1"
        print(f"{label}: {score:.6f} ({summary['n_solved']}/{summary['n_expected_tasks']})" if score is not None else "pass@1: ineligible run (multiple attempts or duplicate trials)")
        print(f"Scored: {summary['n_scored_trials']}; errors: {summary['n_errors']}; timeouts: {summary['n_timeouts']}; missing rewards: {summary['n_missing_rewards']}; invalid rewards: {summary['n_invalid_rewards']}; missing trials: {summary['n_missing_trial_directories']}")
        print(f"Run finished: {summary['run_finished']}; complete for comparison: {summary['complete']}")
        for warning in summary["warnings"]:
            print(f"WARNING: {warning}")
    return 1 if args.fail_on_incomplete and not summary["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
