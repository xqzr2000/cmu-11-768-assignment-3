"""Run a training script unattended, with bounded retries and a time cap (ours).

Used by skyrl_tinker_modal/run_trainer.py, which runs scripts/train_async.py
inside a CPU-only Modal container so the Part 5 run does not depend on the
Codespace staying up. Nothing here imports Modal, so it is tested offline
(tests/test_modal_runner.py).

The three decisions this module makes:

- Fresh start or resume. A run with a resume marker (written every
  `resume_save_every` optimizer steps) resumes from it. A run that died before
  its first marker cannot resume, and its partial log would make
  `check_run_name_available` refuse a fresh start, so those files are moved
  to runs/aborted/<run_name>-<tag>/ and the run starts over.
- Retry or stop. A failed attempt is retried at most `max_retries` times. An
  attempt stopped by the time cap is not retried.
- When to stop. The trainer gets SIGINT when the time cap is reached, so its
  `finally` block still writes the checkpoint manifest; if it has not exited
  after `grace_s` seconds it is killed.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

# Files a run writes under runs/ (see scottyrl.config and scottyrl.metrics).
def _run_files(runs_dir: Path, run_name: str) -> dict[str, Path]:
    return {
        "log": runs_dir / f"{run_name}.jsonl",
        "marker": runs_dir / f"{run_name}.resume.json",
        "manifest": runs_dir / f"{run_name}.checkpoints.json",
        "trajectories": runs_dir / f"{run_name}_trajectories",
    }


def _has_usable_marker(marker: Path) -> bool:
    try:
        return bool(json.loads(marker.read_text()).get("state_path"))
    except (OSError, ValueError):
        return False


def prepare_attempt(runs_dir: Path, run_name: str, *, allow_existing: bool, tag: str) -> str:
    """Return "fresh" or "resume" for the next attempt of `run_name`.

    allow_existing=False is the first attempt of a newly launched run: any
    earlier results for this run_name are refused, as the trainer itself would.
    """
    runs_dir = Path(runs_dir)
    files = _run_files(runs_dir, run_name)
    present = [p for p in files.values() if p.exists()]
    if not present:
        return "fresh"
    if not allow_existing:
        raise SystemExit(
            f"runs/ on the volume already has results for {run_name!r}: "
            f"{', '.join(p.name for p in present)}. Launch with --resume to continue it, "
            "or use a new run_name."
        )
    if _has_usable_marker(files["marker"]):
        return "resume"
    archive = runs_dir / "aborted" / f"{run_name}-{tag}"
    archive.mkdir(parents=True, exist_ok=True)
    for path in present:
        shutil.move(str(path), str(archive / path.name))
    return "fresh"


def trainer_command(
    python: str, script: str, config: str, overrides: list[str], *, resume: bool
) -> list[str]:
    cmd = [python, script, "--config", config]
    for override in overrides:
        cmd += ["--set", override]
    if resume:
        cmd += ["--set", "train.resume=true"]
    return cmd


def run_with_retries(
    run_attempt: Callable[[list[str], float], tuple[int, bool]],
    *,
    runs_dir: Path,
    run_name: str,
    make_command: Callable[[bool], list[str]],
    allow_existing: bool,
    max_retries: int,
    time_cap_s: float,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> int:
    """Run attempts until one succeeds, retries run out, or time runs out.

    `run_attempt(cmd, time_left_s)` returns (exit_code, stopped_by_time_cap).
    Returns the last exit code.
    """
    start = clock()
    code = 1
    for attempt in range(max_retries + 1):
        time_left = time_cap_s - (clock() - start)
        if time_left <= 0:
            log("[runner] time cap reached; not starting another attempt")
            return code
        mode = prepare_attempt(
            runs_dir, run_name,
            allow_existing=allow_existing or attempt > 0,
            tag=time.strftime("%Y%m%dT%H%M%S"),
        )
        cmd = make_command(mode == "resume")
        log(f"[runner] attempt {attempt + 1} of {max_retries + 1} ({mode}): {' '.join(cmd)}")
        code, timed_out = run_attempt(cmd, time_left)
        log(f"[runner] attempt {attempt + 1} exited with code {code}")
        if code == 0:
            return 0
        if timed_out:
            log("[runner] stopped by the time cap; resume by hand with --resume")
            return code
    log("[runner] no retries left")
    return code


def run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    console_path: Path,
    time_left_s: float,
    grace_s: float = 300.0,
) -> tuple[int, bool]:
    """Run `cmd`, copying its output to stdout and to `console_path`.

    At `time_left_s` the process gets SIGINT; after a further `grace_s` it is
    killed. Returns (exit_code, stopped_by_time_cap).
    """
    console_path = Path(console_path)
    console_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    def pump() -> None:
        with console_path.open("a") as console:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                console.write(line)
                console.flush()

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=max(time_left_s, 0))
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    reader.join(timeout=10)
    return proc.returncode, timed_out


def child_env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(extra)
    return env
