"""Tests for the detached trainer runner (ours, Day 4).

The runner runs scripts/train_async.py inside a Modal container so a long run
does not depend on the Codespace staying up. Only the orchestration is tested
here: deciding between a fresh start and a resume, the bounded retry loop, and
the time cap. No Modal calls are made.
"""

import json
import sys
from pathlib import Path

import pytest

from scottyrl.modal_runner import (
    prepare_attempt,
    run_subprocess,
    run_with_retries,
    trainer_command,
)

RUN = "p5_textcraft_grpo"


def write_marker(runs: Path, step: int = 10) -> None:
    (runs / f"{RUN}.resume.json").write_text(
        json.dumps({"step": step, "optimizer_steps": step, "state_path": "tinker://m/weights/r"})
    )


def write_partial(runs: Path) -> None:
    (runs / f"{RUN}.jsonl").write_text('{"step": 0}\n')
    (runs / f"{RUN}.checkpoints.json").write_text("{}")
    traj = runs / f"{RUN}_trajectories"
    traj.mkdir()
    (traj / "a.jsonl").write_text("{}")


# --- prepare_attempt: fresh, resume, or refuse --------------------------------


def test_empty_runs_dir_starts_fresh(tmp_path):
    assert prepare_attempt(tmp_path, RUN, allow_existing=False, tag="t") == "fresh"


def test_existing_results_refused_unless_allowed(tmp_path):
    write_partial(tmp_path)
    with pytest.raises(SystemExit):
        prepare_attempt(tmp_path, RUN, allow_existing=False, tag="t")


def test_marker_means_resume_and_keeps_files(tmp_path):
    write_partial(tmp_path)
    write_marker(tmp_path)
    assert prepare_attempt(tmp_path, RUN, allow_existing=True, tag="t") == "resume"
    assert (tmp_path / f"{RUN}.jsonl").exists()


def test_partial_run_without_marker_is_archived_then_fresh(tmp_path):
    # A run that died before its first resumable checkpoint (step 10) cannot
    # resume; its files would block a fresh start, so they are moved aside.
    write_partial(tmp_path)
    assert prepare_attempt(tmp_path, RUN, allow_existing=True, tag="t") == "fresh"
    assert not (tmp_path / f"{RUN}.jsonl").exists()
    archived = tmp_path / "aborted" / f"{RUN}-t"
    assert (archived / f"{RUN}.jsonl").exists()
    assert (archived / f"{RUN}_trajectories" / "a.jsonl").exists()


# --- trainer_command ------------------------------------------------------------


def test_trainer_command_adds_resume_only_when_resuming():
    base = trainer_command("python", "s.py", "c.yaml", ["train.num_steps=2"], resume=False)
    assert base == ["python", "s.py", "--config", "c.yaml", "--set", "train.num_steps=2"]
    resumed = trainer_command("python", "s.py", "c.yaml", ["train.num_steps=2"], resume=True)
    assert resumed[-2:] == ["--set", "train.resume=true"]


# --- run_with_retries -------------------------------------------------------------


class FakeRunner:
    """Plays back exit codes; on a crash after step 10 it leaves a marker."""

    def __init__(self, runs, outcomes):
        self.runs, self.outcomes, self.commands = runs, list(outcomes), []

    def __call__(self, cmd, time_left):
        self.commands.append(cmd)
        code, timed_out, leave_marker = self.outcomes.pop(0)
        (self.runs / f"{RUN}.jsonl").write_text('{"step": 0}\n')
        if leave_marker:
            write_marker(self.runs)
        return code, timed_out


def retry(runs, runner, max_retries=2, time_cap_s=1000.0):
    return run_with_retries(
        runner,
        runs_dir=runs,
        run_name=RUN,
        make_command=lambda resume: ["train"] + (["RESUME"] if resume else []),
        allow_existing=False,
        max_retries=max_retries,
        time_cap_s=time_cap_s,
        clock=lambda: 0.0,
        log=lambda msg: None,
    )


def test_success_first_time_runs_once(tmp_path):
    runner = FakeRunner(tmp_path, [(0, False, True)])
    assert retry(tmp_path, runner) == 0
    assert runner.commands == [["train"]]


def test_crash_after_checkpoint_resumes(tmp_path):
    runner = FakeRunner(tmp_path, [(1, False, True), (0, False, True)])
    assert retry(tmp_path, runner) == 0
    assert runner.commands == [["train"], ["train", "RESUME"]]


def test_crash_before_checkpoint_restarts_fresh(tmp_path):
    runner = FakeRunner(tmp_path, [(1, False, False), (0, False, True)])
    assert retry(tmp_path, runner) == 0
    assert runner.commands == [["train"], ["train"]]
    assert (tmp_path / "aborted").exists()


def test_retries_are_bounded(tmp_path):
    runner = FakeRunner(tmp_path, [(1, False, True)] * 5)
    assert retry(tmp_path, runner, max_retries=2) == 1
    assert len(runner.commands) == 3


def test_time_cap_is_not_retried(tmp_path):
    runner = FakeRunner(tmp_path, [(130, True, True), (0, False, True)])
    assert retry(tmp_path, runner) == 130
    assert len(runner.commands) == 1


# --- run_subprocess: tee and time cap ----------------------------------------------


def test_run_subprocess_tees_output(tmp_path):
    console = tmp_path / "console.log"
    code, timed_out = run_subprocess(
        [sys.executable, "-c", "print('hello')"], cwd=tmp_path, env=None,
        console_path=console, time_left_s=30, grace_s=5,
    )
    assert (code, timed_out) == (0, False)
    assert "hello" in console.read_text()


def test_run_subprocess_interrupts_at_time_cap(tmp_path):
    # The child catches SIGINT and writes a marker file, standing in for the
    # trainer's `finally` block that writes the checkpoint manifest.
    script = (
        "import time, pathlib\n"
        "try:\n"
        "    print('started', flush=True)\n"
        "    time.sleep(30)\n"
        "except KeyboardInterrupt:\n"
        "    pathlib.Path('cleaned_up').write_text('yes')\n"
        "    raise SystemExit(130)\n"
    )
    code, timed_out = run_subprocess(
        [sys.executable, "-c", script], cwd=tmp_path, env=None,
        console_path=tmp_path / "console.log", time_left_s=1.0, grace_s=10,
    )
    assert timed_out
    assert (tmp_path / "cleaned_up").read_text() == "yes"


# --- run_subprocess: stall watchdog -------------------------------------------------


def test_run_subprocess_interrupts_a_silent_process(tmp_path):
    # A hang is the costly failure: both containers stay up and bill until the
    # time cap. No output for stall_s counts as a crash (retried), not as the
    # time cap (not retried).
    script = (
        "import time\n"
        "print('started', flush=True)\n"
        "time.sleep(30)\n"
    )
    console = tmp_path / "console.log"
    code, timed_out = run_subprocess(
        [sys.executable, "-c", script], cwd=tmp_path, env=None,
        console_path=console, time_left_s=60, grace_s=5, stall_s=1.0, poll_s=0.1,
    )
    assert not timed_out
    assert code != 0
    assert "no output for" in console.read_text()


def test_steady_output_is_not_a_stall(tmp_path):
    script = (
        "import time\n"
        "for _ in range(6):\n"
        "    print('step', flush=True)\n"
        "    time.sleep(0.3)\n"
    )
    code, timed_out = run_subprocess(
        [sys.executable, "-c", script], cwd=tmp_path, env=None,
        console_path=tmp_path / "console.log", time_left_s=60, grace_s=5,
        stall_s=1.0, poll_s=0.1,
    )
    assert (code, timed_out) == (0, False)


def test_stall_is_retried(tmp_path):
    # run_with_retries sees a stall as an ordinary failure: (nonzero, False).
    runner = FakeRunner(tmp_path, [(124, False, True), (0, False, True)])
    assert retry(tmp_path, runner) == 0
    assert runner.commands == [["train"], ["train", "RESUME"]]
