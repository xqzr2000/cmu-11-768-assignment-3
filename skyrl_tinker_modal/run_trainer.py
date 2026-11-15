"""Run the trainer itself on Modal, detached from the Codespace (ours, Day 4).

The course setup runs scripts/train_async.py on your own machine and only the
GPU server on Modal. That ties a 6-8 hour run to the machine staying awake.
This app runs the same script, unchanged, in a CPU-only Modal container next
to the GPU server, so the Codespace can be stopped once the run is launched.

    Codespace                 Modal
    ---------                 -----
    launch_on_modal.py  --->  scottyrl-trainer   (this file, CPU only)
    (then stop it)               scripts/train_async.py
                                   | Tinker API over HTTPS
                                   v
                              scottyrl-textcraft (GPU server, unchanged)

Run logs go to the `scottyrl-trainer-runs` volume, in the same layout as the
repo's runs/ directory, plus <run_name>.console.log with the terminal output.

Deploy once (from the repo root), then launch with scripts/launch_on_modal.py:

    uv run modal deploy skyrl_tinker_modal/run_trainer.py
    uv run python scripts/launch_on_modal.py --config configs/p5_textcraft_grpo.yaml

The same app also runs the Part 5 evaluation (`evaluate`, added Day 5):
scripts/evaluate_checkpoint.py for each model in turn, against the
`scottyrl-evaluation` server, launched with scripts/launch_eval_on_modal.py.

Watch:     uv run modal app logs -f scottyrl-trainer   (Ctrl-C stops watching only)
Download:  uv run modal volume get scottyrl-trainer-runs runs/<file> runs/
Emergency stop: uv run modal app stop scottyrl-trainer
"""

from __future__ import annotations

import threading
from pathlib import Path

import modal

APP_NAME = "scottyrl-trainer"
VOLUME_NAME = "scottyrl-trainer-runs"
REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/repo"
VOL = "/vol"  # working directory of the trainer, so runs/ lands on the volume

# Hard limit for one launch, retries included. The GPU server's own container
# limit is also 12 hours (skyrl_server.server_options).
TIME_CAP_S = 12 * 60 * 60
# Time the trainer gets after SIGINT to write its checkpoint manifest.
GRACE_S = 5 * 60
# No output for this long means the trainer is hung: interrupt and retry. The
# longest normal silence is the step-0 evaluation, about 15-30 minutes on TextCraft.
STALL_S = 60 * 60

# The client side needs no GPU and no PyTorch: the same locked dependencies as
# the Codespace, then the repo's code, configs and data on top.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(REPO), extras=["modal"])
    .env({"PYTHONPATH": REMOTE_REPO, "PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf"})
    .add_local_dir(REPO / "scottyrl", f"{REMOTE_REPO}/scottyrl", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "scripts", f"{REMOTE_REPO}/scripts", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "configs", f"{REMOTE_REPO}/configs")
    .add_local_dir(REPO / "data", f"{REMOTE_REPO}/data")
)

runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOL: runs_volume},
    cpu=4,
    memory=16384,
    timeout=TIME_CAP_S + GRACE_S + 10 * 60,
    nonpreemptible=True,  # a preempted container would kill the run with it
    retries=0,  # retries happen inside, where they can resume
)
def train(
    config: str,
    server_url: str,
    overrides: list[str] | None = None,
    resume: bool = False,
    max_retries: int = 2,
    time_cap_s: int = TIME_CAP_S,
) -> int:
    from scottyrl.config import load_config
    from scottyrl.modal_runner import (
        child_env,
        run_subprocess,
        run_with_retries,
        trainer_command,
    )

    overrides = list(overrides or [])
    config_path = f"{REMOTE_REPO}/{config}"
    run_name = load_config(config_path, overrides).train.run_name
    runs_dir = Path(VOL) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    console = runs_dir / f"{run_name}.console.log"
    env = child_env(TINKER_BASE_URL=server_url)

    # Volume writes become visible outside the container on commit. Commit
    # every minute so the logs can be read while the run is going.
    stop = threading.Event()

    def commit_loop() -> None:
        while not stop.wait(60):
            try:
                runs_volume.commit()
            except Exception as exc:  # a failed commit must not kill the run
                print(f"[runner] volume commit failed: {exc}", flush=True)

    threading.Thread(target=commit_loop, daemon=True).start()

    def log(msg: str) -> None:
        print(msg, flush=True)
        with console.open("a") as f:
            f.write(msg + "\n")

    try:
        code = run_with_retries(
            lambda cmd, time_left: run_subprocess(
                cmd, cwd=Path(VOL), env=env, console_path=console,
                time_left_s=time_left, grace_s=GRACE_S, stall_s=STALL_S,
            ),
            runs_dir=runs_dir,
            run_name=run_name,
            make_command=lambda resume_now: trainer_command(
                "python", f"{REMOTE_REPO}/scripts/train_async.py", config_path,
                overrides, resume=resume_now,
            ),
            allow_existing=resume,
            max_retries=max_retries,
            time_cap_s=time_cap_s,
            log=log,
        )
    finally:
        stop.set()
        runs_volume.commit()
    log(f"[runner] finished {run_name} with exit code {code}")
    runs_volume.commit()
    return code


# One evaluation prints a progress line every 30 seconds, so ten silent
# minutes means it is hung.
EVAL_STALL_S = 10 * 60


def model_slug(model: str) -> str:
    """A file-name-safe form of a served model name (Qwen/Qwen3.5-4B -> Qwen_Qwen3.5-4B)."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in model)


@app.function(
    image=image,
    volumes={VOL: runs_volume},
    cpu=2,  # the Part 5 trainer showed 4 cores and 16 GiB were more than needed
    memory=8192,
    timeout=TIME_CAP_S + GRACE_S + 10 * 60,
    nonpreemptible=True,
    retries=0,
)
def evaluate(
    config: str,
    server_url: str,
    models: list[str],
    tag: str = "p5_eval",
    k: int = 4,
    num_tasks: int = 50,
    concurrency: int = 32,
    min_rps: float | None = None,
    overrides: list[str] | None = None,
    max_retries: int = 2,
    time_cap_s: int = 3 * 60 * 60,
) -> int:
    """Evaluate each of `models` on the server at `server_url`, one after another.

    Files on the volume, per model: runs/<tag>_<model>.jsonl (one line per
    finished episode; a relaunch continues from it) and runs/<tag>_<model>.json
    (the final numbers). runs/<tag>.console.log has the terminal output.

    A crashed or hung evaluation is retried up to `max_retries` times and
    picks up from its saved episodes. One stopped by its own speed check
    (exit code 3) is not retried and the remaining models are not started:
    a slow server would only bill for the same slow run again.
    """
    import time

    from scottyrl.eval_runner import EXIT_TOO_SLOW
    from scottyrl.modal_runner import child_env, run_subprocess

    runs_dir = Path(VOL) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    console = runs_dir / f"{tag}.console.log"
    env = child_env()
    stop = threading.Event()

    def commit_loop() -> None:
        while not stop.wait(60):
            try:
                runs_volume.commit()
            except Exception as exc:
                print(f"[runner] volume commit failed: {exc}", flush=True)

    threading.Thread(target=commit_loop, daemon=True).start()

    def log(msg: str) -> None:
        print(msg, flush=True)
        with console.open("a") as f:
            f.write(msg + "\n")

    start = time.monotonic()
    code = 0
    try:
        for model in models:
            stem = runs_dir / f"{tag}_{model_slug(model)}"
            cmd = [
                "python", f"{REMOTE_REPO}/scripts/evaluate_checkpoint.py",
                "--config", f"{REMOTE_REPO}/{config}",
                "--server", server_url, "--model", model,
                "--k", str(k), "--num-tasks", str(num_tasks),
                "--concurrency", str(concurrency),
                "--out", f"{stem}.jsonl", "--summary", f"{stem}.json",
            ]
            if min_rps is not None:
                cmd += ["--min-rps", str(min_rps)]
            for override in overrides or []:
                cmd += ["--set", override]
            shown = " ".join("<server>" if part == server_url else part for part in cmd)
            for attempt in range(max_retries + 1):
                time_left = time_cap_s - (time.monotonic() - start)
                if time_left <= 0:
                    log("[runner] time cap reached; relaunch to continue from the saved episodes")
                    return code or 1
                log(f"[runner] {model}: attempt {attempt + 1} of {max_retries + 1}: {shown}")
                code, timed_out = run_subprocess(
                    cmd, cwd=Path(VOL), env=env, console_path=console,
                    time_left_s=time_left, grace_s=60, stall_s=EVAL_STALL_S,
                )
                log(f"[runner] {model}: attempt {attempt + 1} exited with code {code}")
                if code == 0 or code == EXIT_TOO_SLOW or timed_out:
                    break
            if code != 0:
                reason = "too slow" if code == EXIT_TOO_SLOW else "failed"
                log(f"[runner] {model} {reason}; stopping here. Finished episodes are saved.")
                return code
        log(f"[runner] finished {tag}: all {len(models)} models evaluated")
        return 0
    finally:
        stop.set()
        runs_volume.commit()
