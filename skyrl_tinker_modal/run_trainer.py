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
                time_left_s=time_left, grace_s=GRACE_S,
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
