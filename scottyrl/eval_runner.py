"""Held-out evaluation that saves as it goes (ours, Day 5).

`scottyrl.evaluate.evaluate_policy` (given) keeps every result in memory and
reports once, at the end. The first Part 5 evaluation showed what that costs:
it ran at a twelfth of the expected speed for 75 minutes with nothing on the
screen to say so, and stopping it lost every finished episode (see
docs/cloud-walker-Modal.md, Nov 13-14). `evaluate_resumable` runs the same
episodes and computes the same metrics, and adds four things:

- Each episode's result is appended to a JSON-lines file the moment it
  finishes. Starting again with the same file skips what is already there.
- A progress line every `progress_every_s` seconds: episodes done, requests
  per second, request latency, retries by cause.
- A speed check. While every slot is busy, fewer than `min_requests_per_s`
  sampling requests per second over `speed_check_s` seconds raises `TooSlow`.
  Finished episodes are already saved, so a later run continues from them.
- One failed sampling call no longer ends the evaluation. An episode that
  loses its first call, or is abandoned later, is run again
  (`retry_abandoned` extra rounds); only episodes still abandoned after
  that count as failures, and the summary says how many.

Nothing here imports Modal or httpx, so it is tested offline
(tests/test_eval_runner.py).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable

from scottyrl import rollout
from scottyrl.agents.base import Agent
from scottyrl.agents.xml_tool_call import XMLToolCallAgent
from scottyrl.envs.base import Env
from scottyrl.metrics import pass_at_k
from scottyrl.policy import Policy
from scottyrl.rollout import rollout_phase, run_episode
from scottyrl.types import Task

# Exit code scripts/evaluate_checkpoint.py uses for TooSlow, so a caller can
# tell "stopped itself, results kept" from a crash.
EXIT_TOO_SLOW = 3


class TooSlow(RuntimeError):
    """The evaluation was stopped by its speed check."""


def load_results(path: str | Path | None, meta: dict | None = None) -> dict[tuple[str, int], dict]:
    """Finished episodes in a results file, keyed by (task_id, attempt).

    Rows of abandoned episodes are left out, so those episodes run again.
    When `meta` is given (the model name, the number of tokens per turn, ...)
    only rows that carry the same values are returned: a file written for one
    model can never be counted towards another. Unreadable lines (a run
    killed mid-write) are skipped.
    """
    done: dict[tuple[str, int], dict] = {}
    if not path or not Path(path).exists():
        return done
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
            key = (str(row["task_id"]), int(row["attempt"]))
        except (ValueError, KeyError, TypeError):
            continue
        if row.get("abandoned"):
            continue
        if meta and any(row.get(name) != value for name, value in meta.items()):
            continue
        done[key] = row
    return done


def _request_stats(policy) -> dict | None:
    stats = getattr(policy, "stats", None)
    return stats if isinstance(stats, dict) and "requests" in stats else None


async def evaluate_resumable(
    policy: Policy,
    tasks: list[Task],
    env_factory: Callable[[Task], Env],
    *,
    k: int = 1,
    concurrency: int = 32,
    results_path: str | Path | None = None,
    meta: dict | None = None,
    success_threshold: float = 1.0,
    agent_cls: type[Agent] = XMLToolCallAgent,
    retry_abandoned: int = 1,
    progress_every_s: float = 30.0,
    speed_check_s: float = 120.0,
    min_requests_per_s: float | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run k rollouts of every task; return the metrics of `evaluate_policy`.

    The returned dict has `evaluate_policy`'s keys (success_rate, pass@k,
    avg_steps, abandoned_episodes, num_tasks, episodes_per_task) plus
    `resumed_episodes` (taken from `results_path`, not run now),
    `wall_time_s`, and, when the policy counts its requests
    (`CompletionsServerPolicy.stats`), `requests`, `request_retries`,
    `request_errors` and `mean_request_s` for this process.

    `min_requests_per_s=None` disables the speed check; it is also skipped
    for a policy that does not count requests.
    """
    wanted = [(task, attempt) for task in tasks for attempt in range(k)]
    saved = load_results(results_path, meta)
    results = {(t.id, a): saved[(t.id, a)] for t, a in wanted if (t.id, a) in saved}
    resumed = len(results)
    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    stats = _request_stats(policy)
    started = time.monotonic()
    sem = asyncio.Semaphore(concurrency)
    unfinished = 0  # episodes of the current round not yet finished

    def save(row: dict) -> None:
        if results_path:
            with open(results_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    async def one(task: Task, attempt: int) -> None:
        nonlocal unfinished
        async with sem:
            began = time.monotonic()
            row = {"task_id": task.id, "attempt": attempt, **(meta or {})}
            try:
                traj = await run_episode(agent_cls(policy), env_factory(task))
                row.update(
                    reward=traj.reward, num_steps=traj.num_steps,
                    abandoned="stop_reason" in traj.info,
                    stop_reason=traj.info.get("stop_reason"),
                )
            except Exception as e:  # the episode's first sampling call failed
                row.update(
                    reward=0.0, num_steps=0, abandoned=True,
                    stop_reason=f"{type(e).__name__}: {e}"[:300],
                )
            row["seconds"] = round(time.monotonic() - began, 1)
            save(row)
            results[(task.id, attempt)] = row
            unfinished -= 1

    def progress_line() -> str:
        finished = sum(1 for row in results.values() if not row.get("abandoned"))
        line = f"[eval] {finished}/{len(wanted)} episodes, {time.monotonic() - started:.0f} s"
        if stats:
            n = stats["requests"]
            line += f" | {n} requests, {n / max(time.monotonic() - started, 1e-9):.1f}/s"
            if n:
                line += f", mean {stats['latency_s'] / n:.1f} s, max {stats['latency_max_s']:.1f} s"
            line += f" | {stats['retries']} retries {stats['errors'] or ''}".rstrip()
        return line

    async def monitor() -> None:
        # (time, requests answered, whether every slot was busy) per tick.
        ticks: list[tuple[float, int, bool]] = []
        while True:
            await asyncio.sleep(progress_every_s)
            log(progress_line())
            # Transcripts are buffered for a MetricsLogger that an evaluation
            # does not have; drop them so a long run does not hold them all.
            rollout.PENDING_TRAJECTORIES.clear()
            rollout.EPISODE_TIMES.clear()
            if not stats or not min_requests_per_s:
                continue
            now = time.monotonic()
            # Busy: enough episodes are left to fill every slot. The last few
            # episodes of a run are slow in requests per second without
            # anything being wrong, so only busy stretches are judged.
            ticks.append((now, stats["requests"], unfinished >= concurrency))
            # Keep one tick at least `speed_check_s` old as the baseline.
            while len(ticks) > 1 and now - ticks[1][0] >= speed_check_s:
                ticks.pop(0)
            window = ticks
            if now - window[0][0] < speed_check_s or not all(t[2] for t in window):
                continue
            rate = (stats["requests"] - window[0][1]) / (now - window[0][0])
            if rate < min_requests_per_s:
                raise TooSlow(
                    f"{rate:.2f} sampling requests per second over the last "
                    f"{now - window[0][0]:.0f} s with all {concurrency} slots busy; "
                    f"the minimum is {min_requests_per_s:.2f}. Finished episodes are saved"
                    + (f" in {results_path}." if results_path else ".")
                )

    watcher = asyncio.create_task(monitor())
    try:
        for _ in range(retry_abandoned + 1):
            todo = [
                (t, a) for t, a in wanted
                if (t.id, a) not in results or results[(t.id, a)].get("abandoned")
            ]
            if not todo:
                break
            unfinished = len(todo)
            with rollout_phase("eval"):
                episodes = asyncio.gather(*(one(t, a) for t, a in todo))
            done, _ = await asyncio.wait({episodes, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:  # only ends by raising
                episodes.cancel()
                await asyncio.gather(episodes, return_exceptions=True)
                watcher.result()
            episodes.result()
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
    log(progress_line())

    rewards_by_task: dict[str, list[float]] = {}
    total_steps = abandoned = 0
    for task, attempt in wanted:
        row = results[(task.id, attempt)]
        rewards_by_task.setdefault(task.id, []).append(float(row["reward"]))
        total_steps += int(row["num_steps"])
        abandoned += bool(row.get("abandoned"))

    summary = {
        **pass_at_k(list(rewards_by_task.values()), k=k, success_threshold=success_threshold),
        "avg_steps": total_steps / len(wanted),
        # Still abandoned after the retry rounds; they count as failures
        # above, so anything but 0 means the numbers are too low.
        "abandoned_episodes": abandoned,
        "num_tasks": len(tasks),
        "episodes_per_task": k,
        "resumed_episodes": resumed,
        "wall_time_s": round(time.monotonic() - started, 1),
    }
    if stats:
        summary.update(
            requests=stats["requests"],
            request_retries=stats["retries"],
            request_errors=dict(stats["errors"]),
            mean_request_s=round(stats["latency_s"] / stats["requests"], 2) if stats["requests"] else None,
        )
    return summary
