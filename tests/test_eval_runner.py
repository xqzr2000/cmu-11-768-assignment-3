"""Tests for the resumable evaluation and the sampling client's retries (ours, Day 5).

Both exist because the first Part 5 evaluation stalled: a few percent of
requests were lost on the way to the server, each held its episode for the
client's 10-minute timeout, and nothing was saved when the run was stopped.
No GPU, no Modal and no network are used here.
"""

import asyncio
import json

import httpx
import pytest

from scottyrl.envs import make_env_factory
from scottyrl.eval_runner import TooSlow, evaluate_resumable, load_results
from scottyrl.evaluate import evaluate_policy
from scottyrl.policy import CompletionsServerPolicy, Completion, MockPolicy
from scottyrl.types import Task

ENV = make_env_factory("number_search")


def tasks(n: int = 3) -> list[Task]:
    # Target 7 in 1..10: "guess(7)" solves it on the first turn.
    return [
        Task(goal="Guess the number.", id=f"t{i}", max_steps=4,
             info={"low": 1, "high": 10, "target": 7})
        for i in range(n)
    ]


class CountingPolicy(MockPolicy):
    """MockPolicy that answers `text` every time and counts its calls.

    `fail_calls` are the call numbers (1-based) that raise; `delay` is the
    time one call takes; `stats` mimics CompletionsServerPolicy.stats.
    """

    def __init__(self, text="<tool>guess(7)</tool>", fail_calls=(), delay=0.0):
        super().__init__([text])
        self.calls = 0
        self.fail_calls = set(fail_calls)
        self.delay = delay
        self.stats = {"requests": 0, "retries": 0, "failed": 0,
                      "latency_s": 0.0, "latency_max_s": 0.0, "errors": {}}

    async def complete(self, prompt_tokens):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.calls in self.fail_calls:
            raise httpx.ConnectError("dropped")
        self.stats["requests"] += 1
        return await super().complete(prompt_tokens)


def run(policy, **kwargs):
    kwargs.setdefault("log", lambda line: None)
    return asyncio.run(evaluate_resumable(policy, tasks(), ENV, **kwargs))


# --- same numbers as the given evaluation ------------------------------------

@pytest.mark.parametrize("text", ["<tool>guess(7)</tool>", "<tool>guess(3)</tool>"])
def test_metrics_match_evaluate_policy(text):
    ours = run(CountingPolicy(text), k=2)
    given = asyncio.run(evaluate_policy(MockPolicy([text]), tasks(), ENV, k=2))
    for key, value in given.items():
        assert ours[key] == value, key


# --- saving and resuming -----------------------------------------------------

def test_each_episode_is_saved_and_a_rerun_skips_them(tmp_path):
    out = tmp_path / "eval.jsonl"
    first = CountingPolicy()
    summary = run(first, k=2, results_path=out, meta={"model": "a3"})
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 6 and first.calls == 6
    assert {(r["task_id"], r["attempt"]) for r in rows} == {(f"t{i}", a) for i in range(3) for a in range(2)}
    assert all(r["model"] == "a3" and r["reward"] == 1.0 for r in rows)
    assert summary["success_rate"] == 1.0 and summary["resumed_episodes"] == 0

    second = CountingPolicy()
    again = run(second, k=2, results_path=out, meta={"model": "a3"})
    assert second.calls == 0
    assert again["resumed_episodes"] == 6 and again["success_rate"] == 1.0


def test_a_partial_file_is_completed_not_restarted(tmp_path):
    out = tmp_path / "eval.jsonl"
    out.write_text(
        json.dumps({"task_id": "t0", "attempt": 0, "model": "a3", "reward": 0.0,
                    "num_steps": 4, "abandoned": False}) + "\n"
        + '{"task_id": "t1", "attem'  # a line cut off when the run was killed
    )
    policy = CountingPolicy()
    summary = run(policy, k=1, results_path=out, meta={"model": "a3"})
    assert policy.calls == 2  # t1 and t2 only
    assert summary["resumed_episodes"] == 1
    assert summary["success_rate"] == pytest.approx(2 / 3)  # the saved failure is kept


def test_rows_of_another_model_are_not_reused(tmp_path):
    out = tmp_path / "eval.jsonl"
    run(CountingPolicy(), k=1, results_path=out, meta={"model": "a3"})
    assert load_results(out, {"model": "base"}) == {}
    base = CountingPolicy("<tool>guess(3)</tool>")
    summary = run(base, k=1, results_path=out, meta={"model": "base"})
    assert base.calls == 3 * 4 and summary["success_rate"] == 0.0


# --- a failed sampling call does not end the evaluation ----------------------

def test_an_episode_that_loses_its_first_call_is_run_again(tmp_path):
    out = tmp_path / "eval.jsonl"
    policy = CountingPolicy(fail_calls={1})
    summary = run(policy, k=1, concurrency=1, results_path=out)
    assert summary["abandoned_episodes"] == 0 and summary["success_rate"] == 1.0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert sum(r["abandoned"] for r in rows) == 1  # recorded, then redone
    assert len(load_results(out)) == 3


def test_an_episode_that_keeps_failing_counts_as_abandoned():
    policy = CountingPolicy(fail_calls={1, 4})  # t0 fails in both rounds
    summary = run(policy, k=1, concurrency=1, retry_abandoned=1)
    assert summary["abandoned_episodes"] == 1
    assert summary["success_rate"] == pytest.approx(2 / 3)


# --- the speed check ---------------------------------------------------------

def test_a_slow_server_stops_the_run_and_keeps_what_finished(tmp_path):
    out = tmp_path / "eval.jsonl"
    # 2 slots, 0.2 s per request: 10 requests per second at best, 100 asked for.
    policy = CountingPolicy("<tool>guess(3)</tool>", delay=0.2)
    with pytest.raises(TooSlow):
        asyncio.run(evaluate_resumable(
            policy, tasks(40), ENV, k=1, concurrency=2, results_path=out,
            progress_every_s=0.05, speed_check_s=0.3, min_requests_per_s=100,
            log=lambda line: None,
        ))
    assert policy.calls < 40  # stopped early, long before 160 requests


def test_a_fast_enough_server_is_left_alone():
    policy = CountingPolicy(delay=0.01)
    summary = asyncio.run(evaluate_resumable(
        policy, tasks(30), ENV, k=1, concurrency=2,
        progress_every_s=0.02, speed_check_s=0.1, min_requests_per_s=5,
        log=lambda line: None,
    ))
    assert summary["success_rate"] == 1.0 and summary["requests"] == 30


# --- CompletionsServerPolicy: timeouts and retries ---------------------------

class StubTokenizer:
    unk_token_id = None
    eos_token = "<|im_end|>"

    def convert_tokens_to_ids(self, name):
        return {"<|im_end|>": 1, "<|endoftext|>": 2}.get(name)

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, tokens, skip_special_tokens=False):
        return "<|im_end|>"


def server_policy(handler, monkeypatch, **kwargs) -> CompletionsServerPolicy:
    policy = CompletionsServerPolicy(
        "http://server/v1", "a3", StubTokenizer(), max_tokens=8, temperature=1.0,
        chat_template="x", **kwargs,
    )
    policy._client = httpx.AsyncClient(
        base_url=policy.base_url, transport=httpx.MockTransport(handler)
    )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("scottyrl.policy.asyncio.sleep", no_sleep)
    return policy


OK = {"choices": [{"text": "<tool>guess(7)</tool><|im_end|>", "finish_reason": "stop"}]}


def test_default_timeouts_are_short():
    policy = CompletionsServerPolicy(
        "http://server/v1", "a3", StubTokenizer(), max_tokens=8, temperature=1.0,
        chat_template="x",
    )
    timeout = policy._http().timeout
    assert timeout.read == 120.0 and timeout.connect == 10.0
    assert timeout.read < 150  # Modal ends a web request at 150 s


@pytest.mark.parametrize("failure", [
    httpx.ConnectError("refused"),
    httpx.ReadTimeout("slow"),
    httpx.RemoteProtocolError("Server disconnected without sending a response."),
    503,
])
def test_dropped_connections_timeouts_and_5xx_are_retried(failure, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) <= 2:
            if isinstance(failure, int):
                return httpx.Response(failure)
            raise failure
        return httpx.Response(200, json=OK)

    policy = server_policy(handler, monkeypatch)
    completion = asyncio.run(policy.complete([1, 2, 3]))
    assert isinstance(completion, Completion) and completion.text.endswith("<|im_end|>")
    assert len(seen) == 3
    assert policy.stats["requests"] == 1 and policy.stats["retries"] == 2
    assert sum(policy.stats["errors"].values()) == 2


def test_gives_up_after_max_retries(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        raise httpx.ConnectError("refused")

    policy = server_policy(handler, monkeypatch, max_retries=2)
    with pytest.raises(httpx.ConnectError):
        asyncio.run(policy.complete([1]))
    assert len(seen) == 3 and policy.stats["failed"] == 1


def test_a_client_error_is_not_retried(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(400, json={"error": "bad request"})

    policy = server_policy(handler, monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(policy.complete([1]))
    assert len(seen) == 1
