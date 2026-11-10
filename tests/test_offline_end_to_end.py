"""Offline dry run of both trainers (ours): real tasks, real environment, real
agent and data pipeline, with a scripted policy and a fake training service.
It catches wiring mistakes before any GPU time is paid for."""

import asyncio
from pathlib import Path

from scottyrl.config import ModelConfig, TrainConfig
from scottyrl.envs.number_search import NumberSearchEnv
from scottyrl.policy import MockPolicy
from scottyrl.train.async_grpo import AsyncGRPOTrainer
from scottyrl.train.sync import SyncTrainer
from scottyrl.types import load_tasks

DATA = Path(__file__).resolve().parents[1] / "data"


class FakeSession:
    tokenizer = None

    def __init__(self):
        self.snapshots, self.updates = [], []

    async def new_sampling_client(self, name):
        self.snapshots.append(name)
        return name

    async def train_step(self, seqs, loss_fn, lr, grad_clip_norm):
        await asyncio.sleep(0)
        self.updates.append((len(seqs), loss_fn, lr, grad_clip_norm))
        return {"loss": 0.0}


class FakeLogger:
    def __init__(self):
        self.records = []

    def log(self, step, metrics):
        self.records.append((step, metrics))


class HalfRightPolicy(MockPolicy):
    """Guesses 228 or 563: solves some tasks and fails others, so GRPO groups
    are a mix of informative and degenerate."""

    counter = 0

    def __init__(self):
        HalfRightPolicy.counter += 1
        guess = 228 if HalfRightPolicy.counter % 2 else 563
        super().__init__([f"<tool>guess({guess})</tool>"])


def build(cls, **overrides):
    values = dict(group_size=4, tasks_per_batch=4, groups_per_batch=4, num_workers=3,
                  queue_capacity=4, num_steps=3, eval_every=1, eval_tasks=3, eval_k=2)
    values.update(overrides)
    session, logger = FakeSession(), FakeLogger()
    tasks = load_tasks(DATA / "number_search_train.jsonl")[:6]
    trainer = cls(session, lambda t: NumberSearchEnv(t), tasks, tasks,
                  ModelConfig(), TrainConfig(**values), logger)
    trainer.make_policy = lambda client: client
    # a fresh scripted policy per episode, so rollouts in a group differ
    trainer.agent_cls = lambda: (lambda policy: _agent(HalfRightPolicy()))
    return trainer, session, logger


def _agent(policy):
    from scottyrl.agents.xml_tool_call import XMLToolCallAgent

    return XMLToolCallAgent(policy)


def keys(logger, prefix):
    return [step for step, m in logger.records if any(k.startswith(prefix) for k in m)]


def test_sync_trainer_runs_end_to_end_offline():
    for algo, loss, group in (("grpo", "ppo", 4), ("reinforce", "importance_sampling", 1)):
        trainer, session, logger = build(SyncTrainer, algo=algo, loss_fn=loss, group_size=group)
        asyncio.run(asyncio.wait_for(trainer.train(), timeout=20))
        assert keys(logger, "train/") == [1, 2, 3]
        assert keys(logger, "eval/") == [1, 2, 3]
        assert session.updates and all(u[1] == loss for u in session.updates)
        assert len(session.snapshots) == 1 + len(session.updates)


def test_async_trainer_runs_end_to_end_offline():
    trainer, session, logger = build(AsyncGRPOTrainer)
    stats = asyncio.run(asyncio.wait_for(trainer.train(), timeout=20))
    assert stats["train_steps"] == 3 and stats["groups_consumed"] == 12
    assert keys(logger, "train/") == [1, 2, 3]
    assert sorted(keys(logger, "eval/")) == [1, 2, 3]
    assert session.snapshots == [f"v{i}" for i in range(len(session.updates) + 1)]
