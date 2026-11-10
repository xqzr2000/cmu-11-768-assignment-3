"""Extra Part 1 tests (ours, not from the course): the docstring examples and
the edge cases of advantages.py and datagen.py, written before the code."""

import asyncio

import pytest

from scottyrl.advantages import compute_group_advantages
from scottyrl.agents.xml_tool_call import XMLToolCallAgent
from scottyrl.datagen import (
    extension_suffix,
    stitch_trajectory,
    trajectory_to_training_sequences,
)
from scottyrl.envs.number_search import NumberSearchEnv
from scottyrl.policy import MockPolicy
from scottyrl.rollout import run_episode
from scottyrl.types import LLMCall, Task, Trajectory
from test_envs import ThinkStrippingPolicy, make_ns_task


def traj(calls, reward=1.0):
    return Trajectory(task=Task(goal="g", id="t", max_steps=4), calls=calls,
                      reward=reward, num_steps=len(calls))


# ---------- advantages ----------

def test_reinforce_is_the_raw_reward():
    assert compute_group_advantages([1.0, 0.0, 0.5], method="reinforce") == [1.0, 0.0, 0.5]
    assert compute_group_advantages([0.0], method="reinforce") == [0.0]
    assert compute_group_advantages([1.0, 1.0], method="reinforce") == [1.0, 1.0]


def test_grpo_subtracts_the_group_mean_without_std_normalisation():
    assert compute_group_advantages([1.0, 0.0, 0.0, 0.0]) == [0.75, -0.25, -0.25, -0.25]
    out = compute_group_advantages([1, 0])  # ints in, floats out
    assert out == [0.5, -0.5] and all(isinstance(a, float) for a in out)


def test_grpo_degenerate_group_is_none():
    assert compute_group_advantages([1.0, 1.0, 1.0]) is None
    assert compute_group_advantages([0.0, 0.0]) is None
    assert compute_group_advantages([0.0, 1e-12]) is not None  # no tolerance


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        compute_group_advantages([1.0], method="grpo")
    with pytest.raises(ValueError):
        compute_group_advantages([], method="grpo")
    with pytest.raises(ValueError):
        compute_group_advantages([1.0, 0.0], method="ppo")


# ---------- extension_suffix ----------

def test_extension_suffix_docstring_examples():
    assert extension_suffix([1, 2, 3], [1, 2, 3, 9, 9]) == [9, 9]
    assert extension_suffix([1, 2, 3], [1, 2, 3]) == []
    assert extension_suffix([1, 2, 3], [1, 9, 3, 4]) is None
    assert extension_suffix([1, 2, 3], [1, 2]) is None
    assert extension_suffix([], [1, 2]) == [1, 2]


# ---------- stitch_trajectory ----------

def test_stitch_extending_calls_make_one_sequence():
    seqs = stitch_trajectory(traj([
        LLMCall([10, 11], [20, 21], [-0.2, -0.3]),
        LLMCall([10, 11, 20, 21, 30], [40], [-0.4]),
    ]))
    assert len(seqs) == 1
    s = seqs[0]
    assert s.tokens == [10, 11, 20, 21, 30, 40]
    assert s.mask == [0, 1, 1, 0, 1]
    assert s.logprobs == [0.0, -0.2, -0.3, 0.0, -0.4]
    assert s.advantages == [0.0] * 5


def test_stitch_broken_extension_starts_a_new_sequence():
    seqs = stitch_trajectory(traj([
        LLMCall([10, 11], [20, 21], [-0.2, -0.3]),
        LLMCall([10, 11, 30], [40], [-0.4]),
    ]))
    assert [(s.tokens, s.mask) for s in seqs] == [
        ([10, 11, 20, 21], [0, 1, 1]),
        ([10, 11, 30, 40], [0, 0, 1]),
    ]
    assert seqs[1].logprobs == [0.0, 0.0, -0.4]


def test_stitch_does_not_mutate_the_trajectory():
    calls = [LLMCall([1, 2], [3], [-0.1]), LLMCall([1, 2, 3, 4], [5], [-0.5])]
    stitch_trajectory(traj(calls))
    assert calls[0].prompt_tokens == [1, 2] and calls[1].prompt_tokens == [1, 2, 3, 4]


def test_stitch_drops_sequences_with_nothing_to_train_on():
    assert stitch_trajectory(traj([])) == []
    # empty completion: no action positions
    assert stitch_trajectory(traj([LLMCall([1, 2, 3], [], [])])) == []
    # a single completion token after an empty prompt is token 0: never a target
    assert stitch_trajectory(traj([LLMCall([], [7], [-0.1])])) == []
    # a broken call with an empty completion is dropped, its neighbours are kept
    seqs = stitch_trajectory(traj([
        LLMCall([1], [2], [-0.2]),
        LLMCall([9, 9], [], []),
        LLMCall([5], [6], [-0.6]),
    ]))
    assert [s.tokens for s in seqs] == [[1, 2], [5, 6]]


def test_stitch_empty_prompt_skips_only_the_first_token():
    (s,) = stitch_trajectory(traj([LLMCall([], [7, 8], [-0.7, -0.8])]))
    assert s.tokens == [7, 8] and s.mask == [1] and s.logprobs == [-0.8]


def test_stitch_empty_completion_then_extension_keeps_one_sequence():
    (s,) = stitch_trajectory(traj([
        LLMCall([1, 2], [], []),
        LLMCall([1, 2, 3], [4], [-0.4]),
    ]))
    assert s.tokens == [1, 2, 3, 4] and s.mask == [0, 0, 1]


def test_stitch_rejects_misaligned_logprobs():
    with pytest.raises(ValueError):
        stitch_trajectory(traj([LLMCall([1, 2], [3, 4], [-0.1])]))


# ---------- real episodes through the agent ----------

def _episode(policy, target=7):
    env = NumberSearchEnv(make_ns_task(target=target, max_steps=4))
    return asyncio.run(run_episode(XMLToolCallAgent(policy), env))


def _check_alignment(t, seqs):
    """Every completion token of the episode appears exactly once as a masked
    target, in order, carrying its own sampling logprob."""
    got = [(s.tokens[j + 1], s.logprobs[j]) for s in seqs for j, m in enumerate(s.mask) if m]
    want = [(tok, lp) for c in t.calls for tok, lp in zip(c.completion_tokens, c.completion_logprobs)]
    assert got == want
    for s in seqs:
        assert all(lp == 0.0 for lp, m in zip(s.logprobs, s.mask) if not m)


def test_normal_episode_is_one_sequence():
    t = _episode(MockPolicy(["<tool>guess(2)</tool>", "<tool>guess(9)</tool>", "<tool>guess(7)</tool>"]))
    seqs = stitch_trajectory(t)
    assert len(t.calls) == 3 and len(seqs) == 1
    assert seqs[0].tokens == t.calls[-1].prompt_tokens + t.calls[-1].completion_tokens
    _check_alignment(t, seqs)


def test_think_stripping_episode_is_one_sequence_per_call():
    t = _episode(ThinkStrippingPolicy([
        "<think>\nstart low\n</think>\n\n<tool>guess(2)</tool>",
        "<think>\nhigher\n</think>\n\n<tool>guess(9)</tool>",
        "<think>\nbisect\n</think>\n\n<tool>guess(7)</tool>",
    ]))
    seqs = stitch_trajectory(t)
    assert len(seqs) == len(t.calls) == 3
    for s, c in zip(seqs, t.calls):
        assert s.tokens == c.prompt_tokens + c.completion_tokens
    _check_alignment(t, seqs)


# ---------- trajectory_to_training_sequences ----------

def test_advantage_is_broadcast_to_action_positions_only():
    t = traj([
        LLMCall([10, 11], [20, 21], [-0.2, -0.3]),
        LLMCall([10, 11, 20, 21, 30], [40], [-0.4]),
    ])
    (s,) = trajectory_to_training_sequences(t, advantage=-0.25)
    assert s.advantages == [0.0, -0.25, -0.25, 0.0, -0.25]
    assert s.mask == [0, 1, 1, 0, 1] and s.logprobs == [0.0, -0.2, -0.3, 0.0, -0.4]
    # zero advantage still yields the sequences
    assert len(trajectory_to_training_sequences(t, advantage=0.0)) == 1
