"""Trajectory -> training data. *** STUDENT IMPLEMENTATION FILE ***

The heart of the data pipeline: turning a multi-turn episode into stitched
token sequences with correct masking, logprob alignment, and advantage
broadcasting. Read the TokenSequence docstring in `scottyrl/types.py`
carefully before starting, especially the off-by-one: position j of
mask/logprobs/advantages describes the *target* token tokens[j+1].

Implement the three functions below. You may add private helpers in this
file, but these signatures must not change.
"""

from __future__ import annotations

from scottyrl.types import TokenSequence, Trajectory


def extension_suffix(accumulated: list[int], prompt_tokens: list[int]) -> list[int] | None:
    """Check the sequence-extension property between an accumulated token
    sequence and the next call's prompt.

    Contract: if `accumulated` is a (non-strict) prefix of `prompt_tokens`,
    return the remaining suffix of `prompt_tokens` (the tokens of the newly
    rendered tail: turn markers, environment observation, generation prompt;
    may be empty). Otherwise return None: the extension is broken and the
    caller must start a new sequence.

    Examples:
        extension_suffix([1, 2, 3], [1, 2, 3, 9, 9]) -> [9, 9]
        extension_suffix([1, 2, 3], [1, 2, 3])       -> []
        extension_suffix([1, 2, 3], [1, 9, 3, 4])    -> None
        extension_suffix([1, 2, 3], [1, 2])          -> None
        extension_suffix([], [1, 2])                 -> [1, 2]  (an empty sequence is a prefix of anything)
    """
    n = len(accumulated)
    if len(prompt_tokens) < n or prompt_tokens[:n] != accumulated:
        return None
    return list(prompt_tokens[n:])


def stitch_trajectory(traj: Trajectory) -> list[TokenSequence]:
    """Stitch an episode's LLM calls into training sequences.

    Output contract:
      - Walk the calls in order, maintaining one growing token sequence of
        context tokens (prompt + rendered env observations) interleaved with
        completion tokens. A call whose prompt extends the sequence built so
        far continues it: only the new context suffix is appended before the
        call's completion. A call that breaks the extension property ends the
        current sequence and starts a new one from that call's full prompt.
      - In every emitted TokenSequence: mask is 1 at exactly the target
        positions whose token came from a completion; logprobs carries the
        sampling logprob at those positions and 0.0 everywhere else;
        advantages is all zeros at this stage.
      - Never emit a sequence with fewer than 2 tokens or with no action
        (mask == 1) positions; there would be nothing to train on.

    Example -- two calls whose prompts extend:
        call 0: prompt=[10, 11], completion=[20, 21], logprobs=[-0.2, -0.3]
        call 1: prompt=[10, 11, 20, 21, 30], completion=[40], logprobs=[-0.4]

    Before call 1, the accumulated sequence is [10, 11, 20, 21], so only
    prompt suffix [30] is appended. The single result is:

        tokens     = [10, 11, 20, 21, 30, 40]
        targets    =      [11, 20, 21, 30, 40]  (= tokens[1:])
        mask       =      [ 0,  1,  1,  0,  1]
        logprobs   =      [0.0, -0.2, -0.3, 0.0, -0.4]
        advantages =      [0.0, 0.0, 0.0, 0.0, 0.0]

    Example -- the second prompt breaks the extension property:
        call 0: prompt=[10, 11], completion=[20, 21]
        call 1: prompt=[10, 11, 30], completion=[40]

    Since [10, 11, 20, 21] is not a prefix of call 1's prompt, this emits two
    results. Their tokens and masks are:

        tokens=[10, 11, 20, 21], mask=[0, 1, 1]
        tokens=[10, 11, 30, 40], mask=[0, 0, 1]

    Returns: list of TokenSequence (possibly empty).
    """
    sequences: list[TokenSequence] = []

    # The sequence being built, one entry per TOKEN (not per target): whether
    # the policy sampled it, and its sampling logprob. The shift to target
    # positions happens once, in _emit.
    tokens: list[int] = []
    sampled: list[int] = []
    logprobs: list[float] = []

    for call in traj.calls:
        if len(call.completion_logprobs) != len(call.completion_tokens):
            raise ValueError(
                "completion_logprobs and completion_tokens differ in length: "
                f"{len(call.completion_logprobs)} vs {len(call.completion_tokens)}"
            )

        suffix = extension_suffix(tokens, call.prompt_tokens)
        if suffix is None:
            # Extension broken: close the current sequence and restart from
            # this call's full prompt.
            _emit(sequences, tokens, sampled, logprobs)
            tokens, sampled, logprobs = [], [], []
            suffix = list(call.prompt_tokens)

        # Context tokens (prompt, env observations): never trained on.
        tokens.extend(suffix)
        sampled.extend([0] * len(suffix))
        logprobs.extend([0.0] * len(suffix))

        # Completion tokens: the policy's own actions.
        tokens.extend(call.completion_tokens)
        sampled.extend([1] * len(call.completion_tokens))
        logprobs.extend(float(lp) for lp in call.completion_logprobs)

    _emit(sequences, tokens, sampled, logprobs)
    return sequences


def _emit(
    out: list[TokenSequence],
    tokens: list[int],
    sampled: list[int],
    logprobs: list[float],
) -> None:
    """Append the finished sequence to `out` if there is anything to train on.

    Position j of mask/logprobs/advantages describes target tokens[j+1], so
    the per-token lists are shifted left by one. Token 0 has no slot: nothing
    precedes it, so it is never a prediction target.
    """
    if len(tokens) < 2:
        return
    mask = sampled[1:]
    if not any(mask):
        return
    out.append(
        TokenSequence(
            tokens=list(tokens),
            mask=list(mask),
            logprobs=[lp if m else 0.0 for lp, m in zip(logprobs[1:], mask)],
            advantages=[0.0] * len(mask),
        )
    )


def trajectory_to_training_sequences(traj: Trajectory, advantage: float) -> list[TokenSequence]:
    """Stitch a trajectory and broadcast a scalar advantage over its action
    tokens: every position with mask == 1 gets `advantage`, every other
    position stays 0.0. (In REINFORCE/GRPO, all tokens the policy generated
    in an episode share the episode-level advantage.)
    """
    advantage = float(advantage)
    sequences = stitch_trajectory(traj)
    for seq in sequences:
        seq.advantages = [advantage if m else 0.0 for m in seq.mask]
    return sequences
