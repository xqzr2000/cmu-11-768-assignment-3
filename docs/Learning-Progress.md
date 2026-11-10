# Learning progress

A running log for Assignment 3. Newest entry at the bottom. Each entry says what was built,
why it works, and what to try yourself.

## Day 0 - What this assignment is

Assignments 1 and 2 used a model as it was: first to act (the agent harness), then to judge
(the evaluator). Assignment 3 changes the model itself. The loop is:

1. **Roll out.** The current policy (Qwen3.5-4B plus a small LoRA adapter) plays an episode
   of a task. Every model call is recorded as token ids, together with the log-probability
   the model gave each token it sampled.
2. **Score.** The environment gives a reward: 1 if the task was solved, 0 otherwise.
3. **Turn episodes into training data.** Mark which tokens the model chose itself, and attach
   an *advantage* to them: a number saying whether this episode went better or worse than
   expected.
4. **Update.** Send the batch to the training server, which nudges the adapter weights so
   that tokens with positive advantage become more likely and tokens with negative advantage
   less likely.
5. **Repeat** with the updated policy.

Where each piece runs:

| Piece | Runs on | Needs a GPU |
|---|---|---|
| Environments, agent, rewards, data pipeline, trainers, tests | Codespaces | No |
| Model sampling (vLLM) and weight updates (SkyRL + Megatron) | Modal, 2 or 4 H100s | Yes |

The two sides talk through the Tinker API. The trainer only needs one setting,
`TINKER_BASE_URL`, pointing at the Modal deployment.

### Why hosted model APIs cannot do this

The DeepSeek and GPT endpoints from Assignments 1 and 2 return text. Training needs three
things they do not expose: the log-probability of every sampled token, a backward pass
through the model, and an optimizer step on its weights. That is why this assignment rents
GPUs instead of buying tokens.

### The two environments

- **Number Search**: guess a hidden integer with too-high / too-low feedback inside a step
  budget. Cheap (about 40 minutes a run); used for six of the seven runs.
- **TextCraft-Synth**: craft target items from an inventory and a recipe book using
  `craft`, `get_info`, `view_inventory`, `finish`. Long episodes; one 6-8 hour run.

### Starting point

`uv run pytest` on the untouched starter: 12 of 25 tests pass, 13 fail with
`NotImplementedError`. Those 13 are the six files this repo fills in.

## Day 1 - Part 1: the data pipeline

Files: `scottyrl/advantages.py`, `scottyrl/datagen.py`. Tests: `tests/test_part1_pipeline.py`
(ours, written first) plus the two course tests for this part.

```bash
uv run pytest tests/test_part1_pipeline.py tests/test_interfaces.py -k "part1 or advantages or datagen"
```

### Advantages: how good was this episode, compared with what?

A reward of 1 on an easy task says little; a reward of 1 on a task the model usually fails
says a lot. The advantage is the reward measured against a baseline.

- **REINFORCE** uses no baseline: advantage = reward. Successful episodes are reinforced,
  failed ones contribute nothing. Correct on average, but noisy.
- **GRPO** runs a *group* of G rollouts of the same task and subtracts the group's mean
  reward. With rewards `[1, 0, 0, 0]` the mean is 0.25 and the advantages are
  `[0.75, -0.25, -0.25, -0.25]`: the one success is pushed up, the failures pushed down.

If every rollout in a group gets the same reward, every advantage is zero and the group
teaches nothing. The function returns `None` for such a *degenerate* group so the trainer can
skip it. Early in training most groups are all-fail, late in training many are all-success,
so the fraction of degenerate groups is one of the diagnostic plots.

This is the Dr.GRPO variant: no division by the group's standard deviation.

### Stitching: one episode, one training sequence

An episode is several model calls. Call 2's prompt is call 1's prompt, plus call 1's
completion, plus the newly rendered tail (the environment's reply and the next turn marker).
When that holds, the whole episode can be laid out as one long token sequence and trained in
one pass. The docstring example:

```
call 0: prompt=[10, 11]              completion=[20, 21]
call 1: prompt=[10, 11, 20, 21, 30]  completion=[40]

tokens  = [10, 11, 20, 21, 30, 40]
targets =     [11, 20, 21, 30, 40]     (tokens[1:])
mask    =     [ 0,  1,  1,  0,  1]
```

The mask is 1 only where the target token was sampled by the policy. Token 30 came from the
environment, so it is 0: the model should learn to act, not to predict the environment.

**The off-by-one.** Training predicts the *next* token, so a sequence of n tokens has n-1
targets, and position j describes `tokens[j+1]`. The code avoids index arithmetic: it keeps
one flag and one logprob per token while building, then drops the first entry of each list
at the end (`sampled[1:]`, `logprobs[1:]`). Token 0 is never a target because nothing
precedes it.

**When the extension breaks.** With `preserve_thinking_history: false` the chat template
removes earlier `<think>` blocks, so the next prompt no longer starts with everything
accumulated so far. `extension_suffix` detects this (returns `None`), the current sequence is
closed, and a new one starts from that call's full prompt. Number Search episodes normally
give one sequence; a think-stripped episode gives one sequence per call. Both cases are
tested against real agent episodes.

**Logprobs.** Each sampled token carries the log-probability the policy gave it at sampling
time. The trainer later compares it with the current policy's log-probability to form the
importance ratio. If these were attached to the wrong positions the ratio would drift far
from 1, which is the first thing to check in a synchronous run.

### Broadcasting

`trajectory_to_training_sequences` copies the episode's single advantage onto every masked
position. All tokens the policy wrote in an episode share credit for its outcome.

### Status

29 tests pass (12 from the starter, 2 course tests for Part 1, 15 of ours). The 11 still
failing belong to Parts 2 and 3.

### Try it

Change the mask in `_emit` to `sampled[:-1]` (the wrong alignment) and run the tests. See
which ones catch it, then change it back.
