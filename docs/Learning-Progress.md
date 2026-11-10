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
