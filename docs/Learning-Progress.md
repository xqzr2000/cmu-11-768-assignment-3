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

## Day 1 (continued) - Parts 2 and 3: the two trainers

Files: `train/common.py`, `tinker_adapter.py` (one method), `train/sync.py`,
`train/async_grpo.py`. All 25 course tests pass, plus 17 of ours, including an offline
end-to-end dry run of both trainers (`tests/test_offline_end_to_end.py`).

### The shared pieces (`train/common.py`)

- `collect_group` runs G rollouts of one task at the same time with `asyncio.gather`. Each
  rollout gets its own environment and agent. Running them together matters: the GPU server
  batches the overlapping model calls, so eight rollouts take about as long as one.
- `groups_to_batch` is Part 1 applied to many groups: compute each group's advantages, skip
  degenerate groups (but still count their rewards), and flatten everything else into one
  list of training sequences.

### Talking to the training server (`submit_update`)

One update is two requests: `forward_backward` (compute the loss and gradients) and
`optim_step` (apply them with Adam). Each returns a future. The code submits **both** before
waiting for either, because the server batches requests that arrive together; waiting for
the first before sending the second would roughly double every step's latency.

### The synchronous loop (`train/sync.py`)

```
policy = snapshot("step_000000")
for step in 1..N:
    groups = collect rollouts with policy      # GPUs sample, trainer idle
    seqs   = build batch
    if seqs:  update; policy = snapshot(step)  # GPUs train, sampling idle
    record; evaluate if due
```

Everything is on-policy: each batch comes from the weights the previous update produced, so
the importance ratio should sit very close to 1. If every group in a batch is degenerate
there is nothing to learn from; the step is recorded as skipped with no update and no new
snapshot.

REINFORCE and GRPO use this same loop. Only `cfg.algo` (the advantage rule) and
`cfg.loss_fn` differ.

### The asynchronous loop (`train/async_grpo.py`)

The sync loop wastes time: sampling waits during updates and the learner waits during
rollouts. The async trainer overlaps them with a producer-consumer design:

```
rollout workers  ->  bounded queue  ->  learner
```

- **Workers** loop forever: take the next task id, capture the current `(policy, version)`
  pair once, generate a whole group with it, put the result on the queue. Capturing the pair
  up front means the version label is still right if a new policy is published mid-rollout.
- **The queue is bounded.** When it is full, `await queue.put()` pauses the worker. That
  backpressure stops workers from piling up data that would be stale by the time it is used.
- **The learner** pulls groups until it has a batch. A group is dropped as *stale* if it was
  sampled more than `max_staleness` versions ago, judged at the moment it is consumed. Then
  it updates and publishes version + 1.
- **Importance weighting** corrects the groups that are a little stale but kept: the PPO loss
  scales each token by how much more or less likely the current policy finds it than the
  policy that sampled it, with clipping.
- **Evaluation** runs as a background task on a captured `(policy, version)`, so training
  does not wait for it.
- **Failure handling.** If a worker dies, the learner would wait forever for a batch. So the
  learner waits on "batch ready OR a background task failed" (`_first_or_failure`), and on
  any error cancels everything and re-raises the original exception.

### What changes in the plan

The code for all five parts is now written, a day ahead of the first schedule. The rest of
the week is running it on Modal; see `docs/Runbook.md`.

## Day 1 (Nov 9-10) - First contact with real GPUs

Everything planned for Day 1 is done: both pull requests merged, 42 tests passing in the
Codespace, Modal account and secrets set up, server image built, and a 10-step probe run on
two H100s.

### Two things that went differently from the plan

- **Modal client too old.** The starter pins `modal` 1.5.5, and a new Modal workspace uses
  image builder `2026.10`, which that client refuses. Fixed with
  `uv lock --upgrade-package modal` (now 1.6.3). Only `modal` and one helper package changed;
  the pinned `tinker==0.24.0` is untouched. This is the kind of drift a pinned course repo
  picks up within weeks, and the reason lockfiles sometimes need a targeted upgrade.
- **First start is slow.** The first request to a new server downloads the Qwen weights into
  a Modal volume before anything else happens: several quiet minutes. Later starts reuse them.

### The probe (`probe_p2b`: GRPO, 8 tasks x 8 rollouts, 10 steps)

| Check | Result |
|---|---|
| Baseline evaluation at step 0 | Present: success rate 0.30, pass@4 0.60 on 20 tasks |
| One training record per step, evaluation every 2 steps | Yes, steps 1-10 |
| `train/importance_weight_mean` | 1.000 on every step, std about 0.02 |
| `train/grad_norm` | 0.07-0.14 (clip limit is 1.0) |
| Sanity warnings | None |
| Step time after warm-up | About 30-70 s, so 50 steps is about 40 minutes |
| Resumable checkpoint | Saved at step 10 |

How to read it:

- **Importance weight near 1 is the alignment check.** In a synchronous run the policy that
  sampled a token is the policy being trained, so the ratio of their probabilities should be
  1. It is, which confirms the off-by-one handling in `datagen.py` on real data. A wrong
  alignment would compare each log-probability with a different token and the ratio would be
  far from 1.
- **Why not exactly 1?** Sampling runs in vLLM and training in Megatron. The two engines
  compute the same model with slightly different numerics, so single tokens range from about
  0.45 to 2.0 while the mean stays at 1.
- **Degenerate groups appeared** (0-25% per step): groups where all eight rollouts got the
  same reward and were skipped, exactly as designed.
- **Evaluation did not improve** (0.30 to 0.325). Expected: ten small updates at a learning
  rate of 1e-5 are a wiring test, not training. The 50-step runs are where the curve should
  rise.
- **The untrained model uses almost its whole step budget** (`eval/avg_steps` about 8.8). It
  is not bisecting yet. Whether it learns to is the question the Day 2 runs answer.
- `rollout_train_logprobs_abs_diff_max` of about 20.6 looks alarming but is very likely a
  server-side metric that includes prompt tokens, where our side records 0 on purpose. This
  is an inference from the numbers, not something checked in the server code.

### Spend so far

Roughly $5: the image build on one H100 and about 12 minutes of two H100s for the probe.
The exact figure is on the Modal dashboard.

### Next

Day 2 in `docs/Runbook.md`: `p2a`, `p2b` and `p41` in parallel on three servers.

## Day 2 (Nov 10-11) - The three synchronous runs

All three finished their 50 steps. Logs are in `runs/`. This is the first evidence that the
model's weights changed in a useful direction.

### Results

Held-out evaluation: 50 validation tasks, 4 rollouts each.

| Run | Algorithm | Episodes per step | Success at step 0 | Step 20 | Step 30 | Step 50 | pass@4 at step 50 |
|---|---|---|---|---|---|---|---|
| `p2a` | REINFORCE | 8 (8 tasks x 1) | 0.340 | 0.430 | 0.935 | 0.955 | 1.00 |
| `p2b` | GRPO, group 8 | 64 (8 tasks x 8) | 0.355 | 0.620 | 0.995 | 0.990 | 1.00 |
| `p41` | GRPO, group 2 | 16 (8 tasks x 2) | 0.330 | 0.385 | 0.505 | 0.565 | 0.92 |

The step-0 numbers differ slightly between runs (0.33 to 0.355) although all three start
from the same untrained model. That spread is sampling noise, and it sets the scale for
reading everything else: differences of a few points mean nothing.

### What the curves say

- **The model learned the task.** Two runs went from about one success in three to nearly
  every episode. Average steps per episode fell from about 8.6 to 7.6 at the same time.
- **Learning came as a jump, not a ramp.** Both `p2a` and `p2b` were still near baseline at
  step 10 and essentially solved by step 30.
- **GRPO with 8 rollouts was the cleanest.** It reached 0.995 and stayed there.
- **Group size 2 learned much more slowly.** It was still climbing at step 50.

### Why group size matters so much: degenerate groups

A GRPO group teaches something only when its rollouts got different rewards. With a success
rate p and a group of 2, the chance that both rollouts agree is p^2 + (1-p)^2. At p = 0.35
that is 0.55, so more than half the groups are thrown away before training starts. The log
agrees: `p41` discarded 53-64% of its groups in every stretch of the run. A group of 8 at the
same success rate is almost never all-same (about 3%), so `p2b` used nearly everything it
collected early on.

The opposite happens at the end. Once `p2b` succeeds 99% of the time, almost every group is
all-success: 97.5% of its groups were degenerate over the last 20 steps, and 18 of its 50
steps were skipped outright with no update. That is not a fault. The baseline has nothing
left to compare against, so the run stops changing the model. It also means roughly the last
third of that run's GPU time bought nothing.

### REINFORCE did better than expected

`p2a` trained on only 8 episodes per step and still reached 0.955. With rewards of 0 or 1
and no baseline, REINFORCE simply pushes up whatever the successful episodes did and ignores
the failures. On a task this simple that is enough. Its curve was rougher on the way (it
dipped to 0.275 at step 10) and it ended a little below GRPO-8. One run each is too little
to rank them firmly; a second seed could reorder `p2a` and `p41`.

### Sanity numbers

| Check | `p2a` | `p2b` | `p41` |
|---|---|---|---|
| `importance_weight_mean`, whole run | 1.000 | 1.000 | 1.000 |
| `importance_weight_std` | 0.014-0.023 | 0.012-0.023 | 0.020-0.023 |
| `grad_norm` (clip is 1.0) | 0.15-0.27 | 0.05-0.10 | 0.23-0.34 |
| Skipped steps | 0 | 18 | 1 |

On-policy all the way through, as a synchronous run should be. `p2b`'s smaller gradient norm
fits its larger batch: 64 episodes average out more noise than 8 or 16.

### Where the time went

Each run took 43-49 minutes. Inside `p2a`, about 11 minutes were rollouts and about 11
minutes were updates; the rest was evaluation and saving snapshots. Sampling and training
take turns, so each GPU role sits idle while the other works. Overlapping them is the whole
point of the asynchronous trainer on Day 3.

### Open question for the report

Episodes got longer in tokens while getting shorter in steps: mean sequence length went from
about 450 to 850-950 tokens in `p2a` and `p2b`. The model is writing more per turn than it
did before training. The transcripts under `runs/*_trajectories/` (kept in the Codespace,
not in git) will show what it is writing; worth comparing one early and one late episode.

### Next

Day 3 in `docs/Runbook.md`: the asynchronous probe, then `p3` and the two `p43` runs.

## Day 3 (Nov 11) - The asynchronous trainer on real GPUs: `p3`

A 10-step probe, then the full 50-step `p3` run. The two `p43` runs are postponed: Modal's
default spend limit on a new account is $20 a month beyond the free credit, and tonight had
room for one run (see `cloud-walker-Modal.md`).

### Same model quality in half the time

`p3` and Day 2's `p2b` use the same algorithm and batch: GRPO, 8 groups of 8 rollouts per
step, 50 steps. Only the trainer differs.

| | `p2b` synchronous | `p3` asynchronous |
|---|---|---|
| Success at step 0 | 0.355 | 0.345 |
| Step 10 | 0.415 | 0.445 |
| Step 20 | 0.620 | 0.690 |
| Step 30 | 0.995 | 0.950 |
| Step 40 | 0.995 | 0.990 |
| Step 50 | 0.990 | 0.990 |
| Wall-clock time for 50 steps | 49.0 min | 24.6 min |
| Time the learner spent waiting for data | 32.4 min | 5.1 min |
| Time spent in updates | 11.6 min | 13.3 min |
| Steps skipped (all groups degenerate) | 18 | 8 |

The learning curves match step for step within noise. The difference is the clock: the
synchronous learner spent two thirds of its run waiting for rollouts, and the asynchronous
one spent a fifth. Same GPUs, same cost per minute, half the minutes.

### What the async run shows that the sync run cannot

- **Evaluation in the background.** The step-10 evaluation line printed after training step
  13, labelled `eval/policy_version 10`. Training did not pause for it, and the label proves
  it measured the policy captured at step 10, not whatever was current when it finished.
- **Policy version and step number separate.** A skipped step does not publish a new
  version, so by step 50 the policy was at version 42. The step-40 evaluation is labelled
  version 39 for the same reason.
- **Stale data, dropped.** The learner pulled 453 groups from the queue and discarded 53
  (12%) because they were sampled more than 2 versions earlier. Those rollouts cost GPU time
  and taught nothing.
- **Stale data, kept and corrected.** Groups up to 2 versions old were used. The importance
  weight still averaged 1.000, with a wider spread than the sync run (0.025 against 0.019),
  and one token at step 22 reached a ratio of 4.98, far outside anything seen on Day 2.
  That is what "slightly off-policy" looks like in numbers, and it is what the PPO loss's
  ratio and clipping are for.

### The end of the run

From step 34 the `[sanity]` warning about degenerate groups fired on most steps, and 7 of
the last 8 steps were skipped. Not a fault: the policy succeeds on 99% of tasks, so nearly
every group is all-success and carries no signal. In the skipped steps
`learner_wait_frac` is 1, since there was no update and the learner only waited.

One line is easy to misread: `[resume] checkpointed at step 41 (40 optimizer steps)`. Step
38 was skipped, so the 40th optimizer step happened at iteration 41. Iterations and
optimizer steps are different counters.

### The probe, for the record

`probe_p3`: 10 steps, `groups_consumed` 80, `groups_dropped_stale` 24, clean shutdown. The
drop rate was higher than in the full run (23% against 12%) because the first update on a
cold server took 112 seconds, and the workers kept filling the queue with version-0 data
while it ran.

### Still to run

`p43_number_search_async_stale4_iw` and `p43_number_search_async_stale4_noiw`: accept data up
to 4 versions old, with and without the importance-weighting correction.
