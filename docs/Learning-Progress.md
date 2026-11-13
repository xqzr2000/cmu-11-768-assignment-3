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
to 4 versions old, with and without the importance-weighting correction. Done on Nov 12; see
the next entry.

## Day 3, finished (Nov 12) - Staleness 4, with and without importance weighting

Modal raised the spend limit, so Day 3 was finished as planned: a 10-step probe of each
`p43` config, then both 50-step runs in parallel on two 2xH100 async servers. `p3` was kept
from Nov 11; it was clean, so there was nothing to gain from rerunning it.

The two runs differ only in the loss:

- **`iw`** uses the PPO loss, as `p3` did: each token's update is scaled by the ratio
  between the current policy's probability and the sampling policy's, clipped to a narrow
  range.
- **`noiw`** uses advantage-weighted cross-entropy: every token the policy wrote is pushed
  up or down by the advantage alone, as if it had been sampled by the current policy. This
  was the first time that loss ran on real GPUs; the probe confirmed it (`loss_fn_is_iw` 0,
  normal gradient norms).

The course runs these one after the other against the `p3` deployment. We ran them side by
side on two fresh servers with the same launch script, which changes the clock time and
nothing else.

### Results

Held-out evaluation: 50 validation tasks, 4 rollouts each.

| Step | `p3` (staleness 2, PPO) | `iw` (staleness 4, PPO) | `noiw` (staleness 4, no IW) |
|---|---|---|---|
| 0 | 0.345 | 0.330 | 0.350 |
| 10 | 0.445 | 0.450 | 0.405 |
| 20 | 0.690 | 0.580 | 0.780 |
| 30 | 0.950 | 0.850 | 0.995 |
| 40 | 0.990 | 0.965 | 0.980 |
| 50 | 0.990 | 0.985 | 1.000 |

| | `p3` | `iw` | `noiw` |
|---|---|---|---|
| Groups consumed / dropped as stale | 400 / 53 | 400 / 0 | 400 / 1 |
| Steps skipped (all groups degenerate) | 8 | 1 | 11 |
| Policy version at step 50 | 42 | 49 | 39 |
| Wall-clock time for 50 steps | 24.6 min | 22.5 min | 23.7 min |
| Learner waiting for data, total | 5.1 min | 1.2 min | 7.1 min |
| `importance_weight_std`, mean over the run | 0.025 | 0.029 | 0.039 |
| `importance_weight_std`, highest step | 0.058 | 0.049 | 0.127 |
| Largest single-token ratio | 4.98 | 3.75 | 5.47 |
| Smallest single-token ratio | 0.121 | 0.315 | 0.013 |

### What the numbers say

- **All three reached the same place.** Every run ended at 0.985-1.0. Raising the
  staleness limit from 2 to 4 cost nothing in final quality on this task, with or without
  the correction.
- **The order on the way up is not evidence.** `noiw` crossed 0.9 first and `iw` last, but
  Day 2 showed that the timing of the jump moves by several steps between runs, and each
  configuration ran once. A second seed could reverse the order.
- **Staleness 4 kept almost everything.** `p3` threw away 53 groups (12%) that were 3 or
  more versions old. At a limit of 4 the learner kept all of them, and the `iw` learner was
  hardly ever idle: 1.2 minutes of waiting in 22.5.
- **The uncorrected run drifted further from its sampler.** In `noiw` the importance
  weights are computed but not used, so they show how off-policy its data really was. From
  step 11 to step 28, the stretch in which it went from 0.4 to 0.995, the spread averaged
  0.056, against 0.031 for `iw` over the same steps and about 0.02 in the synchronous runs,
  and peaked at 0.127 at step 22. Single tokens went as
  low as 0.013: a token the sampling policy chose that the current policy now found about 75
  times less likely. `noiw` still trained on those tokens at full weight. `iw` saw milder
  tails (largest 3.75, smallest 0.315) and its PPO loss clipped them.
- **Why the missing correction did not hurt here.** The mean ratio stayed at 1.000 in every
  run. Most tokens were almost on-policy, and the badly off-policy ones were a small tail. On
  a task this easy, with a small learning rate and a LoRA adapter, a small biased tail does
  not change where training ends up. The correction matters when that tail is a large part
  of the batch: higher staleness, larger learning rates, or a policy changing fast for many
  steps in a row. TextCraft's long episodes are a better place to look for that.
- **Why `noiw` skipped more steps.** It learned fastest, so it spent more of the run at
  99-100% success, where every group is all-success and carries no signal. Skipped steps
  publish no new version, which is why it ended at version 39 against `iw`'s 49, and why its
  learner-wait total is high: in a skipped step the learner only waits.

### For the report

Part 4.3 asks for the standard diagnostic plots of both runs. The importance-weight spread
over steps, `noiw` against `iw`, is the plot that shows the difference; the success curves
alone do not.

### A mistake worth remembering

The two apps were stopped a few minutes after the probes, while the full `iw` run had just
started on its server. That run died in its cold start, before its step-0 evaluation, and
left an empty log and a manifest that blocked a fresh start under the same run name. The fix
was to delete the two files, redeploy and start again; the cost was one extra cold start
per server. Rule added to `Plan.md`: stop an app only after its terminal has printed the
`[checkpoints] wrote ...` line.

### Next

Day 4 in `docs/Runbook.md`: the TextCraft probe, then the 6-8 hour `p5` run. Install `tmux`
in the Codespace first.

## Day 4 (Nov 12-13) - TextCraft, `p5_textcraft_grpo`

Part 5: Qwen3.5-4B with thinking on, trained with asynchronous GRPO on TextCraft (craft a
target item from a recipe book, with a turn limit per task, 75 on the validation set). 16 tasks x 8 attempts per step, 50 steps
planned. The trainer ran on Modal next to the 4xH100 server, so the run did not need the
Codespace (see `docs/cloud-walker-Modal.md`).

**It stopped at step 41 of 50.** Requests on the GPU server started getting stuck partway
through the night; the trainer crashed once and hung once (details below). With the 12-hour
cap an hour away and the server unreliable, we stopped it. The model Day 5 evaluates is the
step-41 snapshot, `tinker://model_367b3579/v41_r2`.

### Results

Held-out evaluation, 50 validation tasks x 4 attempts:

| Step | Success | pass@4 | Average turns |
|---|---|---|---|
| 0 (base model) | 0.115 | 0.26 | 64.5 |
| 10 | 0.195 | 0.44 | 60.5 |
| 20 | 0.335 | 0.62 | 56.4 |
| 30 | **0.500** | **0.70** | 53.0 |

The step-40 evaluation was lost in the first crash. Training rollouts, averaged per stretch:

| Steps | Success | pass@8 | Groups with no signal | Step time |
|---|---|---|---|---|
| 1-10 | 0.16 | 0.46 | 55% | 10.7 min |
| 11-20 | 0.27 | 0.62 | 40% | 12.2 min |
| 21-30 | 0.37 | 0.70 | 35% | 12.7 min |
| 31-41 | 0.53 | 0.71 | 47% | 13.3 min |

### What the numbers say

- **TextCraft learned steadily, without a jump.** Number Search went from 0.4 to 0.99 within
  ten steps once it found the trick. TextCraft climbed by about 0.15 per ten steps and was
  still rising when it stopped (training success 0.58 at step 40, 0.91 at step 41). More
  steps would very likely have helped.
- **Episodes got shorter.** Average turns per evaluation episode fell from 64.5 to 53. The
  base model mostly runs out of turns before finishing; the trained one finishes more often
  and sooner.
- **Most groups at the start carried no signal.** 55% of the early groups were all-fail:
  eight attempts at a hard task, none succeeding, so GRPO has nothing to compare. That
  fell to 35% by steps 21-30, as the model solved more tasks some of the time. Late in the
  run it rose again, now partly from all-success groups.
- **The run was limited by episode generation, not by learning.** An episode took about 15
  minutes on average (12-16k tokens with thinking). The learner waited for data 68% of the
  time; an update took about 3 minutes. The probe's second step (3.7 minutes, update-bound)
  was a poor guide to the full run: 50 steps would have needed about 10.5 hours, not 6-8.
- **Importance weights stayed at 1.000 throughout**, as in Part 4. Staleness was not a
  problem here; time was.

### What went wrong

1. **Stuck requests on the GPU server.** From about step 10, a few rollout requests never
   completed. The Tinker client gives up after 7,200 seconds (`No progress made in 7200s.
   Requests appear to be stuck.`), so those episodes were abandoned and the run carried on.
   After step 42 a request from the learner itself got stuck, and the trainer crashed.
   The runner resumed from the step-40 checkpoint; on the same server, the learner hung
   again after step 42 and the 60-minute watchdog stopped it; the last retry was at step 41
   when we stopped the run. We do not know the server-side cause. The pattern (rare at
   first, then blocking) points at the server degrading over a long session; a fresh server
   for the retry might have done better.
2. **A parser bug** (fixed in PR #10). When the model wrote an argument Python cannot build,
   such as a dict inside a set, `parse_tool_call` raised a `TypeError` instead of treating
   it as a malformed call, and the episode was abandoned. About ten episodes were lost. It
   slightly understates training success; it would have hit the evaluation too.

Together these abandoned 9 training episodes (in the steps kept in the log) and 2
evaluation episodes out of several thousand, so the results above stand.

### For the report

- The success curve against Part 4's, to show the difference between an easy task with a
  sudden jump and a hard one with steady progress.
- Turns per episode falling, as the efficiency gain.
- The step time breakdown (learner waiting 68%), to explain why long-horizon RL with
  thinking is expensive.
- That the run ended at step 41 of 50, and why.

### Next

Day 5: evaluate the step-41 model against the base model. The full plan (2 models, 349 tasks,
4 attempts each, about 2,800 episodes) would take about 5 hours on two H100s, more than the
remaining budget allows. To decide next session: a reduced evaluation, for example the first
50 tasks of each set with 4 attempts (800 episodes, about $15-20).

