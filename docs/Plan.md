# Assignment 3 plan

Goal: finish CMU 11-768 Assignment 3 (ScottyRL) in one week at about one hour a day.
Claude writes the code; Russell runs it in Codespaces, owns the Modal account, and reads
`docs/Learning-Progress.md`.

## Decisions

| Topic | Decision |
|---|---|
| Starter code | Imported from `cmu-agents/assignment-3` with its history; the `upstream` remote points there |
| Coding | Claude; Part 1 in one pull request, Parts 2 and 3 in a second |
| Training backend | Modal, using the course's own launch scripts unchanged |
| Budget | About $250 for all seven runs (Starter plan gives $30 of credit per month). Modal spend limit set to $300 on Nov 12 as a safety margin, not as extra budget |
| Model and configs | Unchanged: `Qwen/Qwen3.5-4B`, LoRA rank 32, provided YAML configs and run names |

## Our adaptations

Same approach as Assignments 1 and 2: keep everything under our own control except what
cannot be, which here is the H100s.

| Course setup | Ours |
|---|---|
| Modal GPUs through the course launch scripts | Same. Rented on Russell's own Modal account; this is the one external dependency |
| Hidden grading suite, `package_submission.py` | Not used. The public contract tests plus our own offline dry run stand in |
| Adapter published to Hugging Face for the graders | Skipped. The adapter is downloaded from the Modal volume and evaluated from the local copy |
| Weights & Biases charts | Skipped. Plots are made from the `runs/*.jsonl` logs in this repo |
| Report submitted to the course | Written to `docs/Report.md` |
| Part 5 trainer run from your own machine | Run on Modal as well, in a CPU-only container next to the GPU server (`run_trainer.py`), so the 6-8 hour run survives the Codespace being stopped |

Model, LoRA rank, environments, algorithms, configs and run names are unchanged, so the
results are comparable with the course's.

## Schedule

All code is written up front; the week is for running it. Each day is about an hour of your
time. Commands are in [Runbook.md](Runbook.md).

| Day | You do | Modal cost |
|---|---|---|
| 1 (done Nov 10) | Merge the pull requests, open a Codespace, run the tests, sign up for Modal, add secrets, build the image, 10-step probe | ~$5 |
| 2 (done Nov 11) | `p2a`, `p2b`, `p41` in parallel on three sync servers | ~$30 |
| 3 (done Nov 12) | Async probe and `p3` on Nov 11 (held to one run by Modal's $20 spend limit); limit raised, both `p43` runs probed and run on Nov 12 | ~$30 |
| 4 (done Nov 13) | Trainer app deployed, TextCraft probe, `p5` run on Modal overnight. Stopped at step 41 of 50 after the GPU server started leaving requests stuck; about 10.5 hours of 4xH100 | ~$195 |
| 5 | Evaluate the step-41 model and the base model on the same 50 validation tasks the training run used (4 attempts each); download the TextCraft trajectories | ~$12-16 |
| 6 | Analysis from the logs and trajectories (no GPU), see "Adapted plan" below | $0 |
| 7 | Read the report Claude builds: RL findings and the cloud experience side by side | $0 |

### Adapted plan (decided Nov 13)

After Day 4 the goal changed from finishing the assignment as written to learning from what
we have. The `p5` run stays at step 41: finishing the last 9 steps would cost about $40-45 on a
fresh server and mostly extend a curve that is already clear.

- **Evaluation, reduced.** The course evaluates both models on the full validation set and a
  hard set, about 2,800 TextCraft episodes. We evaluate the step-41 model and the base model
  on the first 50 validation tasks, 4 attempts each: the same tasks and settings as the
  training run's own evaluations at steps 0-30, so the five points form one curve. The base
  model also checks that the evaluation server (vLLM serving the adapter) agrees with the
  training server's step-0 number. No hard set.
- **Analysis, no GPU.** From the run logs and the saved trajectories (every 5th TextCraft
  step, on the `scottyrl-trainer-runs` volume; every step for Number Search):
  1. How TextCraft episodes fail: out of turns, malformed calls, wrong recipes, and how the
     mix changes over training.
  2. Thinking length against success, early and late in training.
  3. Groups with no signal: how many are all-fail and how many all-success, per stage.
  4. Number Search (Parts 2-4) against TextCraft (Part 5): two shapes of learning.
  5. Cost per point of improvement, and where the GPU time went.
- **Report.** RL findings and the Modal experience side by side, with the AWS comparison.
- **Budget.** About $241 spent; the reduced evaluation brings it to about $255. Going a little
  over the $250 budget is accepted; the $300 Modal limit stays.

Running three Number Search servers at once uses 6 H100s, inside the Starter plan's limit of 10.

## Spending rules

1. No GPU until the tests for that part pass locally.
2. A 10-step probe (`--set train.num_steps=10 --set train.run_name=probe_<name>`) before every full run.
3. `uv run modal app stop <app>` as soon as a run ends. An idle server bills for five more minutes.
   "Ends" means its terminal has printed `[checkpoints] wrote ...`: stopping an app kills any
   run still using it (Nov 12).
4. TextCraft is launched once, only after the async trainer has worked on Number Search.
5. Stop and reassess if the Modal dashboard passes $200.

## Secrets

Kept as Codespaces secrets, not in a `.env` file:

- `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` (from the Modal dashboard; these replace `modal setup`)
