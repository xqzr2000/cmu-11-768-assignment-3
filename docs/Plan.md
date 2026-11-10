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
| Budget | About $250 for all seven runs (Starter plan gives $30 of credit per month) |
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

Model, LoRA rank, environments, algorithms, configs and run names are unchanged, so the
results are comparable with the course's.

## Schedule

All code is written up front; the week is for running it. Each day is about an hour of your
time. Commands are in [Runbook.md](Runbook.md).

| Day | You do | Modal cost |
|---|---|---|
| 1 (done Nov 10) | Merge the pull requests, open a Codespace, run the tests, sign up for Modal, add secrets, build the image, 10-step probe | ~$5 |
| 2 | `p2a`, `p2b`, `p41` in parallel on three sync servers | ~$30 |
| 3 | Async probe, then `p3` and both `p43` runs in parallel on three async servers | ~$30 |
| 4 | TextCraft probe, then launch `p5` and leave it running (6-8 hours) | ~$100-130 |
| 5 | Download the adapter, run the four evaluations, stop every app, push the logs | ~$10-20 |
| 6 | Read the report Claude builds from the logs | $0 |
| 7 | Spare day for a rerun | - |

Running three Number Search servers at once uses 6 H100s, inside the Starter plan's limit of 10.

## Spending rules

1. No GPU until the tests for that part pass locally.
2. A 10-step probe (`--set train.num_steps=10 --set train.run_name=probe_<name>`) before every full run.
3. `uv run modal app stop <app>` as soon as a run ends. An idle server bills for five more minutes.
4. TextCraft is launched once, only after the async trainer has worked on Number Search.
5. Stop and reassess if the Modal dashboard passes $200.

## Secrets

Kept as Codespaces secrets, not in a `.env` file:

- `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` (from the Modal dashboard; these replace `modal setup`)
