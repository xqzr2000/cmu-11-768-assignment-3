# Assignment 3 plan

Goal: finish CMU 11-768 Assignment 3 (ScottyRL) in one week at about one hour a day.
Claude writes the code; Russell runs it in Codespaces, owns the Modal account, and reads
`docs/Learning-Progress.md`.

## Decisions

| Topic | Decision |
|---|---|
| Starter code | Imported from `cmu-agents/assignment-3` with its history; the `upstream` remote points there |
| Coding | Claude, test-first, one branch and pull request per part |
| Training backend | Modal, using the course's own launch scripts unchanged |
| Budget | About $250 for all seven runs (Starter plan gives $30 of credit per month) |
| Model and configs | Unchanged: `Qwen/Qwen3.5-4B`, LoRA rank 32, provided YAML configs and run names |

## Schedule

Each day is sized for about an hour of your time. GPU runs keep going after you step away.

| Day | Claude delivers | You do | Modal cost |
|---|---|---|---|
| 1 | Repo scaffold, Part 1 (`advantages.py`, `datagen.py`) | Open a Codespace, run `uv run pytest`, merge the Part 1 PR, sign up for Modal | $0 |
| 2 | Part 2 (`train/common.py`, `tinker_adapter.py`, `train/sync.py`) | Merge, run tests, `modal setup`, build the image with the smoke test | ~$2 |
| 3 | Probe commands and what to look for | 10-step probe, then `p2a`, `p2b`, `p41` in parallel on three sync apps | ~$30 |
| 4 | Part 3 (`train/async_grpo.py`) | Merge, run tests, probe, then `p3` and both `p43` runs in parallel on three async apps | ~$30 |
| 5 | TextCraft checklist | Short probe, then launch `p5` and leave it running (6-8 hours) | ~$100-130 |
| 6 | Evaluation and export commands | Export the adapter, run the four evaluations, stop every app | ~$10-20 |
| 7 | Plots and report from the run logs | Read the report, close out | $0 |

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
- `HF_TOKEN` (Day 6, to publish the Part 5 adapter)
- `WANDB_API_KEY` (optional live charts)

## Not in scope

This is an independent learning run of the assignment, not a submission for credit.
The hidden grading suite and `package_submission.py` are not used.
