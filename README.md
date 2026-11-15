# CMU 11-768 Assignment 3 · Training agents with reinforcement learning

This is our independent, one-week implementation of the missing ScottyRL training code and a record of running it on rented Modal H100s. We started from the [CMU course starter](https://github.com/cmu-agents/assignment-3) and post-trained `Qwen/Qwen3.5-4B` with a rank-32 LoRA adapter. Six Number Search runs finished. TextCraft reached step 41 of 50, and we evaluated that adapter against the base model.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xqzr2000/cmu-11-768-assignment-3/blob/main/notebooks/CMU_11_768_Assignment_3_Project_Walkthrough.ipynb)

The [walkthrough notebook](notebooks/CMU_11_768_Assignment_3_Project_Walkthrough.ipynb) traces the code from an agent's first tool call to a training batch, runs a small policy update on a Colab T4, and reads the recorded H100 results. The [final report](docs/Report.md) has the diagnostic plots, TextCraft transcript analysis, and cost breakdown.

## What we built

The starter supplied the environments, XML tool-call agent, policy interfaces, launch scripts, configs, and public tests. We implemented the exercise code in six files and tested the data path with a scripted policy before paying for GPU time.

| Course requirement | Implementation and evidence |
| --- | --- |
| **Part 1 · Data pipeline** | [`advantages.py`](scottyrl/advantages.py) assigns an advantage to each episode. [`datagen.py`](scottyrl/datagen.py) stitches model calls and aligns each next-token target with its mask, sampling log probability, and advantage. [Part 1 tests](tests/test_part1_pipeline.py) cover broken prompt prefixes and token alignment. |
| **Part 2 · Synchronous REINFORCE and GRPO** | [`train/common.py`](scottyrl/train/common.py) collects attempts and builds batches. [`train/sync.py`](scottyrl/train/sync.py) collects, updates, and refreshes the sampler in order. [`TinkerSession.submit_update`](scottyrl/tinker_adapter.py#L311) queues forward/backward and the optimizer step before resolving their futures. |
| **Part 3 · Asynchronous GRPO** | [`train/async_grpo.py`](scottyrl/train/async_grpo.py) overlaps rollout generation and updates. Workers label each group with its sampling policy version. The learner consumes groups from a bounded queue and drops groups over the staleness limit. [Offline trainer tests](tests/test_offline_end_to_end.py) exercise both loops. |
| **Part 4 · Ablations** | We ran the [provided configs](configs/) to compare two versus eight rollouts per group, sync versus async scheduling, and async runs with and without importance weighting at staleness four. |
| **Part 5 · TextCraft-Synth** | We ran the provided async GRPO config through [a detached Modal runner](skyrl_tinker_modal/run_trainer.py), then served the step-41 adapter and base model through vLLM. The [evaluation files](runs/) and [trajectory analysis](runs/p5_analysis.json) are committed. |

In **Number Search**, the model guesses a hidden integer and gets higher/lower feedback. In **TextCraft**, it follows recipes, crafts the target items, and calls `finish()` within the turn budget. Both environments supply a binary episode reward. REINFORCE here uses the raw reward. The assignment's GRPO variant subtracts the mean reward of attempts at the same task, without dividing by group standard deviation. Equal-reward groups yield no update.

The Python trainer orchestrates the agent, environment, and data pipeline. It sends batches over the Tinker API to SkyRL on Modal. vLLM samples rollouts and Megatron updates the adapter. Number Search used two H100s per run. Both GPUs took turns sampling and training in sync mode, while async mode assigned one GPU to each job. TextCraft used three sampling H100s and one training H100. Standalone evaluation used two H100s for vLLM inference.

## Differences from the course handout

| Course workflow | Our adaptation |
| --- | --- |
| Self-host SkyRL on Modal using the provided configs | We used the same model, LoRA rank, environments, experiment configs, and GPU layouts on our own Modal account. Codespaces and GitHub Actions gave us a repeatable Python 3.12 and `uv` setup. |
| Launch the Python trainer from your own machine | We launched Number Search from Codespaces. For the overnight TextCraft run, we deployed a CPU trainer on Modal with logs on a volume, checkpoint-aware retries, a time cap, and a watchdog. |
| Complete 50 TextCraft steps and evaluate both models on medium and hard splits | We stopped at step 41 and evaluated each model on the first 50 medium-validation tasks, four attempts per task. The hard split was left unevaluated. |
| Publish the final adapter to Hugging Face for grading | We downloaded our adapter from its Modal state volume and loaded it into vLLM as `a3` alongside the base model. The repo contains the evaluation evidence and checkpoint metadata. |
| Package a submission and run the grading suite | We ran the public tests and added offline coverage for the data pipeline, trainers, recovery, and evaluation. We made plots from JSONL logs and wrote a report for independent study. |

The original requirements remain in [ASSIGNMENT.md](ASSIGNMENT.md). The [original README and launch notes](https://github.com/cmu-agents/assignment-3/blob/main/README.md) and the [SkyRL deployment guide](skyrl_tinker_modal/README.md) cover the starter workflow.

## From local setup to the overnight run

| Stage | What happened and what we changed |
| --- | --- |
| **Day 0 · Get the starter running** | We set up Python 3.12 and `uv` in Codespaces and added a GitHub Actions test job. The untouched starter passed 12 of 25 tests. The 13 failures came from the exercise stubs raising `NotImplementedError`. |
| **Day 1 · Make the pipeline testable** | We wrote tests for the next-token shift and for a second prompt whose prefix no longer contains the first completion. The mask tracks `tokens[1:]`, and a broken prefix starts a new sequence. Offline trainer tests then drove real environments with scripted completions and a fake Tinker session. |
| **Day 1 · Get the first GPU probe through** | Modal client 1.5.5 rejected the new workspace's `2026.10` image builder. A targeted `uv lock --upgrade-package modal` moved it to 1.6.3. We used Codespaces secrets for the CLI credentials, built and smoke-tested the SkyRL image, and waited for the first request to load model weights. The 10-step GRPO probe wrote a checkpoint and logged an average importance ratio near 1.000. |
| **Day 2 · Run three synchronous experiments** | Three terminals watched three independent 2-H100 services: REINFORCE, GRPO with group size 8, and GRPO with group size 2. The small group often had identical rewards, giving us an explanation for its slower learning beyond the success curve alone. |
| **Day 3 · Move to async and hit an account limit** | A $20 spend limit on our workspace postponed the two staleness-four runs. Modal support raised the ceiling and we set our limit to $300. Later we stopped apps while one full run was still in its cold start. Its empty log and manifest blocked reuse of the run name. We cleared those failed-run artifacts, redeployed, and waited for the checkpoint-manifest line at the end of subsequent runs. |
| **Day 4 · Train TextCraft with the Codespace closed** | The CPU trainer used a detached Modal call, while the GPU service did the sampling and updates. An unbuildable Python literal in a tool's arguments raised `TypeError` and abandoned some episodes. We fixed the parser to handle it as a malformed tool call. Later, GPU requests stalled and the Tinker client could wait 7,200 seconds. We added a 60-minute silence watchdog and retried from checkpoints. Stalls returned on the same server, and we kept the step-41 snapshot. The server-side cause remains unknown. |
| **Day 5 · Recover evaluation** | Evaluation from Codespaces stalled around 1.25 requests/s. A 600-second timeout tied up client slots while the server logged short request times. The inferred request-loss rate was about 4%. We added separate 10-second connect and 120-second response timeouts, retries, per-episode persistence, progress and throughput checks. The second attempt ran from a Modal CPU container and recorded zero retries across both models and the probe. For directory downloads, `modal volume get` takes the parent directory as its local destination. |
| **Days 6–7 · Learn from the recorded runs** | We kept the 41-step run and analyzed 2,004 saved TextCraft transcripts. The report follows how tool-call usability, task size, turn limits, and equal-reward groups affected training. Log-based plots let us finish the write-up without another GPU run. |

The full chronology is in the [learning log](docs/Learning-Progress.md), with build, billing, request logs, and volume details in the [Modal journal](docs/cloud-walker-Modal.md).

## Recorded results

| Number Search run | Step 0 success | Step 50 success |
| --- | ---: | ---: |
| Sync REINFORCE (`p2a`) | 34.0% | 95.5% |
| Sync GRPO, group of 8 (`p2b`) | 35.5% | 99.0% |
| Sync GRPO, group of 2 (`p41`) | 33.0% | 56.5% |
| Async GRPO, staleness 2 (`p3`) | 34.5% | 99.0% |
| Async GRPO, staleness 4 with weighting (`iw`) | 33.0% | 98.5% |
| Async GRPO, staleness 4 without weighting (`noiw`) | 35.0% | 100.0% |

Sync and async GRPO with eight attempts per group both reached 99.0%. The async run took about 24.6 minutes versus 49.0 minutes for sync. The unweighted run showed wider importance ratios, although all three async runs finished within 1.5 percentage points of one another. Each configuration ran once.

| TextCraft standalone vLLM evaluation | Per-attempt success | pass@4 |
| --- | ---: | ---: |
| [Base model](runs/p5_eval_Qwen_Qwen3.5-4B.json) | 6.5% | 14% |
| [Step-41 LoRA adapter](runs/p5_eval_a3.json) | 53.5% | 68% |

Each row covers 200 attempts on the same 50 tasks. Separately, periodic evaluation on the training server went from 11.5% at step 0 to 50.0% at step 30. The trajectory analysis found that turns with no usable tool call fell from 64% to 29% between those two points. No task requiring more than 31 crafted items was solved in the analyzed runs. The report estimates roughly $262 in Modal spend for the week, including about $197 for TextCraft training and $12 for the stalled evaluation.

## Run it yourself

On a local machine or in Codespaces, from the repository root:

```bash
uv python install 3.12
uv sync --locked --python 3.12
uv run --locked pytest
```

The tests, scripted rollouts, and reading the committed metrics work offline after installation. The Colab notebook adds a small PyTorch update designed for a single T4. To reproduce the course's multi-GPU experiments, use [the Modal runbook](docs/Runbook.md). Use a fresh run name for new experiments because the original run names already have committed logs. Resume a run against its own persisted state and matching config.

| Find | Path |
| --- | --- |
| Notebook and course requirement mapping | [`notebooks/`](notebooks/) and [`ASSIGNMENT.md`](ASSIGNMENT.md) |
| Model, environments, training loops | [`scottyrl/`](scottyrl/) |
| Launch commands, experiment settings, data | [`scripts/`](scripts/), [`configs/`](configs/), [`data/`](data/) |
| Modal deployment definitions | [`skyrl_tinker_modal/`](skyrl_tinker_modal/) |
| Raw run and evaluation records | [`runs/`](runs/) |
| Decisions, logs, and plotted results | [`docs/`](docs/) |

ScottyRL starter by Apurva Gandhi, Aditya Soni, and Demi Wang, with feedback from CMU 11-768 course staff.
