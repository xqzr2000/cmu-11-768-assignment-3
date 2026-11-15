# Runbook

Commands for each day, run from the repo root in the Codespace unless a step says otherwise.
Replace `<...>` with your own values.

## Every session

- Start: `git pull`, then `uv run modal app list` to confirm nothing is running.
- End: stop every app, `uv run modal app list` again, commit and push run logs.
- `modal app stop` asks for confirmation; add `-y` to skip the prompt.
- A server URL is public to anyone who has it. Do not post it; stopping the app closes it.

- Check the Modal dashboard for apps still running before you start and before you leave.
- A server bills from the moment a run connects until five minutes after it goes idle.
  `modal app stop <app>` ends that at once.
- Start long runs inside `tmux` so they survive a closed browser tab:
  `tmux new -s run`, detach with `Ctrl-b d`, come back with `tmux attach -t run`.
  The Codespace image does not include it: `sudo apt-get update && sudo apt-get install -y tmux`.
  For runs under an hour, two VS Code terminal tabs are enough.
- `<... url>` in a command is a placeholder. Replace the whole thing, brackets included, with
  the URL that `modal deploy` printed; bash reads a bare `<` as a file redirect.
- Stop an app only after its run has printed `[checkpoints] wrote ...`. Stopping kills any
  run still using it.

## Day 1 - Setup, image, probe (~$5) - DONE

Completed Nov 10. Kept here for reference. One fix was needed first: the starter's Modal
client was too old for a new workspace (`uv lock --upgrade-package modal`, already committed).

1. Merge the open pull requests on GitHub, then create a Codespace on `main`.
2. Run the tests. All of them should pass.

   ```bash
   uv run pytest
   ```

3. Sign up at modal.com. In the Modal dashboard create an API token, then add
   `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as Codespaces secrets for this repo
   (GitHub > Settings > Codespaces > Secrets) and restart the Codespace.
4. Build the server image and check it. About 20 minutes on one H100, once.

   ```bash
   cd skyrl_tinker_modal
   uv run modal run serve_number_search_sync.py::smoke_install
   ```

5. Deploy the sync server and run a 10-step probe.

   ```bash
   uv run modal deploy serve_number_search_sync.py     # prints the URL
   cd ..
   export TINKER_BASE_URL=<printed url>
   uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml \
       --set train.num_steps=10 --set train.eval_every=2 --set train.eval_tasks=20 \
       --set train.run_name=probe_p2b
   uv run modal app stop scottyrl-ns-sync
   ```

   The first request takes several minutes while the model loads. Healthy probe: a step-0
   evaluation, one training record per step, and `train/importance_weight_mean` close to 1.
   Paste the terminal output to Claude if anything looks off.

## Day 2 - Synchronous runs (~$30)

Three servers, three terminals (or three tmux windows), one run each, about 40 minutes.

```bash
cd skyrl_tinker_modal
SKYRL_APP_NAME=scottyrl-p2a uv run modal deploy serve_number_search_sync.py
SKYRL_APP_NAME=scottyrl-p2b uv run modal deploy serve_number_search_sync.py
SKYRL_APP_NAME=scottyrl-p41 uv run modal deploy serve_number_search_sync.py
cd ..
```

```bash
# terminal 1
TINKER_BASE_URL=<p2a url> uv run python scripts/train_sync.py --config configs/p2a_number_search_reinforce.yaml
# terminal 2
TINKER_BASE_URL=<p2b url> uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml
# terminal 3
TINKER_BASE_URL=<p41 url> uv run python scripts/train_sync.py --config configs/p41_number_search_grpo_g2.yaml
```

When each finishes:

```bash
uv run modal app stop scottyrl-p2a
uv run modal app stop scottyrl-p2b
uv run modal app stop scottyrl-p41
git add runs/*.jsonl runs/*.json && git commit -m "Add Part 2 and 4.1 run logs" && git push
```

If a run crashes, rerun the same command with `--set train.resume=true` against the same app.

## Day 3 - Asynchronous runs (~$30) - DONE

Completed Nov 12: `p3` with its probe on Nov 11, both `p43` runs with a probe each on Nov 12,
on two servers (`scottyrl-p43iw`, `scottyrl-p43noiw`). Kept here for reference.

Probe first on one async server (about 10 minutes), then the three runs in parallel.

```bash
cd skyrl_tinker_modal
SKYRL_APP_NAME=scottyrl-p3 uv run modal deploy serve_number_search_async.py
SKYRL_APP_NAME=scottyrl-p43iw uv run modal deploy serve_number_search_async.py
SKYRL_APP_NAME=scottyrl-p43noiw uv run modal deploy serve_number_search_async.py
cd ..

TINKER_BASE_URL=<p3 url> uv run python scripts/train_async.py --config configs/p3_number_search_async.yaml \
    --set train.num_steps=10 --set train.eval_every=5 --set train.eval_tasks=20 \
    --set train.run_name=probe_p3
```

```bash
# terminal 1
TINKER_BASE_URL=<p3 url> uv run python scripts/train_async.py --config configs/p3_number_search_async.yaml
# terminal 2
TINKER_BASE_URL=<p43iw url> uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_iw.yaml
# terminal 3
TINKER_BASE_URL=<p43noiw url> uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_noiw.yaml
```

Stop the three apps and push the logs as on Day 2.

## Day 4 - TextCraft on Modal, detached (~$100-130) - DONE

Ran Nov 12-13, stopped at step 41 of 50; see `docs/Learning-Progress.md`. Lessons for any rerun:
`run_trainer.py` asks for 4 cores, 16 GiB and `nonpreemptible=True`, which Modal bills at three
times the base rate ($10.90 for the night); the trainer barely uses its CPU, so 2 cores and 8
GiB would do. If requests start getting stuck (`No progress made in 7200s`), stop both apps
and redeploy the GPU server before resuming, instead of retrying on the same server.


Only after Day 3 looked healthy. The trainer runs on Modal too, in a CPU-only container
(`skyrl_tinker_modal/run_trainer.py`) next to the 4xH100 server, so the Codespace can be
stopped once the full run is launched. Logs go to the `scottyrl-trainer-runs` volume.

The runner retries a crashed trainer twice (resuming from the last 10-step checkpoint, or
starting over if it died before step 10) and stops the run at 12 hours with SIGINT, so the
checkpoint manifest is still written. A trainer that prints nothing for 60 minutes is treated
as hung and handled like a crash, so a hang cannot keep the GPU server billing for 12 hours.

**Tonight, in the Codespace**

```bash
git pull
uv run pytest tests/test_modal_runner.py

# deploy both apps; the second builds a small CPU image the first time (a few minutes)
cd skyrl_tinker_modal && uv run modal deploy serve_textcraft.py && cd ..
uv run modal deploy skyrl_tinker_modal/run_trainer.py

# probe, 2 steps, about 20-30 minutes, most of it the 4xH100 cold start
uv run python scripts/launch_on_modal.py --config configs/p5_textcraft_grpo.yaml \
    --set train.num_steps=2 --set train.eval_every=0 --set train.run_name=probe_p5
uv run modal app logs -f scottyrl-trainer
```

Wait for `[runner] finished probe_p5 with exit code 0` and paste the output to Claude. If it
looks healthy, launch the real run on the same warm server:

```bash
uv run python scripts/launch_on_modal.py --config configs/p5_textcraft_grpo.yaml
uv run modal app logs -f scottyrl-trainer     # see it start, then Ctrl-C (watching only)
```

Then stop the Codespace. The run does not need it.

**In the morning**

```bash
uv run modal app logs scottyrl-trainer        # last 100 lines; look for "[runner] finished"
uv run modal volume ls scottyrl-trainer-runs runs
for f in p5_textcraft_grpo.jsonl p5_textcraft_grpo.resume.json \
         p5_textcraft_grpo.checkpoints.json p5_textcraft_grpo.console.log; do
  uv run modal volume get scottyrl-trainer-runs runs/$f runs/$f
done
uv run modal volume get scottyrl-trainer-runs runs/p5_textcraft_grpo_trajectories runs/
uv run modal app stop scottyrl-textcraft -y
uv run modal app list
git add runs/p5_textcraft_grpo.jsonl runs/p5_textcraft_grpo.resume.json \
        runs/p5_textcraft_grpo.checkpoints.json
git commit -m "Add Part 5 run log" && git push
```

If the runner gave up (no retries left, or the 12-hour cap), continue it with
`uv run python scripts/launch_on_modal.py --config configs/p5_textcraft_grpo.yaml --resume`.

Emergency stop at any time: `uv run modal app stop scottyrl-trainer -y` (kills the trainer;
redeploy `run_trainer.py` before launching again), then `uv run modal app stop scottyrl-textcraft -y`.

## Day 5 - Evaluation, reduced (done Nov 14, about $8-9 for the second attempt)

The step-41 model and the base model on the first 50 validation tasks, 4 attempts each: the
same tasks and settings as the training run's evaluations, so the results extend its curve
(step 0: 0.115, step 30: 0.500). See "Adapted plan" in `Plan.md`.

The first attempt (Nov 13-14) ran from the Codespace and stalled; see the Nov 13-14 entry in
`cloud-walker-Modal.md`. The evaluation now runs in a small Modal container next to the GPU
server, saves every finished episode, and stops by itself if the server is too slow.

**1. Deploy both apps** (the adapter in `runs/p5_adapter` is still in the Codespace from last time)

```bash
git pull
ls runs/p5_adapter                      # adapter_config.json, adapter_model.safetensors
cd skyrl_tinker_modal
LORA_ADAPTER=../runs/p5_adapter uv run modal deploy serve_evaluation.py
cd ..
uv run modal deploy skyrl_tinker_modal/run_trainer.py
```

If `runs/p5_adapter` is missing, download it again first:

```bash
uv run modal volume get scottyrl-textcraft-state \
    checkpoints/model_367b3579/sampler_weights/v41_r2.tar.gz runs/p5_adapter.tar.gz
mkdir -p runs/p5_adapter && tar -xf runs/p5_adapter.tar.gz -C runs/p5_adapter
```

**2. Probe: 2 episodes** (about 10 minutes, most of it the server's cold start; billing starts here)

```bash
uv run python scripts/launch_eval_on_modal.py --model a3 --tag probe_eval --k 1 --num-tasks 2
uv run modal app logs -f scottyrl-trainer
```

Wait for `[runner] finished probe_eval` and paste the output to Claude.

**3. The evaluation** (on the same warm server; the Codespace is not needed while it runs)

```bash
uv run python scripts/launch_eval_on_modal.py --model a3 --model Qwen/Qwen3.5-4B
uv run modal app logs -f scottyrl-trainer
```

What to expect:

- A line every 30 seconds: `[eval] 37/200 episodes, 150 s | 2310 requests, 15.4/s, mean 2.0 s, ...`.
- The speed check. About two and a half minutes in, and from then on, the run stops by itself
  (`STOPPED, too slow`) if it gets fewer than 3.2 requests per second while all 32 slots are
  busy. Finished episodes stay saved; launching the same command again continues from them.
- A few `Sampling request failed ...; retrying` lines are harmless. A steady stream is worth
  pasting to Claude, with the progress line next to it.
- `[runner] finished p5_eval: all 2 models evaluated` at the end. Roughly 15-40 minutes per model.

**4. Download, stop, commit**

```bash
for f in p5_eval_a3.json p5_eval_Qwen_Qwen3.5-4B.json \
         p5_eval_a3.jsonl p5_eval_Qwen_Qwen3.5-4B.jsonl; do
  uv run modal volume get scottyrl-trainer-runs runs/$f runs/$f
done
cat runs/p5_eval_a3.json runs/p5_eval_Qwen_Qwen3.5-4B.json
uv run modal app stop scottyrl-evaluation -y
uv run modal app list

# trajectories for the analysis (not committed; gitignored). The destination is the folder
# to put the directory IN: naming the directory itself fails with "Is a directory".
uv run modal volume get scottyrl-trainer-runs runs/p5_textcraft_grpo_trajectories runs/

git add runs/p5_eval_*.json runs/p5_eval_*.jsonl
git commit -m "Add Part 5 evaluation results" && git push
```

In each result, `abandoned_episodes` should be 0; anything else means that score is too low.
Emergency stop: `uv run modal app stop scottyrl-trainer -y`, then
`uv run modal app stop scottyrl-evaluation -y` (redeploy both before launching again).

## Day 6-7 - Analysis and report

Tell Claude the evaluation results are pushed and the trajectories are in the Codespace.
Claude runs the analyses listed under "Adapted plan" in `Plan.md`, then builds the plots and
`docs/Report.md`: RL findings and the Modal experience side by side.
