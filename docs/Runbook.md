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

## Day 4 - TextCraft on Modal, detached (~$100-130)

Only after Day 3 looked healthy. The trainer runs on Modal too, in a CPU-only container
(`skyrl_tinker_modal/run_trainer.py`) next to the 4xH100 server, so the Codespace can be
stopped once the full run is launched. Logs go to the `scottyrl-trainer-runs` volume.

The runner retries a crashed trainer twice (resuming from the last 10-step checkpoint, or
starting over if it died before step 10) and stops the run at 12 hours with SIGINT, so the
checkpoint manifest is still written.

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
uv run modal volume get scottyrl-trainer-runs runs/p5_textcraft_grpo_trajectories runs/p5_textcraft_grpo_trajectories
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

## Day 5 - Evaluation (~$10-20)

Find the final snapshot, download it from the Modal volume, serve it, evaluate.

```bash
grep final_sampler_path runs/p5_textcraft_grpo.checkpoints.json   # tinker://<model-id>/<snapshot>

mkdir -p runs/p5_adapter
uv run modal volume get scottyrl-textcraft-state \
    checkpoints/<model-id>/sampler_weights/<snapshot>.tar.gz runs/p5_adapter.tar.gz
tar -xf runs/p5_adapter.tar.gz -C runs/p5_adapter

cd skyrl_tinker_modal
LORA_ADAPTER=../runs/p5_adapter uv run modal deploy serve_evaluation.py
cd ..
export EVAL_SERVER=<printed url>

for model in a3 Qwen/Qwen3.5-4B; do
  uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
      --server "$EVAL_SERVER" --model "$model" --k 4 | tee -a runs/p5_eval.txt
  uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
      --server "$EVAL_SERVER" --model "$model" --k 4 --tasks data/textcraft_val_hard.jsonl | tee -a runs/p5_eval.txt
done

uv run modal app stop scottyrl-evaluation
git add -f runs/p5_eval.txt && git commit -m "Add Part 5 evaluation results" && git push
```

## Day 6 - Report

Tell Claude the logs are pushed. Claude builds the plots and `docs/Report.md` from them.
