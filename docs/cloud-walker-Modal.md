# Cloud walker: Modal

A dated journal of renting H100s on Modal to host and train an open-weight model
(`Qwen/Qwen3.5-4B`) for Assignment 3. Newest entry at the bottom.

This log is about the infrastructure. The reinforcement learning itself is in
[Learning-Progress.md](Learning-Progress.md), and the commands for each day are in
[Runbook.md](Runbook.md).

## Where this sits after Mario_Ops

Mario_Ops trained a small network we wrote ourselves on an AWS machine we managed
ourselves. Assignment 3 trains someone else's 4-billion-parameter model on GPUs we never
see. Two things are new:

- **The provider.** AWS is a general cloud: you rent a machine and assemble the rest (image,
  storage, permissions, monitoring). "Neocloud" usually means the GPU-first providers such as
  CoreWeave, Lambda and RunPod, which still rent you a machine. Modal sits at the serverless
  end of that group: you describe a function and its GPUs in Python, and there is no machine
  to log in to.
- **The model.** Open weights mean the model file is ours to download, load, and change.
  That is what makes training possible at all, and it is also why we have to host it.

### AWS in Mario_Ops next to Modal here

The AWS column is read from `Mario_AWS/` in the Mario_Ops repo (its `Dockerfile.deploy`,
`cloud/storage.py`, `cloud/monitoring.py`). Russell: correct anything that does not match
what you actually ran.

| Concern | AWS in Mario_Ops | Modal in Assignment 3 |
|---|---|---|
| What you rent | An EC2 instance, chosen by type (64 vCPUs for the emulators; `g4dn`/`g5` for the GPU image) | A function with `gpu="H100:2"`, 8 CPUs and 32 GiB declared in code |
| Host preparation | Deep Learning AMI, NVIDIA container toolkit, a root volume of 100 GB or more | None. There is no host to prepare |
| Container image | `Dockerfile.deploy`, built with `docker build` and run with `docker run --gpus all` | A `modal.Image` chain written in Python, built on Modal's side and cached there |
| Where the training loop runs | On the instance, inside the container | In the Codespace. The GPUs are a remote service behind a URL |
| Persistent storage | S3 through `boto3`, keys scoped by run id | Modal Volumes mounted as ordinary directories (`/cache/huggingface`, `/state`) |
| Credentials | An IAM role on the instance; no keys in the image | A token pair (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) held as Codespaces secrets |
| Monitoring | CloudWatch custom metrics | Modal's dashboard for logs and spend; training metrics in `runs/*.jsonl` |
| Billing | Per instance, until you stop it | Per second while a container is up, plus a five-minute idle window |
| Interruption | A SIGTERM handler writes a last checkpoint to S3 on spot reclaim | A resumable checkpoint every 10 optimizer steps on the state volume; rerun with `--set train.resume=true` |
| Shutting down | Stop or terminate the instance | `modal app stop <app>`, or do nothing and it scales to zero |

The row that changes how the work feels is the fourth one. On AWS the code went to the
GPU. Here the GPU comes to the code as an HTTPS endpoint, and the Codespace never needs
PyTorch.

## How the Qwen model is hosted and trained

One Modal container holds everything that needs a GPU. All of this is in
`skyrl_tinker_modal/skyrl_server.py`, which the course provides and we use unchanged.

```
Codespace (no GPU)                         Modal container (2 or 4 H100s)
------------------                         ------------------------------
environments, agent,     Tinker API        SkyRL Tinker server, port 9000
trainers, tokenizer  --- over HTTPS --->     vLLM engines     sample tokens + log-probabilities
                                             Megatron         forward/backward, optimizer step
                                           volumes
                                             scottyrl-hf-cache   base weights, shared by every app
                                             <app>-state         checkpoints and adapter snapshots
```

- **Weights.** The base model is pulled from Hugging Face on the first start and stored on
  the `scottyrl-hf-cache` volume (`HF_HOME` points there). Every later start, and every other
  app, reuses that copy.
- **Two engines, one model.** vLLM serves the model for rollouts. Megatron trains it. In the
  synchronous layout both run on the same two GPUs: SkyRL puts vLLM to sleep during an update
  and wakes it with the new weights. The asynchronous layouts give each role its own GPUs.
- **What is trained.** A LoRA adapter of rank 32, not the 4B base weights. The base stays
  frozen on the cache volume.
- **What is saved.** After each update the server archives the adapter alone. The pinned
  SkyRL fork does this on purpose; stock SkyRL would export the full merged model, several GB
  per step.
- **Serving the result.** Part 5 evaluation skips the trainer entirely:
  `serve_evaluation.py` starts plain `vllm serve` with `--enable-lora` and loads the
  downloaded adapter next to the base model, so one endpoint answers as either.

## Journal

### Nov 8-9 - Choosing where to rent

**Decision:** Modal, on Russell's own account, with a ceiling of about $250.

- AWS was ruled out early. Its H100 instances (P5) were listed at about $6.88 per GPU-hour
  and come mainly as 8-GPU machines; we need 2 and 4.
- RunPod, Lambda and Vast.ai were cheaper per GPU-hour ($2.40 to $4.19 on Nov 8) but each
  meant rebuilding the deployment ourselves on a rented machine.
- Modal's H100 was $0.001097 per second on its pricing page, about $3.95 per hour. With the
  CPUs and memory the servers request, a 2xH100 server is about $8.50 per hour and the
  4xH100 server about $16.40.
- What tipped it: the course's launch scripts are written for Modal, and it bills nothing
  while idle. We judged that worth roughly $50-80 over the cheapest option.

Constraint worth remembering: the config sets `policy_num_nodes: 1`, so all GPUs for a run
must be in one machine. Two separate 1xH100 rentals would not work on any provider.

### Nov 9-10 - Account, credentials, image, first GPUs

**Account and credentials**

- Signed up on the free Starter plan ($30 of credit a month, up to 10 GPUs at once).
- Created an API token in Modal's settings. It is a pair: an ID starting `ak-` and a secret
  starting `as-`.
- Stored both as Codespaces secrets for this repo and restarted the Codespace. The Modal CLI
  reads them from the environment, so `modal setup` was never run.
- AWS comparison: there is no role or policy to write. One token carries the whole
  workspace, so it is simpler and also coarser than IAM.

**First failure: client older than the platform**

- `modal run` refused to start. The starter pinned `modal` 1.5.5, and a new workspace
  defaults to image builder `2026.10`, which that client does not accept.
- Fix: `uv lock --upgrade-package modal` (to 1.6.3), then `uv sync --extra modal`.
- Nothing was billed; the check fails on the client before any container starts.
- Lesson: on a managed platform the provider's side moves under a pinned repo. On EC2 the
  AMI and driver stay as you left them.

**Building the image**

- `uv run modal run serve_number_search_sync.py::smoke_install` built the image and ran a
  check on one H100. About 20 minutes, about $2.
- The image starts from `nvidia/cuda:12.8.1-devel-ubuntu22.04`, clones the pinned SkyRL
  commit, and runs `uv sync --extra tinker --extra megatron`.
- The build step itself requests an H100 so that an import check can load the CUDA
  libraries. A broken install fails the build, not the first training run.
- Check output: torch 2.11 on CUDA 12.8, vLLM 0.26.0, GPU `NVIDIA H100 80GB HBM3`.
- The image is cached on Modal and shared by all launch scripts, so this cost is paid once.
- AWS comparison: no `docker build`, no registry push, no pull onto a host with a large
  enough disk.

**Deploying**

- `uv run modal deploy serve_number_search_sync.py` finished in 1.7 seconds and printed a
  public URL of the form `https://<workspace>--scottyrl-ns-sync-tinker-server.us-east.modal.run`.
- Deploying only registers the function. No GPU is allocated, and nothing bills, until a
  request arrives.
- `modal run` gave a temporary address ending in `-dev`; `modal deploy` gives the stable one.
- Modal labels the URL "unauthenticated": anyone who has it can wake the GPUs. We keep it
  out of the repo, and stopping the app closes it.

**First request: cold start**

- The first training request waited several minutes with no output on the Codespace side.
  In that time Modal allocated two H100s, the container downloaded the Qwen weights into the
  cache volume, and vLLM and the trainer came up.
- The server allows up to 45 minutes for this (`startup_timeout`). Later starts skip the
  download.
- Where to watch: the app's page in the Modal dashboard shows the container logs.
- The Codespace downloaded only the tokenizer files. It warned that PyTorch was missing,
  which is correct and harmless.

**The probe on two H100s**

- 10 training steps of GRPO on Number Search, about 12 minutes of 2xH100.
- 30-70 seconds per step after warm-up. A checkpoint was written to the state volume at
  step 10.
- Sampling and training ran in different engines and still agreed: the importance weight
  averaged 1.000. The reading of that number is in Learning-Progress.md.
- Stopped with `uv run modal app stop scottyrl-ns-sync` as soon as it finished.

**Spend so far:** roughly $5 (estimate; the exact figure is on the Modal dashboard).

**Not yet tested**

- Three apps at once (6 H100s): do they all get capacity promptly?
- Whether Modal preempts a container during the 6-8 hour TextCraft run, and how resume behaves.
- 4xH100 availability in one container.
- Pulling an adapter back with `modal volume get` and serving it with vLLM.

## Entry template

```
### <date> - <what we ran>

**What we did**
**What Modal did that surprised us**
**AWS comparison**
**Qwen hosting or training note**
**Spend so far**
```
