"""Evaluate a checkpoint, an exported adapter, or the base model on a
held-out split (given).

Graders run this on the shipped val splits against your exported Part 5
adapter, served with vLLM.

The model settings (base model, chat template, thinking mode, max_tokens,
temperature, stop sequences) come from the same YAML config the run was
trained with, so evaluation renders prompts exactly like training did. The
environment and agent come from the config's `train` section. Any field can
be overridden with `--set`, as in the training scripts.

Samples come from an OpenAI-compatible completions server (vLLM, SGLang).
Pass its URL with --server and the served model name with --model. A trailing
/v1 is added automatically if needed.

Examples:
  # An adapter exported with scripts/export_checkpoint.py, served by vLLM:
  vllm serve Qwen/Qwen3.5-4B --enable-lora --max-lora-rank 32 \\
      --lora-modules a3=<hf_model_id>
  python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \\
      --server http://localhost:8000/v1 --model a3 --k 4
  # The same with SGLang (the adapter is named base:adapter in --model):
  python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --enable-lora \\
      --lora-paths a3=<hf_model_id> --max-lora-rank 32 --port 8000
  python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \\
      --server http://localhost:8000/v1 --model Qwen/Qwen3.5-4B:a3 --k 4

The printed JSON carries the results plus `model_path`, `server` and
`model`, so a saved output says what was evaluated.

Ours (Day 5, see scottyrl/eval_runner.py): with `--out FILE` every episode's
result is appended to FILE as it finishes, and running the same command again
continues from it. A progress line is printed every 30 seconds. If the server
answers fewer than `--min-rps` requests per second for about two minutes
while all slots are busy, the script stops with exit code 3 and keeps what it has.
`--summary FILE` also writes the final JSON (without the server URL) to FILE.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from scottyrl.agents import AGENT_REGISTRY
from scottyrl.config import load_config
from scottyrl.envs import make_env_factory
from scottyrl.eval_runner import EXIT_TOO_SLOW, TooSlow, evaluate_resumable
from scottyrl.policy import CompletionsServerPolicy, load_chat_template
from scottyrl.types import load_tasks

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="YAML config (see configs/)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="SECTION.KEY=VALUE", help="override a config field")
    p.add_argument("--server", required=True, metavar="URL",
                   help="OpenAI-compatible server URL, e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True, metavar="NAME",
                   help="served adapter or base-model name, e.g. a3 or Qwen/Qwen3.5-4B")
    p.add_argument("--tasks", default=None, help="jsonl path; defaults to the env's val split")
    p.add_argument("--k", type=int, default=None, help="rollouts per task (default: train.eval_k)")
    p.add_argument("--num-tasks", type=int, default=None, help="evaluate only the first N tasks")
    p.add_argument("--concurrency", type=int, default=32,
                   help="episodes run at a time (default 32)")
    p.add_argument("--out", default=None, metavar="FILE",
                   help="append each episode's result here; rerun to continue from it")
    p.add_argument("--summary", default=None, metavar="FILE",
                   help="also write the final JSON here")
    p.add_argument("--request-timeout", type=float, default=120.0,
                   help="seconds to wait for one sampling response before retrying")
    p.add_argument("--min-rps", type=float, default=None,
                   help="stop if the server answers fewer requests per second than this "
                        "for 2 minutes with all slots busy (default: concurrency / 10; "
                        "0 disables the check)")
    args = p.parse_args(argv)
    args.server = args.server.rstrip("/")
    if not args.server.endswith("/v1"):
        args.server += "/v1"
    return args


async def main() -> None:
    args = parse_args()

    config = load_config(args.config, args.overrides)
    model_cfg, train_cfg = config.model, config.train
    k = args.k if args.k is not None else train_cfg.eval_k

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model)
    rendering = dict(
        max_tokens=model_cfg.max_tokens,
        temperature=model_cfg.temperature,
        chat_template=load_chat_template(model_cfg.chat_template),
        enable_thinking=model_cfg.enable_thinking,
        preserve_thinking_history=model_cfg.preserve_thinking_history,
        stop_sequences=model_cfg.stop_sequences,
    )
    policy = CompletionsServerPolicy(
        args.server, args.model, tokenizer, timeout=args.request_timeout, **rendering
    )

    tasks = load_tasks(args.tasks or DATA_DIR / f"{train_cfg.env}_val.jsonl")
    if args.num_tasks:
        tasks = tasks[: args.num_tasks]

    await wait_for_server(policy)

    min_rps = args.concurrency / 10 if args.min_rps is None else args.min_rps
    try:
        results = await evaluate_resumable(
            policy,
            tasks,
            make_env_factory(train_cfg.env),
            k=k,
            concurrency=args.concurrency,
            results_path=args.out,
            meta={"model": args.model},
            agent_cls=AGENT_REGISTRY[train_cfg.agent],
            min_requests_per_s=min_rps or None,
            log=lambda line: print(line, file=sys.stderr, flush=True),
        )
    except TooSlow as e:
        print(f"[eval] STOPPED, too slow: {e}", file=sys.stderr, flush=True)
        sys.exit(EXIT_TOO_SLOW)
    results["model_path"] = None
    results["model"] = args.model
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(results, indent=2) + "\n")
    results["server"] = args.server
    print(json.dumps(results, indent=2))


async def wait_for_server(policy: CompletionsServerPolicy, limit_s: float = 30 * 60) -> None:
    """Block until the server lists its models (a cold start takes minutes).

    Each try waits up to 2 minutes; failed tries (timeouts, dropped
    connections, 5xx while the container starts) are repeated until
    `limit_s` has passed. A line every try keeps the terminal alive.
    """
    started = time.monotonic()
    while True:
        waited = time.monotonic() - started
        print(f"Waiting for inference server... ({waited:.0f} s)", file=sys.stderr, flush=True)
        try:
            response = await policy._http().get("/models", timeout=120.0)
            if response.status_code == 200:
                print("Inference server ready.", file=sys.stderr, flush=True)
                return
            problem = f"HTTP {response.status_code}"
        except Exception as e:
            problem = type(e).__name__
        if time.monotonic() - started > limit_s:
            raise SystemExit(f"inference server not ready after {limit_s:.0f} s ({problem})")
        await asyncio.sleep(5)


if __name__ == "__main__":
    # Show the sampling client's retry warnings.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    asyncio.run(main())
