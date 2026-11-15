"""Start the Part 5 evaluation on Modal and return immediately (ours, Day 5).

The evaluation executes in the deployed `scottyrl-trainer` app
(`evaluate` in skyrl_tinker_modal/run_trainer.py), next to the
`scottyrl-evaluation` GPU server, so it does not depend on the Codespace or
on the Codespace's connection to Modal. Each finished episode is saved on the
`scottyrl-trainer-runs` volume; launching the same command again continues.

Examples (from the repo root, after both apps are deployed):
  # 2 tasks x 1 attempt on the adapter: checks the whole path in a few minutes
  uv run python scripts/launch_eval_on_modal.py --model a3 --tag probe_eval --k 1 --num-tasks 2
  # the reduced evaluation: the adapter, then the base model
  uv run python scripts/launch_eval_on_modal.py --model a3 --model Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p5_textcraft_grpo.yaml",
                        help="YAML config, relative to the repo root")
    parser.add_argument("--model", dest="models", action="append", required=True,
                        help="served model name; repeat to evaluate several in order")
    parser.add_argument("--tag", default="p5_eval",
                        help="prefix of the result files on the volume")
    parser.add_argument("--k", type=int, default=4, help="attempts per task")
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--min-rps", type=float, default=None,
                        help="speed check threshold (default: concurrency / 10; 0 disables)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE", help="override a config field")
    parser.add_argument("--server-app", default="scottyrl-evaluation")
    parser.add_argument("--time-cap-hours", type=float, default=3.0)
    args = parser.parse_args()

    if not (REPO / args.config).exists():
        raise SystemExit(f"no such config: {args.config}")

    server_url = modal.Function.from_name(args.server_app, "serve").get_web_url()
    evaluate = modal.Function.from_name("scottyrl-trainer", "evaluate")
    call = evaluate.spawn(
        config=args.config,
        server_url=server_url,
        models=args.models,
        tag=args.tag,
        k=args.k,
        num_tasks=args.num_tasks,
        concurrency=args.concurrency,
        min_rps=args.min_rps,
        overrides=args.overrides,
        time_cap_s=int(args.time_cap_hours * 3600),
    )
    print(f"Evaluation launched on Modal against {args.server_app}: {', '.join(args.models)}")
    print(f"Call id: {call.object_id}")
    print("Watch:  uv run modal app logs -f scottyrl-trainer   (Ctrl-C stops watching only)")
    print("Stop:   uv run modal app stop scottyrl-trainer -y")


if __name__ == "__main__":
    main()
