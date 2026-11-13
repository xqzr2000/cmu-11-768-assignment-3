"""Start a training run on Modal and return immediately (ours, Day 4).

The run executes in the deployed `scottyrl-trainer` app
(skyrl_tinker_modal/run_trainer.py), so the Codespace can be stopped as soon
as this script prints the call id.

Examples (from the repo root, after both apps are deployed):
  uv run python scripts/launch_on_modal.py --config configs/p5_textcraft_grpo.yaml \
      --set train.num_steps=2 --set train.eval_every=0 --set train.run_name=probe_p5
  uv run python scripts/launch_on_modal.py --config configs/p5_textcraft_grpo.yaml
  uv run python scripts/launch_on_modal.py --config configs/p5_textcraft_grpo.yaml --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config, relative to the repo root")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE", help="override a config field")
    parser.add_argument("--resume", action="store_true",
                        help="continue a run that already has results on the volume")
    parser.add_argument("--server-app", default="scottyrl-textcraft",
                        help="deployed GPU server app to train against")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--time-cap-hours", type=float, default=12.0)
    args = parser.parse_args()

    config = Path(args.config)
    if config.is_absolute():
        config = config.relative_to(REPO)
    if not (REPO / config).exists():
        raise SystemExit(f"no such config: {config}")

    server_url = modal.Function.from_name(args.server_app, "tinker_server").get_web_url()
    train = modal.Function.from_name("scottyrl-trainer", "train")
    call = train.spawn(
        config=config.as_posix(),
        server_url=server_url,
        overrides=args.overrides,
        resume=args.resume,
        max_retries=args.max_retries,
        time_cap_s=int(args.time_cap_hours * 3600),
    )
    print(f"Launched on Modal against {args.server_app}.")
    print(f"Call id: {call.object_id}")
    print("Watch:  uv run modal app logs -f scottyrl-trainer   (Ctrl-C stops watching only)")
    print("Stop:   uv run modal app stop scottyrl-trainer")


if __name__ == "__main__":
    main()
