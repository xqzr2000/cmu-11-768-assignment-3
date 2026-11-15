"""Summarize the Part 5 TextCraft transcripts (ours, Day 6). No GPU, no network.

Reads every episode under runs/p5_textcraft_grpo_trajectories/ (written by
MetricsLogger: one directory per logged step, one file per episode) and writes
runs/p5_analysis.json, small enough to commit, plus a short table on screen.

    uv run python scripts/analyze_p5.py

What it measures, per logged step and phase ("train" rollouts, "eval" rollouts):

  outcomes     how each episode ended: solved; out of turns; out of turns with
               the target already crafted (finish() never called); finished
               without the target; abandoned (a lost sampling call).
  turns        what each turn was: a recipe lookup, a craft that worked, a
               craft the environment rejected (and why), an inventory check,
               finish, or a completion with no usable tool call (and whether
               it ran out of tokens while still thinking).
  habits       repeated calls, items per lookup, lookups of items already
               looked up, turns spent per crafted item.
  progress     how much of a task's recipe tree a failed episode completed.
  thinking     tokens per turn, for solved and unsolved episodes, and the
               same comparison between attempts at the SAME task in the same
               step, which removes task difficulty from it.
  size         success by the number of items a task needs crafted
               (`num_craft_steps` in the task data: one craft call each at
               the very least).

In an asynchronous run a step's directory holds the training episodes that
finished during that iteration, sampled by the last few policy versions, and
only every fifth step was saved. Read "step" as a point in training, not as
one exact policy.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

MAX_TOKENS = 512  # per turn, from configs/p5_textcraft_grpo.yaml
SIZE_BINS = [(0, 14), (15, 22), (23, 31), (32, 999)]

CRAFT_ERRORS = [  # first match wins; text the environment returns
    ("wrong_amount", "Wrong amount of"),
    ("missing_ingredient", "Missing required ingredient"),
    ("extra_ingredient", "Extra ingredients not required"),
    ("not_in_inventory", "Insufficient ingredients in inventory"),
    ("count_not_divisible", "is not divisible by recipe result count"),
    ("no_recipe", "No recipe found for"),
    ("bad_arguments", "must be a"),
    ("bad_arguments", "Invalid call"),
]


def size_bin(n: int) -> str:
    for lo, hi in SIZE_BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 999 else f"{lo}+"
    return "?"


def read_episode(path: Path) -> tuple[dict, list[dict]] | None:
    header, turns = None, []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") == "task":
            header = row
        elif row.get("type") == "step":
            turns.append(row)
    return (header, turns) if header else None


def classify_turn(turn: dict) -> tuple[str, str | None]:
    """(kind, detail) of one turn."""
    if "stop_reason" in turn:
        return "abandoned", None
    env = turn.get("env") or ""
    call = turn.get("tool_call")
    if call is None:
        text = turn.get("agent") or ""
        tokens = turn.get("completion_tokens") or 0
        if "</think>" not in text:
            return "no_tool_call", "still_thinking_at_token_limit" if tokens >= MAX_TOKENS - 1 else "no_end_of_thinking"
        return "no_tool_call", "malformed_after_thinking"
    name = call.get("name")
    if name == "get_info":
        return ("lookup", None) if not env.startswith("Invalid call") else ("invalid_call", "get_info")
    if name == "view_inventory":
        return ("inventory", None) if "takes no arguments" not in env else ("invalid_call", "view_inventory")
    if name == "finish":
        return ("finish", None) if not env.startswith("Invalid call") else ("invalid_call", "finish")
    if name == "craft":
        if env.startswith("Successfully crafted"):
            return "craft_ok", None
        for label, needle in CRAFT_ERRORS:
            if needle in env:
                return "craft_rejected", label
        return "craft_rejected", "other"
    return "invalid_call", "unknown_tool"


def analyse_episode(header: dict, turns: list[dict]) -> dict:
    info = header.get("info") or {}
    targets = info.get("target_items") or {}
    kinds: Counter = Counter()
    details: Counter = Counter()
    crafted: Counter = Counter()
    seen_calls: set[str] = set()
    looked_up: set[str] = set()
    repeats = relookups = lookup_items = lookups = 0
    tokens: list[int] = []
    think_chars: list[int] = []
    for turn in turns:
        kind, detail = classify_turn(turn)
        kinds[kind] += 1
        if detail:
            details[f"{kind}:{detail}"] += 1
        if turn.get("completion_tokens") is not None:
            tokens.append(int(turn["completion_tokens"]))
        text = turn.get("agent") or ""
        think_chars.append(len(text.split("</think>")[0]) if "</think>" in text else len(text))
        call = turn.get("tool_call") or {}
        raw = re.sub(r"\s+", "", call.get("raw") or "")
        if raw:
            repeats += raw in seen_calls
            seen_calls.add(raw)
        args = call.get("args") or []
        if kind == "lookup" and args and isinstance(args[0], list):
            lookups += 1
            lookup_items += len(args[0])
            relookups += sum(1 for item in args[0] if item in looked_up)
            looked_up.update(str(item) for item in args[0])
        if kind == "craft_ok" and len(args) == 2 and isinstance(args[1], list) and len(args[1]) == 2:
            crafted[str(args[1][0])] += int(args[1][1])

    last_env = (turns[-1].get("env") or "") if turns else ""
    solved = float(header.get("reward") or 0) >= 1.0
    has_target = bool(targets) and all(crafted[item] >= need for item, need in targets.items())
    if solved:
        outcome = "solved"
    elif turns and "stop_reason" in turns[-1]:
        outcome = "abandoned"
    elif last_env.startswith("Finished, but targets not met"):
        outcome = "finished_without_target"
    elif has_target:
        outcome = "out_of_turns_target_crafted"
    else:
        outcome = "out_of_turns"

    size = int(info.get("num_craft_steps") or 0)
    return {
        "task_id": header.get("task_id"),
        "solved": solved,
        "outcome": outcome,
        "turns": len(turns),
        "size": size,
        "progress": min(1.0, len(crafted) / size) if size else None,
        "kinds": kinds,
        "details": details,
        "repeats": repeats,
        "lookups": lookups,
        "lookup_items": lookup_items,
        "relookups": relookups,
        "tokens_per_turn": sum(tokens) / len(tokens) if tokens else None,
        "tokens_first10": sum(tokens[:10]) / len(tokens[:10]) if tokens else None,
        "think_chars_per_turn": sum(think_chars) / len(think_chars) if think_chars else None,
        "turns_per_item": len(turns) / len(crafted) if solved and crafted else None,
        "policy_version": header.get("policy_version"),
    }


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarise(episodes: list[dict]) -> dict:
    n = len(episodes)
    turns_total = sum(e["turns"] for e in episodes) or 1
    kinds: Counter = Counter()
    details: Counter = Counter()
    for e in episodes:
        kinds.update(e["kinds"])
        details.update(e["details"])
    solved = [e for e in episodes if e["solved"]]
    failed = [e for e in episodes if not e["solved"]]

    # Attempts at the same task that ended differently: the task is held fixed.
    by_task: dict = defaultdict(list)
    for e in episodes:
        by_task[e["task_id"]].append(e)
    diffs, diffs10 = [], []
    for group in by_task.values():
        ok = [e for e in group if e["solved"]]
        bad = [e for e in group if not e["solved"]]
        if ok and bad:
            a, b = mean(e["tokens_per_turn"] for e in ok), mean(e["tokens_per_turn"] for e in bad)
            if a is not None and b is not None:
                diffs.append(a - b)
            a, b = mean(e["tokens_first10"] for e in ok), mean(e["tokens_first10"] for e in bad)
            if a is not None and b is not None:
                diffs10.append(a - b)

    by_size = {}
    for lo, hi in SIZE_BINS:
        label = size_bin(lo)
        group = [e for e in episodes if size_bin(e["size"]) == label]
        if group:
            by_size[label] = {
                "episodes": len(group),
                "success": mean(float(e["solved"]) for e in group),
                "progress_when_failed": mean(e["progress"] for e in group if not e["solved"]),
                "turns_when_solved": mean(e["turns"] for e in group if e["solved"]),
            }

    versions = [e["policy_version"] for e in episodes if isinstance(e["policy_version"], int)]
    return {
        "episodes": n,
        "tasks": len(by_task),
        "success": mean(float(e["solved"]) for e in episodes),
        "policy_versions": [min(versions), max(versions)] if versions else None,
        "outcomes": dict(Counter(e["outcome"] for e in episodes)),
        "avg_turns": mean(e["turns"] for e in episodes),
        "turn_share": {k: round(v / turns_total, 4) for k, v in sorted(kinds.items())},
        "turn_detail_share": {k: round(v / turns_total, 4) for k, v in sorted(details.items())},
        "repeated_call_share": round(sum(e["repeats"] for e in episodes) / turns_total, 4),
        "items_per_lookup": round(
            sum(e["lookup_items"] for e in episodes) / max(1, sum(e["lookups"] for e in episodes)), 2
        ),
        "relookup_share_of_items": round(
            sum(e["relookups"] for e in episodes) / max(1, sum(e["lookup_items"] for e in episodes)), 4
        ),
        "turns_per_item_when_solved": mean(e["turns_per_item"] for e in solved),
        "progress_when_failed": mean(e["progress"] for e in failed),
        "tokens_per_turn": {
            "all": mean(e["tokens_per_turn"] for e in episodes),
            "solved": mean(e["tokens_per_turn"] for e in solved),
            "failed": mean(e["tokens_per_turn"] for e in failed),
            "first10_solved": mean(e["tokens_first10"] for e in solved),
            "first10_failed": mean(e["tokens_first10"] for e in failed),
        },
        "think_chars_per_turn": mean(e["think_chars_per_turn"] for e in episodes),
        "same_task_solved_minus_failed_tokens": {
            "tasks_with_both": len(diffs),
            "mean_difference": mean(diffs),
            "share_solved_used_fewer": mean(float(d < 0) for d in diffs),
            "first10_mean_difference": mean(diffs10),
            "first10_share_solved_used_fewer": mean(float(d < 0) for d in diffs10),
        },
        "by_size": by_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trajectories", default="runs/p5_textcraft_grpo_trajectories")
    parser.add_argument("--out", default="runs/p5_analysis.json")
    args = parser.parse_args()

    root = Path(args.trajectories)
    step_dirs = sorted(p for p in root.glob("step_*") if p.is_dir())
    if not step_dirs:
        raise SystemExit(f"no step_* directories under {root}")

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    unreadable = 0
    for step_dir in step_dirs:
        step = int(step_dir.name.split("_")[1])
        for path in sorted(step_dir.glob("*.jsonl")):
            episode = read_episode(path)
            if episode is None:
                unreadable += 1
                continue
            header, turns = episode
            phase = header.get("phase") or path.name.split("_")[0]
            groups[(phase, step)].append(analyse_episode(header, turns))

    result = {
        "source": str(root),
        "files": sum(len(v) for v in groups.values()),
        "unreadable_files": unreadable,
        "size_bins_are": "items the task needs crafted (num_craft_steps)",
        "train": {}, "eval": {},
    }
    for (phase, step), episodes in sorted(groups.items()):
        result.setdefault(phase, {})[str(step)] = summarise(episodes)
    # Thirds of training, pooled, for steadier numbers than single steps.
    for label, lo, hi in (("early_5-10", 1, 10), ("middle_15-25", 11, 25), ("late_30-40", 26, 99)):
        pooled = [e for (phase, step), eps in groups.items() if phase == "train" and lo <= step <= hi for e in eps]
        if pooled:
            result.setdefault("train_pooled", {})[label] = summarise(pooled)

    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")

    print(f"{result['files']} episodes read ({unreadable} unreadable); wrote {args.out}\n")
    print("phase step  episodes success  avg turns | lookup craft_ok rejected no_call repeated | tokens/turn")
    for phase in ("eval", "train"):
        for step, s in sorted(result.get(phase, {}).items(), key=lambda kv: int(kv[0])):
            share = s["turn_share"]
            print(
                f"{phase:5s} {int(step):4d} {s['episodes']:9d} {s['success']:7.2f} {s['avg_turns']:10.1f} | "
                f"{share.get('lookup', 0):6.2f} {share.get('craft_ok', 0):8.2f} "
                f"{share.get('craft_rejected', 0):8.2f} {share.get('no_tool_call', 0):7.2f} "
                f"{s['repeated_call_share']:8.2f} | {s['tokens_per_turn']['all']}"
            )
    print(f"\nNext: git add {args.out} && git commit -m 'Add Part 5 trajectory analysis' && git push")


if __name__ == "__main__":
    main()
