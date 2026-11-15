"""Tests for scripts/analyze_p5.py (ours, Day 6).

The transcripts are produced by the real environment, agent and logger with a
scripted policy, so the analysis is checked against the file format the
training run actually wrote.
"""

import asyncio
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

from scottyrl.agents.xml_tool_call import XMLToolCallAgent
from scottyrl.envs import make_env_factory
from scottyrl.envs.textcraft.recipe_book import default_recipe_book
from scottyrl.metrics import MetricsLogger
from scottyrl.policy import MockPolicy
from scottyrl.rollout import rollout_phase, run_episode
from scottyrl.types import load_tasks

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("analyze_p5", REPO / "scripts" / "analyze_p5.py")
analyze_p5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_p5)

THINK = "I should look this up.\n</think>\n\n"


def smallest_task():
    tasks = load_tasks(REPO / "data" / "textcraft_val.jsonl")
    return min(tasks, key=lambda t: t.info["num_craft_steps"])


def solving_script(task) -> list[str]:
    """One lookup, then every needed item crafted deepest first, then finish."""
    book = default_recipe_book()
    (target, count), = task.info["target_items"].items()
    need = Counter({target: count})
    order, stack = [], [target]
    while stack:
        item = stack.pop()
        if item in book.recipes and item not in order:
            order.append(item)
            stack.extend(book.recipes[item][0]["ingredients"])
    order.sort(key=book.crafting_depth, reverse=True)
    crafts = []
    for item in order:
        recipe = book.recipes[item][0]
        n = math.ceil(need[item] / recipe["result_count"])
        used = {ing: c * n for ing, c in recipe["ingredients"].items()}
        for ing, c in used.items():
            need[ing] += c
        crafts.append(f"<tool>craft({json.dumps(used)}, ({item!r}, {n * recipe['result_count']}))</tool>")
    crafts.reverse()  # ingredients before the items made from them
    return (
        [THINK + f"<tool>get_info({json.dumps(order)})</tool>"]
        + [THINK + c for c in crafts]
        + [THINK + '<tool>finish("done")</tool>']
    )


def write_transcripts(tmp_path, scripts_by_phase):
    logger = MetricsLogger(tmp_path / "p5.jsonl", trajectory_log_every=5)
    task = smallest_task()
    env_factory = make_env_factory("textcraft")

    async def play():
        for phase, script in scripts_by_phase:
            with rollout_phase(phase):
                await run_episode(XMLToolCallAgent(MockPolicy(script)), env_factory(task))

    asyncio.run(play())
    logger.log(5, {"train/num_groups": 1})
    logger.close()
    return tmp_path / "p5_trajectories", task


def test_solved_and_failed_episodes_are_classified(tmp_path):
    task = smallest_task()
    solve = solving_script(task)
    lookup = solve[0]
    stuck = [
        lookup,
        lookup,                                              # the same call again
        "I keep thinking and never close the tag",           # no </think>: no tool call
        THINK + "<tool>craft({'nope': 1}, ('nope', 1))</tool>",  # no such recipe
        THINK + "<tool>view_inventory()</tool>",
    ]
    root, task = write_transcripts(tmp_path, [("train", solve), ("train", stuck), ("eval", solve)])

    files = sorted(root.glob("step_000005/*.jsonl"))
    assert len(files) == 3
    episodes = {}
    for path in files:
        header, turns = analyze_p5.read_episode(path)
        episodes[path.name] = analyze_p5.analyse_episode(header, turns)

    solved = [e for e in episodes.values() if e["solved"]]
    failed = [e for e in episodes.values() if not e["solved"]]
    assert len(solved) == 2 and len(failed) == 1

    good = solved[0]
    n = task.info["num_craft_steps"]
    assert good["outcome"] == "solved"
    assert good["kinds"]["craft_ok"] == n and good["kinds"]["lookup"] == 1 and good["kinds"]["finish"] == 1
    assert good["progress"] == 1.0 and good["turns"] == n + 2
    assert good["turns_per_item"] == (n + 2) / n

    bad = failed[0]
    assert bad["outcome"] == "out_of_turns" and bad["turns"] == task.max_steps
    assert bad["kinds"]["no_tool_call"] == 1
    assert bad["details"]["craft_rejected:no_recipe"] == 1
    assert bad["progress"] == 0.0
    # everything after the second turn repeats an earlier call or is the same inventory check
    assert bad["repeats"] >= task.max_steps - 5
    assert bad["relookups"] == n  # the second lookup asked for the same items


def test_the_script_writes_a_summary(tmp_path, monkeypatch, capsys):
    task = smallest_task()
    root, _ = write_transcripts(tmp_path, [("train", solving_script(task)), ("eval", ["no call"])])
    out = tmp_path / "analysis.json"
    monkeypatch.setattr("sys.argv", ["analyze_p5.py", "--trajectories", str(root), "--out", str(out)])
    analyze_p5.main()
    result = json.loads(out.read_text())
    assert result["files"] == 2
    assert result["train"]["5"]["success"] == 1.0 and result["eval"]["5"]["success"] == 0.0
    assert result["eval"]["5"]["outcomes"] == {"out_of_turns": 1}
    assert result["train_pooled"]["early_5-10"]["episodes"] == 1
    assert "episodes read" in capsys.readouterr().out
