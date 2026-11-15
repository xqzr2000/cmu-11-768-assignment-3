"""Draw the figures of docs/Report.md from the logs in runs/ (ours, Day 7).

    uv run --with matplotlib python scripts/make_report_plots.py

Reads runs/*.jsonl (the seven training logs), runs/p5_eval_*.jsonl (the Day 5
evaluation) and runs/p5_analysis.json (the Day 6 transcript summary), and
writes PNG files to docs/figures/. No GPU, no network. matplotlib is not a
project dependency; `--with matplotlib` supplies it for this one command.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
OUT = REPO / "docs" / "figures"

# Colors: a fixed categorical order, checked for colorblind separation; text
# and grid stay in neutral ink so only the data carries color.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL = "#b9b8b1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9,
})


def load(run: str) -> list[dict]:
    return [json.loads(line) for line in (RUNS / f"{run}.jsonl").read_text().splitlines() if line.strip()]


def series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    points = [(r["step"], r[key]) for r in rows if key in r and r[key] is not None]
    return [p[0] for p in points], [p[1] for p in points]


def pass_key(rows: list[dict], prefix: str) -> str | None:
    keys = sorted({k for r in rows for k in r if k.startswith(f"{prefix}/pass@")})
    return keys[0] if keys else None


def dots(ax, x, y, color, marker="o", **kw):
    ax.plot(x, y, color=color, marker=marker, markersize=7.5, markeredgecolor=SURFACE,
            markeredgewidth=2, **kw)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(REPO))


# --- the standard diagnostic plots ------------------------------------------

def diagnostics(name: str, title: str, runs: list[tuple[str, str, str]], skip_degenerate: set[str] = frozenset()) -> None:
    """Ten panels, one line per run. `runs` is (run name, legend label, color)."""
    data = {run: load(run) for run, _, _ in runs}
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.0))
    panels = [
        ("train/success_rate", "train/success_rate", (0, 1.02)),
        ("eval/success_rate", "eval/success_rate", (0, 1.02)),
        ("train/pass@", "train/pass@k", (0, 1.02)),
        ("eval/pass@", "eval/pass@k", (0, 1.02)),
        ("train/grad_norm", "train/grad_norm", None),
        ("train/importance_weight_mean", "train/importance_weight_mean", (0.9, 1.1)),
        ("train/importance_weight_std", "train/importance_weight_std", None),
        ("train/importance_weight_max", "train/importance_weight_max", None),
        ("train/frac_degenerate_groups", "train/frac_degenerate_groups", (0, 1.02)),
        ("train/entropy_proxy", "train/entropy_proxy", None),
    ]
    for ax, (key, label, ylim) in zip(axes.flat, panels):
        notes = []
        keys_used = set()
        for run, legend, color in runs:
            rows = data[run]
            k = key
            if key.endswith("pass@"):
                k = pass_key(rows, key.split("/")[0])
                if k is None:
                    continue
                keys_used.add(k)
                notes.append(f"{legend}: {k}")
            if key == "train/frac_degenerate_groups" and run in skip_degenerate:
                label = f"{label}\n({legend.split()[0]}: not applicable)"
                continue
            x, y = series(rows, k)
            if key.startswith("eval/"):
                dots(ax, x, y, color, label=legend)
            else:
                ax.plot(x, y, color=color, label=legend)
        if len(keys_used) == 1:  # every run logs the same key: name it exactly
            label, notes = keys_used.pop(), []
        ax.set_title(label)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        if notes:  # the runs use different k: say which, under the panel
            ax.text(0.0, -0.13, "\n".join(notes), transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    for ax in axes[1]:
        ax.set_xlabel("training step")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if len(runs) > 1:
        fig.legend(handles, labels, loc="upper right", ncol=len(runs), bbox_to_anchor=(0.985, 1.0))
    fig.suptitle(title, x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.2)
    save(fig, name)


# --- Part 3: the same learning in half the time ------------------------------

def wallclock() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for run, label, color in (("p2b_number_search_grpo", "synchronous (p2b)", BLUE),
                              ("p3_number_search_async", "asynchronous (p3)", ORANGE)):
        rows = load(run)
        start = rows[0]["time"]
        when = {r["step"]: (r["time"] - start) / 60 for r in rows if "train/step_time_s" in r}
        when[0] = 0.0
        x, y = [], []
        for r in rows:
            if "eval/success_rate" in r and r["step"] in when:
                x.append(when[r["step"]]); y.append(r["eval/success_rate"])
        dots(ax, x, y, color, label=f"{label}: step 50 reached at {x[-1]:.0f} min")
    ax.set_ylim(0, 1.05); ax.set_xlim(left=-1)
    ax.set_xlabel("minutes since the baseline evaluation"); ax.set_ylabel("eval/success_rate")
    ax.set_title("Number Search, GRPO with groups of 8: same curve, half the clock")
    ax.legend(loc="lower right")
    save(fig, "part3_wallclock.png")


# --- Part 5 ------------------------------------------------------------------

def eval_by_task(model_file: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for line in (RUNS / model_file).read_text().splitlines():
        row = json.loads(line)
        if not row.get("abandoned"):
            out[row["task_id"]].append(float(row["reward"]))
    return out


def textcraft_curve() -> None:
    rows = load("p5_textcraft_grpo")
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    x, y = series(rows, "train/success_rate")
    ax.plot(x, y, color=BLUE, label="train/success_rate (training tasks, per step)")
    x, y = series(rows, "eval/success_rate")
    dots(ax, x, y, ORANGE, label="eval/success_rate, training server (50 held-out tasks x 4)")
    a3 = eval_by_task("p5_eval_a3.jsonl"); base = eval_by_task("p5_eval_Qwen_Qwen3.5-4B.jsonl")
    rate = lambda d: sum(map(sum, d.values())) / sum(map(len, d.values()))
    dots(ax, [0, 41], [rate(base), rate(a3)], AQUA, marker="s", linestyle="none",
         label="same tasks, evaluation server (Day 5)")
    ax.annotate(f"step 41 adapter\n{rate(a3):.3f}", (41, rate(a3)), xytext=(10, -4), textcoords="offset points",
                fontsize=8.5, color=INK2, va="center")
    ax.annotate(f"base model: {rate(base):.3f}", (0, rate(base)), xytext=(8, -12), textcoords="offset points",
                fontsize=8.5, color=INK2)
    ax.set_ylim(0, 1.0); ax.set_xlim(-1.5, 50); ax.set_xticks(range(0, 41, 10))
    ax.set_xlabel("training step (the run stopped at 41 of 50)"); ax.set_ylabel("success rate")
    ax.set_title("TextCraft: a steady climb over 41 steps")
    ax.legend(loc="upper left")
    save(fig, "part5_curve.png")


def stacked(ax, labels, parts, width=0.42):
    """parts: [(legend, color, values)], bottom first; 2px surface gaps between segments."""
    bottoms = [0.0] * len(labels)
    for legend, color, values in parts:
        ax.bar(labels, values, width, bottom=bottoms, color=color, edgecolor=SURFACE, linewidth=2, label=legend)
        for i, v in enumerate(values):
            if v >= 0.07:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color=INK if color in (NEUTRAL, YELLOW, AQUA) else "white")
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], names[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5))


def textcraft_turns() -> None:
    analysis = json.loads((RUNS / "p5_analysis.json").read_text())["eval"]
    steps = ["0", "10", "20", "30"]
    share = lambda step, *kinds: sum(analysis[step]["turn_share"].get(k, 0) for k in kinds)
    parts = [
        ("craft that worked", BLUE, [share(s, "craft_ok") for s in steps]),
        ("craft rejected", ORANGE, [share(s, "craft_rejected") for s in steps]),
        ("recipe lookup", AQUA, [share(s, "lookup") for s in steps]),
        ("inventory check, finish, other calls", YELLOW, [share(s, "inventory", "finish", "invalid_call", "abandoned") for s in steps]),
        ("no usable tool call", NEUTRAL, [share(s, "no_tool_call") for s in steps]),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    stacked(ax, [f"step {s}" for s in steps], parts)
    ax.set_title("TextCraft: what each turn was (evaluation episodes)")
    ax.set_ylabel("share of turns")
    save(fig, "part5_turns.png")


def textcraft_size() -> None:
    analysis = json.loads((RUNS / "p5_analysis.json").read_text())["eval"]
    sizes = {}
    for line in (REPO / "data" / "textcraft_val.jsonl").read_text().splitlines():
        task = json.loads(line); sizes[task["id"]] = task["info"]["num_craft_steps"]
    a3 = eval_by_task("p5_eval_a3.jsonl")
    bins = [("0-14", "14 or fewer items", 0, 14, BLUE), ("15-22", "15-22 items", 15, 22, ORANGE),
            ("23-31", "23-31 items", 23, 31, AQUA), ("32+", "32 or more items", 32, 999, YELLOW)]
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    for key, label, lo, hi, color in bins:
        x = [0, 10, 20, 30]
        y = [analysis[str(s)]["by_size"][key]["success"] for s in x]
        rewards = [r for task, rs in a3.items() if lo <= sizes[task] <= hi for r in rs]
        last = sum(rewards) / len(rewards)
        n = sum(1 for task in a3 if lo <= sizes[task] <= hi)
        dots(ax, x, y, color, label=f"{label} ({n} tasks)")
        ax.plot([30, 41], [y[-1], last], color=color, linestyle=(0, (1, 2.2)))
        dots(ax, [41], [last], color, marker="s")
        ax.annotate(label, (41, last), xytext=(9, {"0-14": 7, "15-22": -9}.get(key, -3)), textcoords="offset points",
                    fontsize=8.5, color=INK2)
    ax.set_ylim(-0.03, 1.0); ax.set_xlim(-1.5, 53)
    ax.set_xticks([0, 10, 20, 30, 41])
    ax.set_xlabel("training step (squares: step 41, measured on the evaluation server)")
    ax.set_ylabel("eval success rate")
    ax.set_title("TextCraft: success by how many items a task needs crafted")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    save(fig, "part5_size.png")


def textcraft_groups() -> None:
    rows = [r for r in load("p5_textcraft_grpo") if "train/num_groups" in r]
    blocks = [(1, 10), (11, 20), (21, 30), (31, 41)]
    fail, mixed, succ = [], [], []
    for lo, hi in blocks:
        rs = [r for r in rows if lo <= r["step"] <= hi]
        groups = sum(r["train/num_groups"] for r in rs)
        degenerate = sum(r["train/num_degenerate_groups"] for r in rs)
        all_fail = sum(round(r["train/num_groups"] * (1 - r["train/pass@8"])) for r in rs)
        fail.append(all_fail / groups); succ.append((degenerate - all_fail) / groups)
        mixed.append((groups - degenerate) / groups)
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    stacked(ax, [f"steps {lo}-{hi}" for lo, hi in blocks], [
        ("all 8 attempts failed (no signal)", NEUTRAL, fail),
        ("mixed results (trained on)", BLUE, mixed),
        ("all 8 succeeded (no signal)", ORANGE, succ),
    ])
    ax.set_title("TextCraft: which rollout groups carried a training signal")
    ax.set_ylabel("share of groups")
    save(fig, "part5_groups.png")


def main() -> None:
    diagnostics("diag_sync.png", "Number Search, synchronous runs (Parts 2 and 4.1)", [
        ("p2a_number_search_reinforce", "p2a REINFORCE", BLUE),
        ("p2b_number_search_grpo", "p2b GRPO, group 8", ORANGE),
        ("p41_number_search_grpo_g2", "p41 GRPO, group 2", AQUA),
    ], skip_degenerate={"p2a_number_search_reinforce"})
    diagnostics("diag_async.png", "Number Search, asynchronous runs (Parts 3 and 4.3)", [
        ("p3_number_search_async", "p3 staleness 2, PPO", BLUE),
        ("p43_number_search_async_stale4_iw", "p43 staleness 4, PPO", ORANGE),
        ("p43_number_search_async_stale4_noiw", "p43 staleness 4, no importance weighting", AQUA),
    ])
    diagnostics("diag_textcraft.png", "TextCraft, asynchronous GRPO (Part 5, stopped at step 41)", [
        ("p5_textcraft_grpo", "p5", BLUE),
    ])
    wallclock()
    textcraft_curve()
    textcraft_turns()
    textcraft_size()
    textcraft_groups()


if __name__ == "__main__":
    main()
