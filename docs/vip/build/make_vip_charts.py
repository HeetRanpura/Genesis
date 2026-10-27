"""Result charts for the VIP decks and reports (docs/vip/diagrams/v12-v14_*.png), drawn from committed result files.

    .venv/bin/python docs/vip/build/make_vip_charts.py

Images rather than native PowerPoint charts: Keynote (the team's presentation app) does not render pptxgenjs charts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "vip" / "diagrams"
INK, INK2, GRID = "#1b1b1a", "#52514e", "#e3e3e0"
COL = {"random": "#b5b4ae", "greedy": "#8a8984", "heuristic": "#1baf7a", "ff": "#eb6834", "gru": "#2a78d6"}
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def frame(ax, title):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=12)


def label_bars(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=12, color=INK)


def throughput():
    rows = {}
    for line in (ROOT / "reports/stage1/bench.txt").read_text().splitlines():
        m = re.match(r"\s*(\d+)x\d+ agents=\s*(\d+) obs=True .*?(\d+) steps/min", line)
        if m:
            rows[(int(m[1]), int(m[2]))] = int(m[3]) / 1000
    keys = [(256, 256), (256, 1000), (256, 2000), (512, 1000)]
    labels = ["256 agents", "1,000 agents", "2,000 agents", "1,000 agents\n512 x 512 world"]
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=200)
    bars = ax.bar(labels, [rows[k] for k in keys], color=["#2a78d6", "#2a78d6", "#2a78d6", "#9cc0ea"], width=0.6)
    ax.axhline(100, color="#eb6834", lw=1.4, ls="--")
    ax.text(3.35, 102, "target 100k", color="#eb6834", ha="right", fontsize=11)
    label_bars(ax, bars, "{:.0f}k")
    ax.set_ylabel("env steps per minute (thousands)")
    ax.set_ylim(0, 165)
    frame(ax, "Throughput with observations (MacBook Air M4)")
    fig.tight_layout()
    fig.savefig(OUT / "v12_throughput.png")
    plt.close(fig)


def mortality():
    base = json.loads((ROOT / "reports/stage1/baselines.json").read_text())["per_run"]
    pts = {}
    for p in ("random", "greedy", "heuristic"):
        pts[p] = [r["deaths_per_agent_year"] for r in base if r["policy"] == p]
    for p in ("ff", "gru"):
        pts[p] = [r["deaths_per_agent_year"] for r in json.loads((ROOT / f"reports/stage2/eval_{p}_s0.json").read_text())["per_run"]]
    ci = {p: json.loads((ROOT / f"reports/stage2/eval_{p}_s0.json").read_text())["summary"]["deaths_per_agent_year"] for p in ("ff", "gru")}
    for p, c in json.loads((ROOT / "reports/stage2/eval_gru_s0.json").read_text())["vs_baselines"].items():
        ci[p] = c["baseline"]
    order = ["random", "greedy", "heuristic", "ff", "gru"]
    names = ["Random", "Greedy", "Heuristic", "Memoryless\nlearner", "GRU\nlearner"]
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=200)
    means = [ci[p][0] for p in order]
    bars = ax.bar(names, means, color=[COL[p] for p in order], width=0.6)
    ax.errorbar(range(5), means, yerr=[[m - ci[p][1] for m, p in zip(means, order)], [ci[p][2] - m for m, p in zip(means, order)]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=5)
    for i, p in enumerate(order):
        ax.scatter(np.full(len(pts[p]), i) + np.linspace(-0.12, 0.12, len(pts[p])), pts[p], s=14, color="white", edgecolor=INK, lw=0.7, zorder=3)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2 + 0.33, m, f"{m:.2f}", va="center", fontsize=11, color=INK)
    ax.set_ylabel("deaths per agent-year (lower is better)")
    ax.set_ylim(0, 4.6)
    frame(ax, "Held-out mortality, worlds 0-4 x 2 years (95% CI, dots = worlds)")
    fig.tight_layout()
    fig.savefig(OUT / "v13_mortality.png")
    plt.close(fig)


def energy():
    s = json.loads((ROOT / "reports/stage2/energy_economy.json").read_text())["summary"]
    series = [("heuristic", "Heuristic", COL["heuristic"]), ("ppo:ff_s0", "Memoryless", COL["ff"]), ("ppo:gru_s0", "GRU", COL["gru"])]
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=200)
    x = np.arange(4)
    for k, (key, name, c) in enumerate(series):
        tot = np.sum([v for v in s[key]["intake_by_season"].values()], axis=0)
        bars = ax.bar(x + (k - 1) * 0.26, tot, 0.25, color=c, label=name)
        label_bars(ax, bars, "{:.1f}")
    ax.set_xticks(x, seasons)
    ax.set_ylabel("energy eaten per agent")
    ax.set_ylim(0, 3.7)
    ax.legend(frameon=False, loc="upper right")
    frame(ax, "Energy eaten per agent, by season (validation worlds)")
    fig.tight_layout()
    fig.savefig(OUT / "v14_energy_by_season.png")
    plt.close(fig)


if __name__ == "__main__":
    throughput()
    mortality()
    energy()
    print("ok")
