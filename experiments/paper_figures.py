"""Summary figures for the paper from saved results (no simulation): held-out mortality incl. the replication,
causes of death, training curves, energy economy.  -> paper/figures/*.pdf"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
R2 = ROOT / "reports" / "stage2"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})
CAUSE_COL = {"starvation": "#e67e22", "thirst": "#2980b9", "cold": "#7fb3d5", "fire": "#e74c3c", "predator": "#7b241c",
             "old_age": "#7f8c8d", "strike": "#000", "other": "#ccc"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "reports" / "stage1" / "baselines.json").read_text())
    ev = {k: json.loads((R2 / f"eval_{k}.json").read_text()) for k in ("ff_s0", "gru_s0")}
    orig = {k: json.loads((R2 / "rppo_original" / f"eval_{k}.json").read_text()) for k in ("ff_s0", "gru_s0")}
    rows = [("Random", base["summary"]["random"]["deaths_per_agent_year"], "#9aa0a6"),
            ("Greedy", base["summary"]["greedy"]["deaths_per_agent_year"], "#e6a23c"),
            ("Heuristic", base["summary"]["heuristic"]["deaths_per_agent_year"], "#3a7bd5"),
            ("Memoryless\n(original)", orig["ff_s0"]["summary"]["deaths_per_agent_year"], "#c39bd3"),
            ("Memoryless\n(replication)", ev["ff_s0"]["summary"]["deaths_per_agent_year"], "#8e44ad"),
            ("GRU\n(original)", orig["gru_s0"]["summary"]["deaths_per_agent_year"], "#82c785"),
            ("GRU\n(replication)", ev["gru_s0"]["summary"]["deaths_per_agent_year"], "#2e7d32")]
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    for i, (lab, ci, c) in enumerate(rows):
        ax.bar(i, ci[0], color=c, yerr=[[ci[0] - ci[1]], [ci[2] - ci[0]]], capsize=3)
        ax.text(i, ci[2] + 0.08, f"{ci[0]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.set_ylabel("deaths / agent-year\n(held-out worlds, 95% CI)")
    fig.savefig(OUT / "heldout_mortality.pdf"); plt.close(fig)

    # causes of death on held-out worlds (summed over 5 seeds)
    def causes(per_run):
        c = {}
        for r in per_run:
            for k, v in r["deaths_by_cause"].items():
                c[k] = c.get(k, 0) + v
        return c
    groups = [("Random", causes([r for r in base["per_run"] if r["policy"] == "random"])),
              ("Greedy", causes([r for r in base["per_run"] if r["policy"] == "greedy"])),
              ("Heuristic", causes([r for r in base["per_run"] if r["policy"] == "heuristic"])),
              ("PPO memoryless", causes(ev["ff_s0"]["per_run"])), ("PPO GRU", causes(ev["gru_s0"]["per_run"]))]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, (lab, c) in enumerate(groups):
        tot, left = sum(c.values()), 0.0
        for k, col in CAUSE_COL.items():
            f = c.get(k, 0) / max(tot, 1)
            ax.barh(i, f, left=left, color=col, label=k if i == 0 else None)
            left += f
    ax.set_yticks(range(len(groups)), [g[0] for g in groups]); ax.invert_yaxis()
    ax.set_xlabel("share of deaths (held-out worlds)")
    ax.legend(frameon=False, fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.4, -0.55))
    fig.savefig(OUT / "causes.pdf"); plt.close(fig)

    # training curves
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.1))
    for name, c, lab in (("ff_s0", "#8e44ad", "memoryless"), ("gru_s0", "#2e7d32", "GRU")):
        rows_ = [json.loads(line) for line in open(ROOT / "runs" / "stage2" / name / "train.jsonl")]
        st = np.array([r["agent_steps"] for r in rows_]) / 1e6
        k = 40
        sm = lambda x: np.convolve(x, np.ones(k) / k, mode="valid")  # noqa: E731
        axs[0].plot(st[k - 1:], sm([r["deaths_per_agent_year"] for r in rows_]), color=c, label=lab)
        axs[1].plot(st, [r["ent"] for r in rows_], color=c)
        axs[2].plot(st[k - 1:], sm([r["mean_hydration"] for r in rows_]), color=c)
        axs[2].plot(st[k - 1:], sm([r["mean_energy"] for r in rows_]), color=c, ls="--")
    axs[0].axhline(base["summary"]["heuristic"]["deaths_per_agent_year"][0], color="#3a7bd5", ls=":", lw=0.8)
    axs[0].set(xlabel="agent-steps (M)", ylabel="deaths / agent-year", title="Training-world mortality")
    axs[1].set(xlabel="agent-steps (M)", ylabel="nats", title="Policy entropy")
    axs[2].set(xlabel="agent-steps (M)", title="Hydration (—), energy (--)")
    axs[0].legend(frameon=False, fontsize=6)
    fig.savefig(OUT / "training.pdf"); plt.close(fig)

    # energy economy
    en = json.loads((R2 / "energy_economy.json").read_text())["summary"]
    order = ["heuristic:grazer", "heuristic:fruit", "heuristic:hunter", "heuristic", "ppo:ff_s0", "ppo:gru_s0"]
    labs = ["Grazer", "Fruit-first", "Hunter", "Heuristic\n(mixed)", "PPO\nmemoryless", "PPO\nGRU"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.3))
    bottom = np.zeros(len(order))
    for src, col in (("plants", "#6aa84f"), ("fruit", "#e69138"), ("meat / carried food", "#a61c00")):
        v = np.array([en[p]["intake_per_agent_year"][src] for p in order])
        axs[0].bar(range(len(order)), v, bottom=bottom, color=col, label=src.split(" /")[0]); bottom += v
    axs[0].axhline(5.0, color="k", ls=":", lw=0.8, label="basal metabolism")
    axs[0].set_xticks(range(len(order)), [lb.replace("\n", " ") for lb in labs], fontsize=6.5, rotation=35, ha="right")
    axs[0].set(ylabel="energy eaten / agent-year", title="Diet (validation worlds)")
    axs[0].legend(frameon=False, fontsize=6)
    seasons = ["spring", "summer", "autumn", "winter"]
    for p, lab, ls in zip(order, labs, ["-", "-", "-", "-", "--", "--"]):
        tot = np.sum([en[p]["intake_by_season"][s] for s in en[p]["intake_by_season"]], 0)
        axs[1].plot(seasons, tot, ls, marker="o", ms=2.5, label=lab.replace("\n", " "))
    axs[1].set(ylabel="energy eaten per agent per season", title="Seasonal intake")
    axs[1].legend(frameon=False, fontsize=5.5, ncol=2)
    fig.savefig(OUT / "energy.pdf"); plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
