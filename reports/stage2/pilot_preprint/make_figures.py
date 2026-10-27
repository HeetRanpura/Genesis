"""Figures and statistics for the GENESIS preprint (reports/stage2/pilot_preprint/).

Reads only existing outputs; never trains or re-evaluates:
  reports/stage1/baselines.json, runs/stage1_baselines/<policy>_s<k>/metrics.csv      (baselines, worlds 0-4 x 2 yr)
  reports/stage2/rppo_original/eval_{gru,ff}_s0.json, runs/stage2_pilot/eval/{gru,ff}_s0/s<k>/ (pilot learners)
  reports/stage2/pilot_preprint/data/*short*                                                     (worlds 0-1 x 0.5 yr, exploratory)
  runs/stage2_pilot/<run>/train.jsonl, eval.jsonl                                             (training logs)
Writes fig_*.pdf / fig_*.png and stats.json next to this file.

    .venv/bin/python reports/stage2/pilot_preprint/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.survival import bootstrap_ci, hedges_g  # noqa: E402
from analysis.trajectories import load  # noqa: E402
from pilot_runs import recording, train_log  # noqa: E402

OUT = Path(__file__).resolve().parent
DATA = OUT / "data"

# one categorical order everywhere (validated: dataviz validate_palette.js, light mode, adjacent pairs)
POLICIES = ["gru", "ff", "heuristic", "greedy", "random"]
LABEL = {"gru": "GRU learner", "ff": "Memoryless learner", "heuristic": "Heuristic", "greedy": "Greedy", "random": "Random"}
COLOR = {"gru": "#2a78d6", "ff": "#eb6834", "heuristic": "#1baf7a", "greedy": "#eda100", "random": "#e87ba4"}
CAUSES = ["starvation", "thirst", "cold", "fire", "predator", "old_age"]
CCOLOR = dict(zip(CAUSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
SEASONS = ["spring", "summer", "autumn", "winter"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def jl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def metrics_csv(p):
    return np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding=None)


# ------------------------------------------------------------------ load the 2-year held-out evaluation
def per_run_long():
    base = json.loads((ROOT / "reports/stage1/baselines.json").read_text())["per_run"]
    runs = {p: [r for r in base if r["policy"] == p] for p in ("heuristic", "greedy", "random")}
    for p in ("gru", "ff"):
        runs[p] = json.loads((ROOT / f"reports/stage2/rppo_original/eval_{p}_s0.json").read_text())["per_run"]
    for p in POLICIES:
        runs[p] = sorted(runs[p], key=lambda r: r["seed"])
        for r in runs[p]:
            n = sum(r["deaths_by_cause"].values())
            r["agent_years"] = n / r["deaths_per_agent_year"]
    return runs


def metrics_dir(p, seed):
    if p in ("gru", "ff"):
        return recording(p, seed)
    return ROOT / f"runs/stage1_baselines/{p}_s{seed}"


STATS: dict = {}


# ------------------------------------------------------------------ Fig: main mortality result
def fig_mortality(runs):
    rows = {}
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for i, p in enumerate(POLICIES[::-1]):
        d = np.array([r["deaths_per_agent_year"] for r in runs[p]])
        m, lo, hi = bootstrap_ci(d)
        rows[p] = dict(mean=m, ci=[lo, hi], per_world=d.tolist(),
                       km1=bootstrap_ci([r["km_survival_1yr"] for r in runs[p]]),
                       median_life_days=float(np.median([r["median_lifetime_days"] for r in runs[p]])))
        ax.plot([lo, hi], [i, i], color=COLOR[p], lw=2, solid_capstyle="round")
        ax.scatter(d, np.full(len(d), i) + 0.22, s=9, color=COLOR[p], alpha=0.55, lw=0)
        ax.scatter([m], [i], s=34, color=COLOR[p], edgecolor="white", lw=1.2, zorder=3)
        ax.text(hi + 0.08, i, f"{m:.2f}", va="center", color=INK, fontsize=7)
    ax.set_yticks(range(len(POLICIES)), [LABEL[p] for p in POLICIES[::-1]])
    ax.set_xlabel("deaths per agent-year (held-out worlds 0–4, 2 yr)")
    ax.set_xlim(0, 4.8)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_mortality")
    comp = {}
    for a in ("gru", "ff"):
        for b in ("heuristic", "greedy", "random"):
            da = [r["deaths_per_agent_year"] for r in runs[a]]
            db = [r["deaths_per_agent_year"] for r in runs[b]]
            ca, cb = rows[a]["ci"], rows[b]["ci"]
            comp[f"{a}_vs_{b}"] = dict(hedges_g=hedges_g(da, db), cis_overlap=bool(ca[0] <= cb[1] and cb[0] <= ca[1]),
                                       paired_wins=int(sum(x < y for x, y in zip(da, db))))
    da = [r["deaths_per_agent_year"] for r in runs["gru"]]
    db = [r["deaths_per_agent_year"] for r in runs["ff"]]
    comp["gru_vs_ff"] = dict(hedges_g=hedges_g(da, db), paired_wins_gru=int(sum(x < y for x, y in zip(da, db))))
    STATS["mortality_2yr"] = rows
    STATS["comparisons_2yr"] = comp


# ------------------------------------------------------------------ Fig: cause-specific death rates
def fig_causes(runs):
    rates = {}
    for p in POLICIES:
        ay = sum(r["agent_years"] for r in runs[p])
        rates[p] = {c: sum(r["deaths_by_cause"].get(c, 0) for r in runs[p]) / ay for c in CAUSES}
        per = {c: [r["deaths_by_cause"].get(c, 0) / r["agent_years"] for r in runs[p]] for c in CAUSES}
        rates[p]["ci"] = {c: bootstrap_ci(per[c]) for c in CAUSES}
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for i, p in enumerate(POLICIES[::-1]):
        left = 0.0
        for c in CAUSES:
            w = rates[p][c]
            if w <= 0:
                continue
            ax.barh(i, w, left=left, height=0.62, color=CCOLOR[c], edgecolor="white", lw=1.0)
            if w > 0.28:
                ax.text(left + w / 2, i, f"{w:.2f}", ha="center", va="center", fontsize=6, color="white")
            left += w
    ax.set_yticks(range(len(POLICIES)), [LABEL[p] for p in POLICIES[::-1]])
    ax.set_xlabel("deaths per agent-year, by cause")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CCOLOR[c]) for c in CAUSES]
    ax.legend(handles, [c.replace("_", " ") for c in CAUSES], ncol=6, fontsize=6.5, loc="lower center",
              bbox_to_anchor=(0.4, 1.0), handlelength=0.9, columnspacing=0.7, handletextpad=0.3)
    save(fig, "fig_causes")
    STATS["cause_rates_2yr"] = rates


# ------------------------------------------------------------------ Fig: seasonality of death and energy
def fig_seasons():
    yl = 3840
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.9))
    w = 0.16
    for j, p in enumerate(POLICIES):
        dz, ez = np.zeros((5, 4)), np.zeros((5, 4))
        for s in range(5):
            m = metrics_csv(metrics_dir(p, s) / "metrics.csv")
            tot = sum(m[f"deaths_{c}"] for c in CAUSES)
            inc = np.diff(np.concatenate([[0], tot]))
            for q in range(4):
                sel = m["season"] == q
                ticks = sel.sum() * (m["tick"][1] - m["tick"][0])
                dz[s, q] = inc[sel].sum() / (256 * ticks / yl)
                ez[s, q] = m["mean_energy"][sel].mean()
        out[p] = dict(deaths_per_agent_year=dz.mean(0).tolist(), deaths_range=[dz.min(0).tolist(), dz.max(0).tolist()],
                      mean_energy=ez.mean(0).tolist())
        x = np.arange(4) + (j - 2) * w
        axes[0].bar(x, dz.mean(0), width=w * 0.88, color=COLOR[p], label=LABEL[p])
        axes[0].errorbar(x, dz.mean(0), yerr=[dz.mean(0) - dz.min(0), dz.max(0) - dz.mean(0)], fmt="none",
                         ecolor=INK2, elinewidth=0.6, capsize=0)
        axes[1].plot(range(4), ez.mean(0), color=COLOR[p], lw=2, marker="o", ms=4, mec="white", mew=0.8)
    axes[0].set_xticks(range(4), SEASONS)
    axes[0].set_ylabel("deaths per agent-year")
    axes[0].grid(axis="x", visible=False)
    axes[0].legend(fontsize=6.5, ncol=1, loc="upper left", handlelength=1)
    axes[1].set_xticks(range(4), SEASONS)
    axes[1].set_ylabel("mean energy of living agents")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlim(-0.2, 3.2)
    fig.subplots_adjust(wspace=0.28)
    axes[0].set_title("a  mortality by season", loc="left", fontsize=8)
    axes[1].set_title("b  energy reserve by season", loc="left", fontsize=8)
    save(fig, "fig_seasons")
    STATS["by_season"] = out


# ------------------------------------------------------------------ Fig: horizon (0.5 yr spring-summer vs 2 yr)
def fig_horizon(runs):
    short_base = json.loads((DATA / "baselines_short_0.5yr_seeds0-1.json").read_text())["per_run"]
    short = {p: [r["deaths_per_agent_year"] for r in short_base if r["policy"] == p] for p in ("heuristic", "greedy", "random")}
    for p in ("gru", "ff"):
        short[p] = [r["deaths_per_agent_year"] for r in json.loads((DATA / f"eval_short_{p}_s0.json").read_text())["per_run"]]
    long = {p: [r["deaths_per_agent_year"] for r in runs[p]] for p in POLICIES}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.7), sharex=True)
    for ax, src, title in ((axes[0], short, "a  first half-year (spring–summer), worlds 0–1"),
                           (axes[1], long, "b  two full years, worlds 0–4")):
        for i, p in enumerate(POLICIES[::-1]):
            d = np.array(src[p])
            ax.barh(i, d.mean(), height=0.6, color=COLOR[p])
            ax.scatter(d, np.full(len(d), i), s=8, color=INK, zorder=3, lw=0)
            ax.text(d.max() + 0.07, i, f"{d.mean():.2f}", va="center", fontsize=7)
        ax.set_yticks(range(len(POLICIES)), [LABEL[p] for p in POLICIES[::-1]])
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlabel("deaths per agent-year")
    axes[1].set_yticklabels([])
    save(fig, "fig_horizon")
    STATS["horizon"] = dict(short_0_5yr={p: dict(mean=float(np.mean(v)), per_world=v) for p, v in short.items()},
                            long_2yr={p: float(np.mean(v)) for p, v in long.items()})


# ------------------------------------------------------------------ Fig: training dynamics
def moving(x, k):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[k:] - c[:-k]) / k


def fig_training():
    runs = {"gru_s0": "gru", "ff_s0": "ff", "gru_s1": "gru", "ff_s1": "ff"}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.9))
    info = {}
    for name, p in runs.items():
        t = jl(train_log(name))
        steps = np.array([r["agent_steps"] for r in t]) / 1e6
        ent = np.array([r["ent"] for r in t])
        mort = np.array([r["deaths_per_agent_year"] for r in t])
        ls = "-" if name.endswith("s0") else (0, (2, 1.5))
        k = 40
        axes[0].plot(steps, ent, color=COLOR[p], lw=1.4 if name.endswith("s0") else 1.0, ls=ls)
        if len(t) > k:
            axes[1].plot(steps[k - 1:], moving(mort, k), color=COLOR[p], lw=1.4 if name.endswith("s0") else 1.0, ls=ls)
        info[name] = dict(iters=len(t), agent_steps=int(t[-1]["agent_steps"]), entropy_start=float(ent[0]),
                          entropy_end=float(ent[-20:].mean()), steps_per_s=float(np.median([r["agent_steps_per_s"] for r in t])))
    axes[0].set_ylabel("policy entropy (nats)")
    axes[0].axhline(np.log(11), color=INK2, lw=0.6, ls=":")
    axes[0].text(0.3, np.log(11) - 0.12, "uniform over 11 actions", fontsize=6, color=INK2, va="top")
    axes[1].set_ylabel("training mortality (40-iter mean)")
    for ax in axes[:2]:
        ax.set_xlabel("agent-steps (millions)")
    # held-out mini-evaluations of the replication runs vs the matched baseline protocol
    sb = json.loads((DATA / "baselines_short_0.5yr_seeds0-1.json").read_text())["per_run"]
    for b in ("heuristic", "greedy", "random"):
        v = np.mean([r["deaths_per_agent_year"] for r in sb if r["policy"] == b])
        axes[2].axhline(v, color=COLOR[b], lw=1.2)
        axes[2].text(21.8, v + {"heuristic": 0.07, "greedy": -0.07, "random": 0}[b], LABEL[b], fontsize=6,
                     color=INK2, va="center")
    ev = {}
    for name in ("gru_s1", "ff_s1"):
        e = jl(train_log(name, "eval"))
        x = [r["agent_steps"] / 1e6 for r in e]
        y = [r["deaths_per_agent_year"] for r in e]
        axes[2].plot(x, y, color=COLOR[runs[name]], lw=2, marker="o", ms=5, mec="white", mew=0.8)
        ev[name] = [dict(iter=r["iter"], agent_steps=r["agent_steps"], deaths_per_agent_year=r["deaths_per_agent_year"]) for r in e]
    for p in ("gru", "ff"):
        s = json.loads((DATA / f"eval_short_{p}_s0.json").read_text())["per_run"]
        axes[2].scatter([19.7], [np.mean([r["deaths_per_agent_year"] for r in s])], marker="D", s=22,
                        color=COLOR[p], edgecolor="white", lw=0.8, zorder=3)
    axes[2].set_xlim(0, 21.5)
    axes[2].set_xlabel("agent-steps (millions)")
    axes[2].set_ylabel("held-out mortality, 0.5 yr")
    axes[0].set_title("a", loc="left", fontsize=8)
    axes[1].set_title("b", loc="left", fontsize=8)
    axes[2].set_title("c", loc="left", fontsize=8)
    handles = [plt.Line2D([], [], color=COLOR["gru"], lw=1.6), plt.Line2D([], [], color=COLOR["ff"], lw=1.6),
               plt.Line2D([], [], color=INK2, lw=1.4), plt.Line2D([], [], color=INK2, lw=1.0, ls=(0, (2, 1.5)))]
    fig.legend(handles, ["GRU", "memoryless", "pilot (seed 0)", "replication (seed 1, partial)"], ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.08), fontsize=7)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_training")
    STATS["training"] = info
    STATS["replication_mini_evals"] = ev


# ------------------------------------------------------------------ Fig: learned control policies from recordings
def behaviour():
    groups = {"move": [1, 2, 3, 4], "eat": [6], "drink": [7], "rest": [5], "pickup/drop": [9, 10], "hunt": [8], "noop": [0]}
    gcolor = dict(zip(groups, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.0))
    bins = np.linspace(0, 1, 21)
    mids = (bins[1:] + bins[:-1]) / 2
    for k, p in enumerate(("gru", "ff")):
        A, E, W = [], [], []
        for s in range(5):
            d = recording(p, s)
            a = load(d, "agents")
            w = load(d, "world")
            ev = load(d, "events")
            A.append(a)
            E.append(ev)
            W.append(w)
        a = np.concatenate(A)
        season = (a["tick"] // 960) % 4
        budget = {}
        for q in range(4):
            sel = season == q
            n = sel.sum()
            budget[SEASONS[q]] = {g: float(np.isin(a["action"][sel], ids).sum() / n) for g, ids in groups.items()}
        overall = {g: float(np.isin(a["action"], ids).mean()) for g, ids in groups.items()}
        # stacked action budget by season
        ax = axes[0]
        for q in range(4):
            bottom = 0
            for g in groups:
                v = budget[SEASONS[q]][g]
                ax.bar(q + (k - 0.5) * 0.4, v, width=0.36, bottom=bottom, color=gcolor[g], edgecolor="white", lw=0.6)
                bottom += v
        # control curves
        drink = np.array([((a["hydration"] >= lo) & (a["hydration"] < hi) & (a["action"] == 7)).sum() /
                          max(1, ((a["hydration"] >= lo) & (a["hydration"] < hi)).sum()) for lo, hi in zip(bins[:-1], bins[1:])])
        eat = np.array([((a["energy"] >= lo) & (a["energy"] < hi) & (a["action"] == 6)).sum() /
                        max(1, ((a["energy"] >= lo) & (a["energy"] < hi)).sum()) for lo, hi in zip(bins[:-1], bins[1:])])
        axes[1].plot(mids, drink, color=COLOR[p], lw=2, label=LABEL[p])
        axes[2].plot(mids, eat, color=COLOR[p], lw=2, label=LABEL[p])
        # behavioural inflexibility: DRINK actions that produced no drink event, drought vs normal
        fail = {}
        for a_s, ev_s, w_s in zip(A, E, W):
            dr = a_s[a_s["action"] == 7]
            ok = ev_s[ev_s[:, 1] == 18]
            keys_ok = set(zip(ok[:, 0].astype(np.int64), ok[:, 2].astype(np.int64)))
            hit0 = np.array([(int(t), int(u)) in keys_ok for t, u in zip(dr["tick"], dr["uid"])])
            hit1 = np.array([(int(t) + 1, int(u)) in keys_ok for t, u in zip(dr["tick"], dr["uid"])])
            hit = hit0 if hit0.mean() >= hit1.mean() else hit1
            dl = dict(zip(w_s["tick"].astype(np.int64), w_s["drought_left"]))
            dro = np.array([dl.get(int(t), 0) > 0 for t in dr["tick"]])
            for key, m in (("drought", dro), ("normal", ~dro)):
                f = fail.setdefault(key, [0, 0])
                f[0] += int((~hit[m]).sum())
                f[1] += int(m.sum())
        fail = {k2: dict(failed=v[0], drink_actions=v[1], share=v[0] / max(1, v[1])) for k2, v in fail.items()}
        kills_small = int(sum((e[:, 1] == 5).sum() for e in E))
        kills_large = int(sum((e[:, 1] == 6).sum() for e in E))
        attacks_large = int(sum((e[:, 1] == 19).sum() for e in E))
        eat_src = {n: int(sum((e[:, 1] == i).sum() for e in E)) for n, i in (("plant", 2), ("fruit", 3), ("meat", 4))}
        agent_ticks = len(a)
        out[p] = dict(action_budget=overall, budget_by_season=budget, drink_failure=fail,
                      kills_small=kills_small, kills_large=kills_large, attacks_large=attacks_large,
                      eat_events=eat_src, agent_ticks=agent_ticks,
                      mean_hydration=float(a["hydration"].mean()), mean_energy=float(a["energy"].mean()))
    for ax, thr, lab in ((axes[1], (0.55, 0.95), "hydration"), (axes[2], (0.6, 0.9), "energy")):
        ax.axvspan(thr[0], thr[1], color=COLOR["heuristic"], alpha=0.10, lw=0)
        ax.axvline(thr[0], color=COLOR["heuristic"], lw=1.0, ls=":")
        ax.text(thr[0] + 0.01, 0.55, "heuristic\nhysteresis", fontsize=6, color=INK2, va="top", transform=ax.get_xaxis_transform())
        ax.set_xlabel(f"own {lab}")
        ax.set_xlim(0, 1)
    axes[1].set_ylabel("P(drink | hydration)")
    axes[2].set_ylabel("P(eat | energy)")
    fig.legend([plt.Line2D([], [], color=COLOR[p], lw=2) for p in ("gru", "ff")], [LABEL["gru"], LABEL["ff"]],
               ncol=2, loc="upper center", bbox_to_anchor=(0.66, 1.07), fontsize=7)
    axes[0].set_xticks(range(4), [s[:3] for s in SEASONS])
    axes[0].set_ylabel("share of actions")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="x", visible=False)
    axes[0].set_xlabel("season (left bar GRU, right bar memoryless)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=gcolor[g]) for g in groups]
    axes[0].legend(handles, list(groups), fontsize=5.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24),
                   handlelength=0.8, columnspacing=0.6)
    for ax, t in zip(axes, "abc"):
        ax.set_title(t, loc="left", fontsize=8)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_behaviour")
    STATS["behaviour_learners"] = out


def main():
    runs = per_run_long()
    fig_mortality(runs)
    fig_causes(runs)
    fig_seasons()
    fig_horizon(runs)
    fig_training()
    behaviour()
    (OUT / "stats.json").write_text(json.dumps(STATS, indent=1, default=float))
    print(json.dumps({k: STATS[k] for k in ("comparisons_2yr", "horizon")}, indent=1, default=float))


if __name__ == "__main__":
    main()
