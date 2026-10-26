"""Behavioural analysis of every agent type on one held-out world, from flight recordings (no new simulation).

    python experiments/behaviour_analysis.py --seed 0     -> paper/figures/*.pdf, paper/stats.json

Agent types: random, greedy, heuristic (runs/stage1_baselines/<p>_s<seed>) and the two learners
(runs/stage2_eval/<run>/s<seed>). Each recording holds every agent at every tick: position, body state, action,
reward, death and cause (plus the learners' value estimate and action probabilities). Recordings are loaded one
at a time (memory and heat), only the columns needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.survival import kaplan_meier  # noqa: E402
from analysis.trajectories import info, load  # noqa: E402
from sim import load_config  # noqa: E402
from sim.constants import BASE_ACTION_NAMES, DEATH_NAMES  # noqa: E402
from sim.worldgen import generate_world  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
POLICIES = {"random": "Random", "greedy": "Greedy", "heuristic": "Heuristic", "ff_s0": "PPO (memoryless)",
            "gru_s0": "PPO (GRU)"}
COL = {"random": "#9aa0a6", "greedy": "#e6a23c", "heuristic": "#3a7bd5", "ff_s0": "#8e44ad", "gru_s0": "#2e7d32"}
GROUPS = {"move": [1, 2, 3, 4], "rest": [5], "eat": [6], "drink": [7], "hunt": [8], "pickup/drop": [9, 10], "noop": [0]}
GROUP_COL = ["#7f8c8d", "#34495e", "#e67e22", "#2980b9", "#c0392b", "#f1c40f", "#bdc3c7"]
YEAR, DAY = 3840, 120
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})


def run_dir(p, seed):
    return ROOT / "runs" / ("stage1_baselines" / Path(f"{p}_s{seed}") if p in ("random", "greedy", "heuristic")
                            else Path("stage2_eval") / p / f"s{seed}")


def lifetimes_from_rows(a, start_tick):
    """Per-agent (duration, died) from agent rows: an agent's life runs from its first row (founders: run start)."""
    order = np.argsort(a["uid"], kind="stable")
    uid, tick, died = a["uid"][order], a["tick"][order], a["died"][order]
    first = np.r_[True, uid[1:] != uid[:-1]]
    last = np.r_[uid[1:] != uid[:-1], True]
    t0 = np.maximum(tick[first], start_tick)
    t1 = tick[last] + 1
    return (t1 - t0).astype(float), died[last].astype(bool)


def nn_ratio(xy_by_tick, habitat_cells, rng, weights=None):
    """Mean nearest-neighbour distance among agents / the same for equally many points drawn from the habitat the
    policy actually used (cells it occupied over the run; weights = how often, i.e. its own place preferences).
    <1: aggregated beyond what habitat (or, weighted, beyond what shared place preference) explains."""
    p = None if weights is None else weights / weights.sum()
    obs, null = [], []
    for xy in xy_by_tick:
        if len(xy) < 3:
            continue
        d, _ = cKDTree(xy).query(xy, k=2)
        obs.append(d[:, 1].mean())
        nd = []
        for _ in range(3):
            pick = rng.choice(len(habitat_cells), len(xy), p=p)
            pts = habitat_cells[pick] + rng.random((len(xy), 2)) * 0.01
            dd, _ = cKDTree(pts).query(pts, k=2)
            nd.append(dd[:, 1].mean())
        null.append(np.mean(nd))
    obs, null = np.array(obs), np.array(null)
    return float(np.mean(obs / null)), obs, null


def analyse(p, seed, W, rng):
    d = run_dir(p, seed)
    meta = info(d)
    start = int(meta["run"].get("start_tick", meta["index"]["start_tick"] if meta["index"] else 0))
    a = load(d, "agents")
    w = load(d, "world")
    tod = dict(zip(w["tick"], w["time_of_day"]))
    season = dict(zip(w["tick"], w["season"]))
    t_rel = a["tick"] - start
    s = {}
    # A. action budget
    counts = np.bincount(a["action"], minlength=len(BASE_ACTION_NAMES))[:len(BASE_ACTION_NAMES)]
    s["action_budget"] = {g: float(counts[idx].sum() / counts.sum()) for g, idx in GROUPS.items()}
    # B. need-conditioned choice
    bins = np.linspace(0, 1, 11)
    def cond(var, act):
        b = np.clip(np.digitize(a[var], bins) - 1, 0, 9)
        n = np.bincount(b, minlength=10)
        k = np.bincount(b, weights=(a["action"] == act), minlength=10)
        return (k / np.maximum(n, 1)).tolist(), n.tolist()
    s["p_drink_by_hydration"], s["n_by_hydration"] = cond("hydration", 7)
    s["p_eat_by_energy"], s["n_by_energy"] = cond("energy", 6)
    # C. circadian
    tb = (np.vectorize(tod.get)(a["tick"]) * 24).astype(int) % 24
    n = np.bincount(tb, minlength=24)
    s["rest_by_hour"] = (np.bincount(tb, weights=a["action"] == 5, minlength=24) / np.maximum(n, 1)).tolist()
    s["move_by_hour"] = (np.bincount(tb, weights=np.isin(a["action"], [1, 2, 3, 4]), minlength=24) / np.maximum(n, 1)).tolist()
    # D. seasons
    sb = np.vectorize(season.get)(a["tick"]).astype(int)
    s["body_by_season"] = {v: [float(a[v][sb == k].mean()) for k in range(4)] for v in ("energy", "hydration", "warmth")}
    dm = a["died"] == 1
    s["deaths_by_season_cause"] = {DEATH_NAMES[c]: np.bincount(sb[dm & (a["death_cause"] == c)], minlength=4).tolist()
                                   for c in range(len(DEATH_NAMES)) if (dm & (a["death_cause"] == c)).any()}
    # E. distance to fresh water
    dist = W["dist_fresh"][a["y"], a["x"]]
    s["water_distance_median"] = float(np.median(dist))
    s["water_distance_hist"] = np.histogram(np.clip(dist, 0, 40), bins=40, range=(0, 40), density=True)[0].tolist()
    # F. home range (radius of gyration of agents alive >= 4 days)
    order = np.argsort(a["uid"], kind="stable")
    uid, xs, ys = a["uid"][order], a["x"][order].astype(float), a["y"][order].astype(float)
    cut = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1], True])
    rg = []
    for i0, i1 in zip(cut[:-1], cut[1:]):
        if i1 - i0 >= 4 * DAY:
            rg.append(np.sqrt(xs[i0:i1].var() + ys[i0:i1].var()))
    s["home_range_rg_median"] = float(np.median(rg)) if rg else float("nan")
    s["home_range_rg"] = rg
    # G. aggregation vs habitat-matched null (every 40th tick)
    ticks = np.unique(a["tick"])[::40]
    by_t = [np.stack([a["x"][a["tick"] == t], a["y"][a["tick"] == t]], 1).astype(float) for t in ticks]
    habitat, visits = np.unique(np.stack([a["x"], a["y"]], 1), axis=0, return_counts=True)
    habitat = habitat.astype(float)
    s["nn_ratio"], obs_nn, null_nn = nn_ratio(by_t, habitat, rng)
    s["nn_ratio_weighted"], _, null_w = nn_ratio(by_t, habitat, rng, weights=visits.astype(float))
    s["nn_obs_mean"], s["nn_null_mean"], s["nn_null_weighted_mean"] = (float(obs_nn.mean()), float(null_nn.mean()),
                                                                     float(null_w.mean()))
    # H. individual behavioural profiles (agents alive >= 4 days)
    prof, uids = [], []
    act = a["action"][order]
    for i0, i1 in zip(cut[:-1], cut[1:]):
        if i1 - i0 >= 4 * DAY:
            c = np.bincount(act[i0:i1], minlength=len(BASE_ACTION_NAMES))
            prof.append([c[idx].sum() / (i1 - i0) for idx in GROUPS.values()])
            uids.append(uid[i0])
    s["profiles"] = prof
    # J. survival
    dur, died = lifetimes_from_rows(a, start)
    tt, ss = kaplan_meier(dur, died)
    s["km"] = [tt.tolist(), ss.tolist()]
    s["deaths_per_agent_year"] = float(died.sum() / (dur.sum() / YEAR))
    # K. occupancy
    s["occupancy"] = np.histogram2d(a["y"], a["x"], bins=[W["terrain"].shape[0] // 2, W["terrain"].shape[1] // 2],
                                    range=[[0, W["terrain"].shape[0]], [0, W["terrain"].shape[1]]])[0]
    # I. learner internals
    if p.endswith("_s0"):
        v = load(d, "x_value")
        pr = load(d, "x_probs")
        if v is not None:
            sp = np.array([1.0, 1.0, 1.0, 0.8])
            body = np.stack([a["energy"], a["hydration"], a["health"], a["warmth"]], 1)
            drive = np.sqrt((np.maximum(sp - body, 0) ** 2).sum(1))
            s["value_drive_corr"] = float(np.corrcoef(v, drive)[0, 1])
            s["value_by_season"] = [float(v[sb == k].mean()) for k in range(4)]
            ent = -(pr * np.log(pr + 1e-12)).sum(1)
            s["entropy_by_hour"] = (np.bincount(tb, weights=ent, minlength=24) / np.maximum(n, 1)).tolist()
            s["value_by_drive"] = np.histogram(drive, bins=10, range=(0, 1.2), weights=v)[0] / np.maximum(
                np.histogram(drive, bins=10, range=(0, 1.2))[0], 1)
            s["value_by_drive"] = s["value_by_drive"].tolist()
    del a
    return s


def figures(S, W):
    names = [p for p in POLICIES if p in S]
    lab = [POLICIES[p] for p in names]
    # survival
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for p in names:
        tt, ss = S[p]["km"]
        ax.step(np.array(tt) / DAY, ss, where="post", color=COL[p], label=POLICIES[p])
    ax.set(xlabel="days since spawn", ylabel="survival (Kaplan–Meier)")
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(OUT / "survival_km.pdf"); plt.close(fig)
    # action budget
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    left = np.zeros(len(names))
    for (g, c) in zip(GROUPS, GROUP_COL):
        v = np.array([S[p]["action_budget"][g] for p in names])
        ax.barh(lab, v, left=left, color=c, label=g); left += v
    ax.set(xlabel="share of all actions"); ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.45, -0.55))
    fig.savefig(OUT / "action_budget.pdf"); plt.close(fig)
    # homeostatic control
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.3))
    xc = np.arange(10) / 10 + 0.05
    for p in names:
        axs[0].plot(xc, S[p]["p_drink_by_hydration"], "o-", ms=2.5, color=COL[p], label=POLICIES[p])
        axs[1].plot(xc, S[p]["p_eat_by_energy"], "o-", ms=2.5, color=COL[p], label=POLICIES[p])
    axs[0].set(xlabel="own hydration", ylabel="P(drink)", title="Drinking vs thirst")
    axs[1].set(xlabel="own energy", ylabel="P(eat)", title="Eating vs hunger")
    axs[1].legend(frameon=False, fontsize=6)
    fig.savefig(OUT / "homeostatic_control.pdf"); plt.close(fig)
    # circadian
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.2))
    for p in names:
        axs[0].plot(range(24), S[p]["rest_by_hour"], color=COL[p], label=POLICIES[p])
        axs[1].plot(range(24), S[p]["move_by_hour"], color=COL[p])
    for ax_, t in zip(axs, ("P(rest) by hour", "P(move) by hour")):
        ax_.axvspan(0, 6, color="#1c2833", alpha=0.08, lw=0); ax_.axvspan(18, 24, color="#1c2833", alpha=0.08, lw=0)
        ax_.set(xlabel="hour of day (shaded: night)", title=t, xlim=(0, 23))
    axs[0].legend(frameon=False, fontsize=6)
    fig.savefig(OUT / "circadian.pdf"); plt.close(fig)
    # seasonal body state + death causes
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.3))
    seasons = ["spring", "summer", "autumn", "winter"]
    for p in names:
        axs[0].plot(seasons, S[p]["body_by_season"]["energy"], "o-", color=COL[p], label=POLICIES[p])
    axs[0].set(ylabel="mean energy", title="Energy through the year")
    w = 0.16
    for i, p in enumerate(names):
        tot = np.zeros(4)
        for c, v in S[p]["deaths_by_season_cause"].items():
            tot += np.array(v)
        axs[1].bar(np.arange(4) + (i - 2) * w, tot, w, color=COL[p], label=POLICIES[p])
    axs[1].set_xticks(range(4), seasons); axs[1].set(title="Deaths by season (seed 0, 2 years)")
    axs[0].legend(frameon=False, fontsize=6)
    fig.savefig(OUT / "seasonal.pdf"); plt.close(fig)
    # water distance + home range
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for p in names:
        axs[0].plot(np.arange(40) + 0.5, S[p]["water_distance_hist"], color=COL[p], label=POLICIES[p])
        if S[p]["home_range_rg"]:
            axs[1].hist(S[p]["home_range_rg"], bins=30, range=(0, 60), histtype="step", density=True, color=COL[p])
    axs[0].set(xlabel="cells to nearest fresh water", ylabel="density of agent-ticks", title="Distance kept from water")
    axs[1].set(xlabel="radius of gyration (cells)", ylabel="density of agents", title="Home range (agents alive ≥ 4 days)")
    axs[0].legend(frameon=False, fontsize=6)
    fig.savefig(OUT / "water_homerange.pdf"); plt.close(fig)
    # occupancy maps
    fig, axs = plt.subplots(1, len(names), figsize=(7.0, 1.7))
    land = np.isin(W["terrain"][::2, ::2], [0], invert=True)
    for ax_, p in zip(axs, names):
        occ = np.log1p(S[p]["occupancy"])
        ax_.imshow(np.where(land, 0.15, 0.0), cmap="Greys", vmin=0, vmax=1)
        ax_.imshow(np.ma.masked_where(occ == 0, occ), cmap="magma")
        ax_.set_title(POLICIES[p], fontsize=7); ax_.set_xticks([]); ax_.set_yticks([])
        for sp_ in ax_.spines.values():
            sp_.set_visible(False)
    fig.savefig(OUT / "occupancy.pdf"); plt.close(fig)
    # aggregation
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    xb = np.arange(len(names))
    ax.bar(xb - 0.2, [S[p]["nn_ratio"] for p in names], 0.4, color=[COL[p] for p in names], label="habitat null")
    ax.bar(xb + 0.2, [S[p]["nn_ratio_weighted"] for p in names], 0.4, color=[COL[p] for p in names], alpha=0.45,
           hatch="///", label="place-preference null")
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xticks(xb, lab, rotation=25)
    ax.set(ylabel="NN distance / null", title="Aggregation (<1 = closer than null)")
    ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_ylim(0, 1.3)
    fig.savefig(OUT / "aggregation.pdf"); plt.close(fig)
    # individual profiles: PCA over all policies' agents
    from numpy.linalg import svd
    X, who = [], []
    for p in names:
        X += S[p]["profiles"]; who += [p] * len(S[p]["profiles"])
    X = np.array(X); who = np.array(who)
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, sv, Vt = svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for p in names:
        m = who == p
        ax.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.5, color=COL[p], label=POLICIES[p])
    ax.set(xlabel="PC1", ylabel="PC2", title="Individual behavioural profiles")
    ax.legend(frameon=False, fontsize=6, markerscale=3)
    fig.savefig(OUT / "individuals.pdf"); plt.close(fig)
    loadings = {g: Vt[0][i].round(3).tolist() for i, g in enumerate(GROUPS)}
    # learner internals
    lp = [p for p in names if "value_drive_corr" in S[p]]
    if lp:
        fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.2))
        xd = np.linspace(0, 1.2, 11)[:-1] + 0.06
        for p in lp:
            axs[0].plot(xd, S[p]["value_by_drive"], "o-", ms=2.5, color=COL[p], label=f"{POLICIES[p]} (r = {S[p]['value_drive_corr']:.2f})")
            axs[1].plot(range(24), S[p]["entropy_by_hour"], color=COL[p])
        axs[0].set(xlabel="homeostatic drive (distance from set-point)", ylabel="value estimate", title="What the critic tracks")
        axs[1].axvspan(0, 6, color="#1c2833", alpha=0.08, lw=0); axs[1].axvspan(18, 24, color="#1c2833", alpha=0.08, lw=0)
        axs[1].set(xlabel="hour of day", ylabel="policy entropy (nats)", title="Decisiveness through the day")
        axs[0].legend(frameon=False, fontsize=6)
        fig.savefig(OUT / "internals.pdf"); plt.close(fig)
    return loadings


def kmeans_silhouette(X, rng, kmax=5):
    """Best silhouette over k = 2..kmax (plain k-means, standardised profiles): > ~0.5 = distinct types."""
    from scipy.cluster.vq import kmeans2
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    best = (-1.0, 0)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) if len(X) <= 1500 else None
    if D is None:
        idx = rng.choice(len(X), 1500, replace=False)
        X = X[idx]
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    for k in range(2, kmax + 1):
        _, lab = kmeans2(X, k, minit="++", seed=int(rng.integers(1 << 30)))
        if len(np.unique(lab)) < 2:
            continue
        a_ = np.array([D[i, lab == lab[i]].sum() / max(1, (lab == lab[i]).sum() - 1) for i in range(len(X))])
        b_ = np.array([min(D[i, lab == c].mean() for c in np.unique(lab) if c != lab[i]) for i in range(len(X))])
        sil = float(np.mean((b_ - a_) / np.maximum(a_, b_)))
        best = max(best, (sil, k))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    W = generate_world(load_config(), a.seed)
    rng = np.random.default_rng(0)
    S = {}
    for p in POLICIES:
        if not (run_dir(p, a.seed) / "trajectory").exists():
            print("missing", p); continue
        S[p] = analyse(p, a.seed, W, rng)
        print(f"{p:10s} deaths/agent-yr {S[p]['deaths_per_agent_year']:.2f}  water-dist median {S[p]['water_distance_median']:.1f}  "
              f"home-range {S[p]['home_range_rg_median']:.1f}  NN ratio {S[p]['nn_ratio']:.2f} "
              f"(place-weighted {S[p]['nn_ratio_weighted']:.2f})  "
              f"budget {({k: round(v, 3) for k, v in S[p]['action_budget'].items()})}", flush=True)
    loadings = figures(S, W)
    stats = {}
    for p, s in S.items():
        sil = kmeans_silhouette(np.array(s["profiles"]), rng) if len(s["profiles"]) > 20 else (float("nan"), 0)
        stats[p] = {k: v for k, v in s.items() if k not in ("occupancy", "profiles", "home_range_rg", "km",
                                                            "water_distance_hist")}
        stats[p]["n_profiled_agents"] = len(s["profiles"])
        stats[p]["best_silhouette"], stats[p]["best_k"] = sil
    stats["_pc1_loadings"] = loadings
    (ROOT / "paper" / "stats.json").write_text(json.dumps(stats, indent=1, default=float))
    print("silhouettes:", {p: round(stats[p]["best_silhouette"], 3) for p in S})


if __name__ == "__main__":
    main()
