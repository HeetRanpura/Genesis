"""Preregistered H6/H7 analyses + exploratory memory/value analyses for the GENESIS preprint.

    .venv/bin/python reports/stage2/pilot_preprint/analysis_h6_h7.py

H7a  aggregation: Clark-Evans R of live agents, observed vs a null that relocates every agent to a random passable
     cell with the same distance to fresh water (keeps the water confound). All 5 policies, worlds 0-4 x 2 yr.
     Baselines are re-run with the identical protocol (deterministic, so identical to the canonical runs) and only
     positions are sampled; learner positions come from the evaluation flight recordings.
H7b  cooperative hunting: large-prey attacks by >= 2 distinct agents within 6 ticks vs a shuffled-timing null.
H6   linear probes (preregistered thresholds: R^2 >= 0.5 for own hydration/energy, AUC >= 0.7 for predator visible),
     on the GRU hidden state vs the memoryless core output; held-out split by agent. Replays world 0, first year.
Exploratory: (i) memory test -- decode distance to the nearest fresh water on ticks with NO water in view, from the
     GRU state vs the same network's pre-core encoding; (ii) value calibration -- AUC of the recorded value estimate
     for "dies within 200 ticks".
Writes stats_h6_h7.json, fig_aggregation, fig_probes, fig_value, fig_paths next to this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from agents.baselines import make_baseline  # noqa: E402
from pilot_net import ActorCritic  # noqa: E402
from pilot_runs import checkpoint, recording  # noqa: E402
from analysis.survival import bootstrap_ci  # noqa: E402
from analysis.trajectories import load, replay  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.constants import DEEP_WATER, MOUNTAIN  # noqa: E402
from sim.layout import I_UID, I_X, I_Y  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import COLOR, LABEL, POLICIES, INK, INK2, plt, save  # noqa: E402

OUT = Path(__file__).resolve().parent
EV_ATTACK_LARGE, EV_KILL_LARGE, EV_DEATH = 19, 6, 1
SAMPLE = 120                      # one position sample per sim day
STATS: dict = {}


# ------------------------------------------------------------------------------------------- geometry
def world_geometry(seed):
    env = GenesisEnv(load_config(None, overrides={"agents": {"n_initial": 256, "max_agents": 256}}), seed=seed)
    passable = (env.terrain != DEEP_WATER) & (env.terrain != MOUNTAIN)
    dist = ndimage.distance_transform_edt(~env.fresh)
    return env, passable, dist


BINS = np.array([0, 1.5, 2.5, 3.5, 5, 8, 13, 21, 34, 55, 1e9])


def clark_evans(xy, area):
    n = len(xy)
    if n < 3:
        return np.nan
    d, _ = cKDTree(xy).query(xy, k=2)
    return d[:, 1].mean() / (0.5 * np.sqrt(area / n))


def ce_with_null(xy, passable, dist, rng, reps=20):
    area = passable.sum()
    obs = clark_evans(xy.astype(float), area)
    py, px = np.nonzero(passable)
    pb = np.digitize(dist[py, px], BINS)
    ab = np.digitize(dist[xy[:, 1], xy[:, 0]], BINS)
    pools = {b: np.flatnonzero(pb == b) for b in np.unique(ab)}
    nulls = []
    for _ in range(reps):
        pick = np.empty(len(xy), np.int64)
        for b in np.unique(ab):
            m = ab == b
            pick[m] = rng.choice(pools[b], m.sum(), replace=pools[b].size < m.sum())
        jit = rng.uniform(-0.01, 0.01, (len(xy), 2))         # break exact ties on shared cells
        nulls.append(clark_evans(np.c_[px[pick], py[pick]] + jit, area))
    return obs, float(np.mean(nulls)), float(np.mean(dist[xy[:, 1], xy[:, 0]]))


def positions_baseline(policy, seed, years=2.0):
    cfg = load_config(None, overrides={"agents": {"n_initial": 256, "max_agents": 256}})
    env = GenesisEnv(cfg, seed=seed)
    pol = make_baseline(policy, env, seed)
    obs = env.observe()
    out = []
    for k in range(int(years * env.year_len)):
        if k % SAMPLE == 0:
            al = env.alive
            out.append(np.c_[env.I[al, I_X], env.I[al, I_Y]].copy())
        a = pol.act(obs, env.alive)
        obs, _, _, info = env.step(a)
        if info["born"].any():
            pol.reset(info["born"])
    return out


def positions_learner(arm, seed):
    a = load(recording(arm, seed), "agents")
    t0 = a["tick"].min()
    out = []
    for t in range(t0, a["tick"].max() + 1, SAMPLE):
        r = a[a["tick"] == t]
        out.append(np.c_[r["x"].astype(np.int64), r["y"].astype(np.int64)])
    return out


def aggregation():
    rng = np.random.default_rng(0)
    res = {p: dict(R_obs=[], R_null=[], ratio=[], dist_water=[]) for p in POLICIES}
    for seed in range(5):
        _, passable, dist = world_geometry(seed)
        for p in POLICIES:
            pos = positions_learner(p, seed) if p in ("gru", "ff") else positions_baseline(p, seed)
            vals = np.array([ce_with_null(xy, passable, dist, rng) for xy in pos])
            res[p]["R_obs"].append(float(np.nanmean(vals[:, 0])))
            res[p]["R_null"].append(float(np.nanmean(vals[:, 1])))
            res[p]["ratio"].append(float(np.nanmean(vals[:, 0] / vals[:, 1])))
            res[p]["dist_water"].append(float(np.nanmean(vals[:, 2])))
        print("aggregation world", seed, {p: round(res[p]["ratio"][-1], 3) for p in POLICIES}, flush=True)
    for p in POLICIES:
        res[p]["ratio_ci"] = bootstrap_ci(res[p]["ratio"])
    STATS["aggregation"] = res

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.9))
    for i, p in enumerate(POLICIES[::-1]):
        ro, rn = np.array(res[p]["R_obs"]), np.array(res[p]["R_null"])
        axes[0].scatter(rn, np.full(5, i) - 0.12, s=14, marker="o", facecolor="white", edgecolor=COLOR[p], lw=1.2, zorder=3)
        axes[0].scatter(ro, np.full(5, i) + 0.12, s=14, color=COLOR[p], lw=0, zorder=3)
        m, lo, hi = res[p]["ratio_ci"]
        axes[1].plot([lo, hi], [i, i], color=COLOR[p], lw=2, solid_capstyle="round")
        axes[1].scatter([m], [i], s=34, color=COLOR[p], edgecolor="white", lw=1.2, zorder=3)
        axes[1].text(hi + 0.01, i, f"{m:.2f}", va="center", fontsize=7, color=INK)
    for ax in axes:
        ax.set_yticks(range(5), [LABEL[p] for p in POLICIES[::-1]])
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    axes[0].set_xlabel("Clark–Evans R (filled: observed, open: water-matched null)")
    axes[1].axvline(1.0, color=INK2, lw=0.8, ls=":")
    axes[1].set_xlabel("R observed / R null  (<1 = aggregated beyond water)")
    axes[0].set_title("a", loc="left", fontsize=8)
    axes[1].set_title("b", loc="left", fontsize=8)
    save(fig, "fig_aggregation")


# ------------------------------------------------------------------------------------------- coop hunts
def coop_hunts():
    rng = np.random.default_rng(1)
    out = {}
    for p in POLICIES:
        tot = dict(attacks=0, multi=0, kills_large=0, null_multi=[], agent_years=0.0)
        null_runs = []
        for seed in range(5):
            if p in ("gru", "ff"):
                ev = load(recording(p, seed), "events")
            else:
                ev = np.load(ROOT / f"runs/stage1_baselines/{p}_s{seed}/events.npy")
            att = ev[ev[:, 1] == EV_ATTACK_LARGE]
            tot["attacks"] += len(att)
            tot["kills_large"] += int((ev[:, 1] == EV_KILL_LARGE).sum())
            tot["multi"] += multi_episodes(att)
            t_lo, t_hi = (att[:, 0].min(), att[:, 0].max()) if len(att) else (0, 0)
            null_runs.append((att, t_lo, t_hi))
        nulls = []
        for _ in range(1000):
            n = 0
            for att, lo, hi in null_runs:
                if len(att) < 2:
                    continue
                sh = att.copy()
                sh[:, 0] = rng.permutation(att[:, 0])
                n += multi_episodes(sh)
            nulls.append(n)
        nulls = np.array(nulls)
        tot["null_mean"] = float(nulls.mean())
        tot["p_value"] = float((np.sum(nulls >= tot["multi"]) + 1) / (len(nulls) + 1))
        tot.pop("null_multi")
        tot.pop("agent_years")
        out[p] = tot
    STATS["coop_hunts"] = out
    print("coop hunts", {p: (v["attacks"], v["multi"], round(v["null_mean"], 2), v["kills_large"]) for p, v in out.items()})


def multi_episodes(att, window=6):
    """Attacks on the same animal by >= 2 distinct agents within `window` ticks (counted once per pair window)."""
    n = 0
    for j in np.unique(att[:, 3]):
        a = att[att[:, 3] == j]
        a = a[np.argsort(a[:, 0])]
        for i in range(len(a)):
            later = a[(a[:, 0] > a[i, 0]) & (a[:, 0] <= a[i, 0] + window)]
            if np.any(later[:, 2] != a[i, 2]):
                n += 1
    return n


# ------------------------------------------------------------------------------------------- probes
def load_net(arm):
    ck = checkpoint(arm)
    env = GenesisEnv(load_config(None, overrides={"agents": {"n_initial": 256, "max_agents": 256}}), seed=0)
    net = ActorCritic(env.n_actions, env.obs_shapes, ck["cfg"]["hidden"], ck["cfg"]["recurrent"])
    net.load_state_dict(ck["net"])
    net.eval()
    return net


@torch.no_grad()
def collect(arm, years=1.0, every=8):
    run = recording(arm, 0)
    net = load_net(arm)
    rec_v = load(run, "x_value")
    rec_a = load(run, "agents")
    h = None
    prev_uid = None
    X_core, X_enc, Y, V_re, V_rec = [], [], [], [], []
    row0 = 0
    _, _, dist = world_geometry(0)
    t_end = None
    for t, env, obs in replay(run, with_obs=True, verify=True):
        if t_end is None:
            t_end = t + int(years * env.year_len)
        if t >= t_end:
            break
        al = np.flatnonzero(env.alive)
        n_rows = len(al)
        uid = env.I[:, I_UID].copy()
        if h is None:
            h = torch.zeros(env.N, net.hidden)
            prev_uid = np.full(env.N, -1)
        reset = torch.from_numpy(((uid != prev_uid) | ~env.alive).astype(np.float32))
        prev_uid = np.where(env.alive, uid, -1)
        g = torch.from_numpy(obs["grid"])
        s = torch.from_numpy(obs["self"])
        nb = torch.from_numpy(obs["neighbors"])
        hd = torch.from_numpy(obs["heard"])
        x = net.encode(g, s, nb, hd)
        if net.recurrent:
            h = net.core(x, h * (1.0 - reset).unsqueeze(1))
            z = h
        else:
            z = torch.relu(net.core(x))
        v = net.v(z).squeeze(1)
        if (t % every) == 0:
            grid = obs["grid"][al]
            pred = (grid[..., 11] > 0).any(axis=(1, 2))
            water_vis = (grid[..., 2] > 0).any(axis=(1, 2))
            xs, ys = env.I[al, I_X], env.I[al, I_Y]
            Y.append(np.c_[env.A[al, 1], env.A[al, 0], pred, water_vis, dist[ys, xs], uid[al]])
            X_core.append(z[al].numpy().astype(np.float32))
            X_enc.append(x[al].numpy().astype(np.float32))
            V_re.append(v[al].numpy())
            V_rec.append(rec_v[row0:row0 + n_rows])
        row0 += n_rows
    del rec_a
    Y = np.concatenate(Y)
    return dict(core=np.concatenate(X_core), enc=np.concatenate(X_enc), Y=Y,
                v_re=np.concatenate(V_re), v_rec=np.concatenate(V_rec))


def split_by_agent(uid, frac=0.2, seed=0):
    u = np.unique(uid)
    test = np.random.default_rng(seed).choice(u, int(len(u) * frac), replace=False)
    return ~np.isin(uid, test), np.isin(uid, test)


def ridge_r2(X, y, tr, te, lam=1.0):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    A = np.c_[(X - mu) / sd, np.ones(len(X))]
    W = np.linalg.solve(A[tr].T @ A[tr] + lam * np.eye(A.shape[1]), A[tr].T @ y[tr])
    p = A[te] @ W
    return float(1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())


def auc(score, label):
    label = label.astype(bool)
    if label.all() or (~label).all():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_auc(X, y, tr, te, l2=1e-3, steps=200):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps)
    trt = torch.from_numpy(tr)

    def closure():
        opt.zero_grad()
        z = Xt[trt] @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt[trt]) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        s = (Xt[torch.from_numpy(te)] @ w + b).numpy()
    return auc(s, y[te])


def probes():
    out = {}
    for arm in ("gru", "ff"):
        D = collect(arm)
        Y = D["Y"]
        tr, te = split_by_agent(Y[:, 5])
        rng = np.random.default_rng(0)
        res = {"n_samples": int(len(Y)), "n_test_agents": int(len(np.unique(Y[te, 5]))),
               "value_replay_vs_recorded_corr": float(np.corrcoef(D["v_re"], D["v_rec"])[0, 1]),
               "value_replay_max_abs_diff": float(np.abs(D["v_re"] - D["v_rec"]).max())}
        for feat in ("core", "enc"):
            X = D[feat]
            r = {"hydration_r2": ridge_r2(X, Y[:, 0], tr, te), "energy_r2": ridge_r2(X, Y[:, 1], tr, te),
                 "predator_auc": logistic_auc(X, Y[:, 2], tr, te)}
            nw = Y[:, 3] == 0                      # no water in view: memory test (exploratory)
            r["water_dist_r2_no_water_in_view"] = ridge_r2(X[nw], np.log1p(Y[nw, 4]), tr[nw], te[nw])
            ys = rng.permutation(Y[:, 0])
            r["shuffled_hydration_r2"] = ridge_r2(X, ys, tr, te)
            res[feat] = r
        res["predator_visible_rate"] = float(Y[:, 2].mean())
        res["no_water_in_view_rate"] = float((Y[:, 3] == 0).mean())
        out[arm] = res
        print("probes", arm, json.dumps(res, indent=0)[:600], flush=True)
    STATS["probes"] = out

    labels = ["own hydration\n(R²)", "own energy\n(R²)", "predator visible\n(AUC)", "water distance,\nnone in view (R²)"]
    keys = ["hydration_r2", "energy_r2", "predator_auc", "water_dist_r2_no_water_in_view"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.9), sharey=False)
    series = [("gru", "core", "GRU state h", COLOR["gru"], None), ("gru", "enc", "GRU arm, pre-core", COLOR["gru"], "////"),
              ("ff", "core", "memoryless core", COLOR["ff"], None), ("ff", "enc", "memoryless, pre-core", COLOR["ff"], "////")]
    thr = {"hydration_r2": 0.5, "energy_r2": 0.5, "predator_auc": 0.7}
    for ax, k, lab in zip(axes, keys, labels):
        for i, (arm, f, name, col, hatch) in enumerate(series):
            val = out[arm][f][k]
            ax.bar(i, val, width=0.78, color=col if hatch is None else "white", edgecolor=col, hatch=hatch, lw=1.2,
                   label=name)
            ax.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=6, color=INK)
        if k in thr:
            ax.axhline(thr[k], color=INK2, lw=0.8, ls=":")
        ax.set_xticks([])
        ax.set_xlabel(lab, fontsize=7)
        ax.set_ylim(0.5 if k == "predator_auc" else 0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=6, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.32), handlelength=1.2, columnspacing=0.8)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig_probes")


# ------------------------------------------------------------------------------------------- value calibration
def value_calibration(horizon=200):
    out = {}
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for arm in ("gru", "ff"):
        V, L = [], []
        for seed in range(5):
            run = recording(arm, seed)
            a = load(run, "agents")
            v = load(run, "x_value")
            died = a["died"].astype(bool)
            dtick = {int(u): int(t) for u, t in zip(a["uid"][died], a["tick"][died])}
            end = int(a["tick"].max())
            td = np.array([dtick.get(int(u), -1) for u in a["uid"]])
            lab = (td >= 0) & (td - a["tick"] <= horizon)
            known = lab | (end - a["tick"] > horizon)
            sel = np.flatnonzero(known)[::4]
            V.append(v[sel])
            L.append(lab[sel])
        V, L = np.concatenate(V), np.concatenate(L)
        q = np.quantile(V, np.linspace(0, 1, 11))
        b = np.clip(np.digitize(V, q[1:-1]), 0, 9)
        rate = [float(L[b == i].mean()) for i in range(10)]
        out[arm] = dict(auc_low_value_predicts_death=auc(-V, L), base_rate=float(L.mean()), n=int(len(V)),
                        death_rate_by_value_decile=rate)
        ax.plot(range(1, 11), rate, color=COLOR[arm], lw=2, marker="o", ms=4, mec="white", mew=0.8,
                label=f"{LABEL[arm]} (AUC {out[arm]['auc_low_value_predicts_death']:.2f})")
    ax.set_xlabel("value-estimate decile (1 = lowest)")
    ax.set_ylabel(f"P(death within {horizon} ticks)")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=6.5)
    save(fig, "fig_value")
    STATS["value_calibration"] = out
    print("value", out)


# ------------------------------------------------------------------------------------------- example paths
def paths():
    from matplotlib.colors import ListedColormap
    env, _, _ = world_geometry(0)
    terr = env.terrain
    tcol = ["#c9d6e3", "#9cc3e6", "#efe6c8", "#dfe9c6", "#d3e6bd", "#b9cfa8", "#e6dcc8", "#d6d2cb", "#a9a49c", "#a9a49c"]
    groups = {"move": [1, 2, 3, 4], "eat": [6], "drink": [7], "rest": [5]}
    gcol = {"move": INK2, "eat": "#eb6834", "drink": "#2a78d6", "rest": "#c98500"}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.3))
    for ax, arm in zip(axes, ("gru", "ff")):
        a = load(recording(arm, 0), "agents")
        u, cnt = np.unique(a["uid"], return_counts=True)
        pick = u[np.argsort(cnt)[-3:]]
        rows = np.concatenate([a[a["uid"] == k] for k in pick])
        rx, ry = rows["x"].astype(np.int64), rows["y"].astype(np.int64)
        x0, x1 = rx.min() - 12, rx.max() + 12
        y0, y1 = ry.min() - 12, ry.max() + 12
        ax.imshow(terr, cmap=ListedColormap(tcol), vmin=-0.5, vmax=9.5, interpolation="nearest")
        for k in pick:
            r = a[a["uid"] == k]
            ax.plot(r["x"], r["y"], color=INK, lw=0.4, alpha=0.5)
            for g, ids in groups.items():
                m = np.isin(r["action"], ids)
                ax.scatter(r["x"][m], r["y"][m], s=1.5, color=gcol[g], lw=0, label=g if k == pick[0] else None)
        ax.set_xlim(max(0, x0), min(255, x1))
        ax.set_ylim(min(255, y1), max(0, y0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        life = [int(cnt[u == k][0]) for k in pick]
        ax.set_title(f"{LABEL[arm]}: 3 longest-lived agents ({', '.join(f'{x / 120:.0f} d' for x in life)})",
                     fontsize=7.5, loc="left")
    axes[1].legend(fontsize=6.5, markerscale=4, loc="lower right", framealpha=0.9, frameon=True)
    save(fig, "fig_paths")


def probes_fig_only():
    """Redraw fig_probes from stats_h6_h7.json without replaying."""
    out = json.loads((OUT / "stats_h6_h7.json").read_text())["probes"]
    STATS["probes"] = out
    labels = ["own hydration\n(R²)", "own energy\n(R²)", "predator visible\n(AUC)", "water distance,\nnone in view (R²)"]
    keys = ["hydration_r2", "energy_r2", "predator_auc", "water_dist_r2_no_water_in_view"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.9))
    series = [("gru", "core", "GRU state h", COLOR["gru"], None), ("gru", "enc", "GRU arm, pre-core", COLOR["gru"], "////"),
              ("ff", "core", "memoryless core", COLOR["ff"], None), ("ff", "enc", "memoryless, pre-core", COLOR["ff"], "////")]
    thr = {"hydration_r2": 0.5, "energy_r2": 0.5, "predator_auc": 0.7}
    for ax, k, lab in zip(axes, keys, labels):
        for i, (arm, f, name, col, hatch) in enumerate(series):
            val = out[arm][f][k]
            ax.bar(i, val, width=0.78, color=col if hatch is None else "white", edgecolor=col, hatch=hatch, lw=1.2, label=name)
            ax.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=6, color=INK)
        if k in thr:
            ax.axhline(thr[k], color=INK2, lw=0.8, ls=":")
        ax.set_xticks([])
        ax.set_xlabel(lab, fontsize=7)
        ax.set_ylim(0.5 if k == "predator_auc" else 0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=6, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.32), handlelength=1.2, columnspacing=0.8)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig_probes")


def aggregation_fig():
    s = json.loads((OUT / "stats_h6_h7.json").read_text())
    agg, bl = s["aggregation_strict"], s["blind_ablation"]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.95), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for ax, key, title in ((axes[0], "ratio_water_ci", "a  null: same distance to water"),
                           (axes[1], "ratio_habitat_ci", "b  null: water × terrain × fertility")):
        for i, p in enumerate(POLICIES[::-1]):
            m, lo, hi = agg[p][key]
            pts = agg[p]["R_water" if "water" in key else "R_habitat"]
            ax.plot([lo, hi], [i, i], color=COLOR[p], lw=2, solid_capstyle="round")
            ax.scatter(pts, np.full(len(pts), i) + 0.22, s=8, color=COLOR[p], alpha=0.6, lw=0)
            ax.scatter([m], [i], s=34, color=COLOR[p], edgecolor="white", lw=1.2, zorder=3)
            ax.text(lo - 0.012, i, f"{m:.2f}", va="center", ha="right", fontsize=7, color=INK)
        ax.axvline(1.0, color=INK2, lw=0.8, ls=":")
        ax.set_xlim(0.66, 1.06)
        ax.set_yticks(range(5), [LABEL[p] for p in POLICIES[::-1]])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("R observed / R null")
        ax.set_title(title, loc="left", fontsize=7.5)
    axes[1].set_yticklabels([])
    ax = axes[2]
    for i, arm in enumerate(("gru", "ff")):
        for j, cond in enumerate(("intact", "blind")):
            v = bl[f"{arm}_{cond}"]
            x = i * 2.6 + j
            ax.bar(x, v["ratio"], width=0.8, color=COLOR[arm] if cond == "intact" else "white", edgecolor=COLOR[arm],
                   hatch=None if cond == "intact" else "////", lw=1.2)
            ax.scatter([x, x], v["ratio_per_world"], s=8, color=INK, zorder=3, lw=0)
            ax.text(x, v["ratio"] + 0.03, f"{v['ratio']:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks([0, 1, 2.6, 3.6], ["GRU", "GRU\nblind", "mem.", "mem.\nblind"], fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_ylabel("R obs / R null (water)", fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set_title("c  blind to other agents", loc="left", fontsize=7.5)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig_aggregation")


if __name__ == "__main__" and len(sys.argv) > 1:
    for name in sys.argv[1:]:
        globals()[name]()
    sys.exit(0)

if __name__ == "__main__":
    coop_hunts()
    value_calibration()
    paths()
    probes()
    aggregation()
    (OUT / "stats_h6_h7.json").write_text(json.dumps(STATS, indent=1, default=float))
    print("done")
