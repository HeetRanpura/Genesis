"""Stage 1 evidence figures -> reports/stage1/*.png

    python experiments/stage1_evidence.py      (run experiments/run_baselines.py first for survival curves)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.baselines import make_baseline  # noqa: E402
from analysis.survival import kaplan_meier  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.constants import CAVE, DEATH_NAMES, OBS_CHANNELS, TERRAIN_NAMES  # noqa: E402
from sim.env import SEASON_NAMES  # noqa: E402
from sim.layout import I_UID, I_X, I_Y  # noqa: E402
from sim.worldgen import generate_world  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "stage1"
OUT.mkdir(parents=True, exist_ok=True)
TCOL = ["#1d3b6e", "#4f8fc0", "#e8d9a0", "#a8c66c", "#6dbb57", "#2f6b38", "#c9a66b", "#8f8a80", "#57525c", "#e0663a"]
TCMAP = ListedColormap(TCOL)
POL_COL = {"random": "#9aa0a6", "greedy": "#e6a23c", "heuristic": "#3a7bd5"}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def terrain_img(ax, T, title=None, trees=None):
    ax.imshow(T, cmap=TCMAP, vmin=-0.5, vmax=9.5, interpolation="nearest")
    cy, cx = np.nonzero(T == CAVE)
    ax.scatter(cx, cy, s=10, c=TCOL[CAVE], edgecolors="k", linewidths=0.4, zorder=3)
    if trees is not None:
        ax.scatter(trees[0], trees[1], s=1.2, c="#ffd23f", zorder=2)
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title)


def fig_terrain(cfg):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, s in zip(axs, [0, 1, 2, 3]):
        W = generate_world(cfg, s)
        frac = np.bincount(W["terrain"].ravel(), minlength=10) / W["terrain"].size
        terrain_img(ax, W["terrain"], f"seed {s} — caves {int((W['terrain'] == CAVE).sum())}, "
                    f"trees {W['tree_x'].size}, fresh {100 * W['fresh'].mean():.1f}%", (W["tree_x"], W["tree_y"]))
    handles = [Patch(color=TCOL[k], label=TERRAIN_NAMES[k]) for k in range(10)]
    handles.append(Patch(color="#ffd23f", label="fruit tree"))
    fig.legend(handles=handles, loc="lower center", ncol=11, frameon=False)
    fig.suptitle("Procedural worlds (256×256): same generator, four seeds", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(OUT / "fig1_terrain_seeds.png")
    plt.close(fig)


def fig_layers(cfg):
    W = generate_world(cfg, 0)
    Wu = generate_world(load_config(overrides={"world": {"resource_layout": "uniform"}}), 0)
    fig, axs = plt.subplots(1, 5, figsize=(20, 4.4))
    for ax, (arr, title, cmap) in zip(axs, [
            (W["elevation"], "elevation (rank-normalised)", "terrain"),
            (W["moisture"], "moisture (land quantile)", "YlGnBu"),
            (np.where(W["ocean"], np.nan, W["fertility"]), "fertility — patchy", "YlGn"),
            (np.where(Wu["ocean"], np.nan, Wu["fertility"]), "fertility — uniform (ablation)", "YlGn")]):
        im = ax.imshow(arr, cmap=cmap, interpolation="nearest")
        ax.set_title(title), ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    hyd = np.zeros(W["terrain"].shape + (3,))
    hyd[:] = (0.93, 0.91, 0.86)
    hyd[W["ocean"]] = (0.12, 0.23, 0.43)
    hyd[W["lake"]] = (0.2, 0.45, 0.8)
    hyd[W["river"]] = (0.1, 0.55, 0.95)
    hyd[W["stream"]] = (0.55, 0.78, 0.98)
    axs[4].imshow(hyd, interpolation="nearest")
    axs[4].set_title("hydrology: ocean (salt), lakes, rivers, streams (dry in drought)")
    axs[4].set_xticks([]), axs[4].set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "fig2_layers.png")
    plt.close(fig)


def timeline_run(cfg, years=2, policy="heuristic", seed=0):
    env = GenesisEnv(cfg, seed=seed)
    pol = make_baseline(policy, env, seed)
    obs = env.observe()
    rows, snaps = [], {}
    tracked = np.flatnonzero(env.alive)[:6]
    tracked_uid = env.I[tracked, I_UID].copy()
    paths = {int(u): [] for u in tracked_uid}
    snap_ticks = {int((1 + (s + 0.5) / 4) * env.year_len): SEASON_NAMES[s] for s in range(4)}
    fire_frames = []
    for k in range(int(years * env.year_len)):
        obs, r, d, info = env.step(pol.act(obs, env.alive))
        if info["born"].any():
            pol.reset(info["born"])
        if k % 20 == 0:
            rows.append(env.metrics())
        for s, u in zip(tracked, tracked_uid):
            if env.I[s, I_UID] == u and env.alive[s] and k < 1500:
                paths[int(u)].append((env.I[s, I_X], env.I[s, I_Y]))
        if env.tick in snap_ticks:
            snaps[snap_ticks[env.tick]] = dict(plant=env.plant.copy(), snow=env.snow.copy(), fire=env.fire.copy(),
                                               agents=(env.I[env.alive, I_X].copy(), env.I[env.alive, I_Y].copy()),
                                               temp=env.global_temp())
    return env, rows, snaps, paths



def _season_shade(ax, ticks, env):
    for y0 in range(0, int(ticks.max()) + 1, env.year_len):
        for s, col in enumerate(["#e9f5e1", "#fff4d6", "#fbe6d4", "#e3eef9"]):
            a = y0 + s * env.season_len
            ax.axvspan(a / env.tpd, (a + env.season_len) / env.tpd, color=col, zorder=0, lw=0)


def plot_timeline(env, rows):
    t = np.array([r["tick"] for r in rows])
    days = t / env.tpd
    g = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    fig, axs = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    ax = axs[0]
    _season_shade(ax, t, env)
    ax.plot(days, g("temp"), c="#c0392b", lw=0.7)
    ax.set_ylabel("sea-level temp °C")
    ax2 = ax.twinx()
    w = g("weather")
    ax2.fill_between(days, 0, (w == 1).astype(float), color="#3a7bd5", alpha=0.25, step="mid", label="rain")
    ax2.fill_between(days, 0, (w == 2).astype(float) * 0.6, color="#888", alpha=0.3, step="mid", label="fog")
    ax2.fill_between(days, 0, g("drought") * 0.9, color="#e67e22", alpha=0.35, step="mid", label="drought")
    ax2.fill_between(days, 0, g("cold_snap") * 0.9, color="#8e44ad", alpha=0.35, step="mid", label="cold snap")
    ax2.fill_between(days, 0, g("famine") * 0.5, color="#7f6000", alpha=0.25, step="mid", label="famine")
    ax2.set_yticks([]), ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", ncol=5, fontsize=7, frameon=False)
    ax.set_title("Climate: seasons (shaded spring/summer/autumn/winter), day/night, weather, disturbances")
    ax = axs[1]
    _season_shade(ax, t, env)
    ax.plot(days, g("plant_total"), c="#2e7d32", label="plant biomass (sum)")
    ax.set_ylabel("plant biomass")
    ax2 = ax.twinx()
    ax2.plot(days, g("fruit_total"), c="#f39c12", label="fruit on trees")
    ax2.set_ylabel("fruit")
    ax.set_title("Vegetation: seasonal regrowth + grazing")
    ax = axs[2]
    _season_shade(ax, t, env)
    for k, c in [("small_prey", "#8d6e63"), ("large_prey", "#5d4037"), ("predators", "#b71c1c")]:
        ax.plot(days, g(k), c=c, label=k)
    ax.set_yscale("log"), ax.set_ylabel("count"), ax.legend(frameon=False, fontsize=8)
    ax.set_title("Fauna (own dynamics: grazing, fleeing, herding, predation, reproduction, immigration floor)")
    ax = axs[3]
    _season_shade(ax, t, env)
    for k, c in [("energy", "#e67e22"), ("hydration", "#2980b9"), ("health", "#27ae60"), ("warmth", "#c0392b")]:
        ax.plot(days, g(f"mean_{k}"), c=c, label=k, lw=0.9)
    ax.set_ylim(0, 1.02), ax.set_ylabel("mean body state"), ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.set_title("Heuristic-baseline agents: homeostatic variables (population mean)")
    ax = axs[4]
    _season_shade(ax, t, env)
    for k, c in [("fire_cells", "#e74c3c"), ("flood_cells", "#3498db"), ("snow_cells", "#95a5a6")]:
        ax.plot(days, g(k), c=c, label=k)
    dc = {n: np.diff(g(f"deaths_{n}"), prepend=0) for n in DEATH_NAMES}
    ax2 = ax.twinx()
    ax2.plot(days, np.cumsum(sum(dc.values())), c="k", lw=1, label="cumulative deaths")
    ax2.set_ylabel("cumulative deaths")
    ax.set_ylabel("cells"), ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("day")
    ax.set_title("Disturbance footprint and mortality")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_timeline.png")
    plt.close(fig)


def plot_seasons(env, snaps):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, s in zip(axs, SEASON_NAMES):
        if s not in snaps:
            continue
        sn = snaps[s]
        img = np.zeros(env.terrain.shape + (3,))
        water = np.isin(env.terrain, [0, 1])
        p = np.clip(sn["plant"] / 0.9, 0, 1)
        img[..., 0] = 0.78 - 0.55 * p
        img[..., 1] = 0.70 - 0.15 * p
        img[..., 2] = 0.52 - 0.35 * p
        img[env.terrain == 8] = (0.35, 0.33, 0.37)
        img[water] = (0.15, 0.3, 0.55)
        sw = sn["snow"][..., None]
        img = img * (1 - sw) + sw * 0.97
        img[sn["fire"] > 0] = (1.0, 0.35, 0.1)
        img[sn["fire"] < 0] = img[sn["fire"] < 0] * 0.4
        ax.imshow(img, interpolation="nearest")
        ax.scatter(*sn["agents"], s=2, c="#ff00aa")
        ax.set_title(f"{s} (year 2), sea-level {sn['temp']:.0f}°C")
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Plant biomass across seasons (green = dense), snow (white), burn scars (dark), agents (magenta)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_seasons.png")
    plt.close(fig)


def plot_paths(env, paths):
    fig, ax = plt.subplots(figsize=(7, 7))
    terrain_img(ax, env.terrain, "Heuristic agents: first 1,500 ticks of six tracked lives")
    for (u, p), c in zip(paths.items(), plt.cm.tab10.colors):
        if len(p) > 1:
            p = np.array(p)
            ax.plot(p[:, 0], p[:, 1], c=c, lw=1.2)
            ax.scatter(p[0, 0], p[0, 1], c=[c], s=25, marker="o", edgecolors="k")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_paths.png")
    plt.close(fig)


def plot_survival():
    root = ROOT / "runs" / "stage1_baselines"
    js = ROOT / "reports" / "stage1" / "baselines.json"
    if not js.exists():
        print("skip survival: run experiments/run_baselines.py first")
        return
    data = json.loads(js.read_text())
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    ypd = 120
    for p in ["random", "greedy", "heuristic"]:
        D, O, C = [], [], []
        for f in sorted(root.glob(f"{p}_s*/lifetimes.npz")):
            z = np.load(f)
            D.append(z["dur"]), O.append(z["observed"]), C.append(z["cause"])
        D, O, C = np.concatenate(D), np.concatenate(O), np.concatenate(C)
        tt, ss = kaplan_meier(D, O)
        axs[0].step(tt / ypd, ss, where="post", c=POL_COL[p], label=f"{p} (n={len(D)})")
        counts = np.array([(C == k).sum() for k in range(len(DEATH_NAMES))], float)
        axs[2].barh([f"{p}"], [0], color="w")
        left = 0.0
        for k, col in enumerate(["#e67e22", "#2980b9", "#7fb3d5", "#e74c3c", "#7b241c", "#7f8c8d", "#000", "#ccc"]):
            frac = counts[k] / max(1, counts.sum())
            axs[2].barh(p, frac, left=left, color=col, label=DEATH_NAMES[k] if p == "random" else None)
            left += frac
    axs[0].set_xlabel("days since spawn"), axs[0].set_ylabel("Kaplan-Meier survival")
    axs[0].set_title("Survival by baseline (5 seeds × 2 years, 256 agents)")
    axs[0].legend(frameon=False)
    s = data["summary"]
    names = list(s)
    m = [s[n]["deaths_per_agent_year"][0] for n in names]
    lo = [m[i] - s[n]["deaths_per_agent_year"][1] for i, n in enumerate(names)]
    hi = [s[n]["deaths_per_agent_year"][2] - m[i] for i, n in enumerate(names)]
    axs[1].bar(names, m, yerr=[lo, hi], color=[POL_COL[n] for n in names], capsize=4)
    for i, n in enumerate(names):
        for r in data["per_run"]:
            if r["policy"] == n:
                axs[1].scatter(i + 0.25, r["deaths_per_agent_year"], c="k", s=8)
    axs[1].set_ylabel("deaths / agent-year")
    axs[1].text(0.98, 0.95, "mean ± 95% bootstrap CI\ndots = individual seeds", transform=axs[1].transAxes, ha="right", va="top", fontsize=8)
    axs[1].set_title("Mortality rate")
    axs[2].set_title("Causes of death (fraction)")
    axs[2].legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(OUT / "fig5_baseline_survival.png")
    plt.close(fig)


def plot_observation(cfg):
    env = GenesisEnv(cfg, seed=0)
    pol = make_baseline("heuristic", env, 0)
    obs = env.observe()
    for _ in range(200):
        obs, *_ = env.step(pol.act(obs, env.alive))
    alive = np.flatnonzero(env.alive)
    # pick the agent that sees the most distinct channels
    best = max(alive, key=lambda i: int((obs["grid"][i].reshape(-1, 16).max(0) > 0).sum()))
    g = obs["grid"][best]
    fig, axs = plt.subplots(2, 9, figsize=(18, 4.6))
    x, y = env.I[best, I_X], env.I[best, I_Y]
    R = env.R
    crop = env.terrain[max(0, y - R):y + R + 1, max(0, x - R):x + R + 1]
    axs[0, 0].imshow(crop, cmap=TCMAP, vmin=-0.5, vmax=9.5)
    axs[0, 0].set_title("true terrain")
    for k in range(16):
        ax = axs[(k + 1) // 9, (k + 1) % 9]
        ax.imshow(g[:, :, k], cmap="magma", vmin=0, vmax=255)   # grid is channel-last
        ax.set_title(OBS_CHANNELS[k], fontsize=8)
    axs[1, 8].axis("off")
    for ax in axs.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"Egocentric observation of one agent (11×11, 16 channels, occlusion by mountains/dense forest); "
                 "self (energy, hydration, health, warmth, age): " + ", ".join(f"{v:.2f}" for v in obs["self"][best][:5]), y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_observation.png")
    plt.close(fig)


def plot_disturbances(cfg):
    env = GenesisEnv(cfg, seed=2)
    env.tick = int(1.4 * env.season_len)    # mid-summer
    env.plant[:] = np.maximum(env.plant, env.fertility * 0.8)
    land = np.flatnonzero((env.terrain == 5).ravel())
    fy, fx = divmod(int(land[len(land) // 2]), env.W)
    env.trigger_fire(fx, fy)
    ry, rx = np.nonzero(env.world["river"])
    env.trigger_flood(int(rx[len(rx) // 3]), int(ry[len(ry) // 3]), days=4)
    frames = []
    a = np.zeros(env.N, np.int64)
    for k in range(241):
        env.step(a, compute_obs=False)
        if k in (10, 80, 240):
            frames.append((k, env.fire.copy(), env.flood.copy()))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (k, fire, flood) in zip(axs, frames):
        img = TCMAP(env.terrain / 9.0)[..., :3] * 0.85
        img[flood > 0] = (0.25, 0.6, 1.0)
        img[fire < 0] = (0.15, 0.12, 0.1)
        img[fire > 0] = (1.0, 0.45, 0.05)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"t+{k} ticks: burning {int((fire > 0).sum())}, scar {int((fire < 0).sum())}, "
                     f"flooded {int((flood > 0).sum())}")
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Triggered disturbances (logged): wildfire spreading through forest fuel; river flood on the floodplain")
    fig.tight_layout()
    fig.savefig(OUT / "fig8_disturbances.png")
    plt.close(fig)


def main():
    cfg = load_config()
    fig_terrain(cfg)
    fig_layers(cfg)
    env, rows, snaps, paths = timeline_run(cfg)
    plot_timeline(env, rows)
    plot_seasons(env, snaps)
    plot_paths(env, paths)
    plot_survival()
    plot_observation(cfg)
    plot_disturbances(cfg)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
