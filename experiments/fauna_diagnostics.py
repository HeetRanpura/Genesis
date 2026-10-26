"""Is fauna food-limited or cap-limited?

Runs heuristic-baseline agents and records fauna counts, births, immigration and deaths by cause.
A species counts as food-limited when its cap almost never binds (births are not being blocked),
the immigration floor contributes little, and starvation/predation, not the cap, end lives.

    python experiments/fauna_diagnostics.py --seeds 3 --years 3 [--tag before]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.baselines import make_baseline  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.layout import FSTAT_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPECIES = ["small_prey", "large_prey", "predators"]


def run(seed, years, n_agents, overrides=None):
    cfg = load_config(overrides={"agents": {"n_initial": n_agents, "max_agents": n_agents}, **(overrides or {})})
    env = GenesisEnv(cfg, seed=seed)
    pol = make_baseline("heuristic", env, seed)
    obs = env.observe()
    t, counts, plant = [], [], []
    for k in range(int(years * env.year_len)):
        obs, r, d, info = env.step(pol.act(obs, env.alive))
        if info["born"].any():
            pol.reset(info["born"])
        if k % 20 == 0:
            t.append(env.tick)
            counts.append(env.fcount[:3].copy())
            plant.append(float(env.plant.sum()))
    return env, np.array(t), np.array(counts), np.array(plant)


def summarise(env, counts):
    out = {}
    for sp, name in enumerate(SPECIES):
        c = counts[:, sp]
        cap, floor = env.fauna_max[sp], env.fauna_min[sp]
        st = {n: int(v) for n, v in zip(FSTAT_NAMES, env.fstat[sp])}
        deaths = sum(st[n] for n in ("starved", "old_age", "fire", "eaten", "hunted"))
        out[name] = dict(mean=float(c.mean()), min=int(c.min()), max=int(c.max()), cap=int(cap), floor=int(floor),
                         frac_time_at_cap=float((c >= cap - 1).mean()),
                         frac_time_at_floor=float((c <= floor).mean()),
                         immigrant_share_of_recruits=st["immigrants"] / max(1, st["births"] + st["immigrants"]),
                         death_causes={n: round(st[n] / max(1, deaths), 3) for n in
                                       ("starved", "old_age", "fire", "eaten", "hunted")},
                         **st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--tag", default="current")
    args = ap.parse_args()
    out_dir = ROOT / "reports" / "stage1"
    results, series = [], []
    for seed in range(args.seeds):
        env, t, counts, plant = run(seed, args.years, args.agents)
        s = summarise(env, counts)
        results.append(dict(seed=seed, **s))
        series.append((t, counts, plant, env))
        print(f"seed {seed}: " + " | ".join(
            f"{n}: mean {v['mean']:.0f} [{v['min']}-{v['max']}] cap {v['frac_time_at_cap']:.0%} floor "
            f"{v['frac_time_at_floor']:.0%} immig {v['immigrant_share_of_recruits']:.0%} deaths {v['death_causes']}"
            for n, v in s.items()), flush=True)
    agg = {n: {k: float(np.mean([r[n][k] for r in results])) for k in
               ("mean", "frac_time_at_cap", "frac_time_at_floor", "immigrant_share_of_recruits")} for n in SPECIES}
    (out_dir / f"fauna_diagnostics_{args.tag}.json").write_text(json.dumps(dict(args=vars(args), per_seed=results,
                                                                               mean_over_seeds=agg), indent=2))
    fig, axs = plt.subplots(len(series), 1, figsize=(13, 3.2 * len(series)), sharex=True, squeeze=False)
    for ax, (t, counts, plant, env) in zip(axs[:, 0], series):
        days = t / env.tpd
        for y0 in range(0, int(t.max()) + 1, env.year_len):
            for s_, col in enumerate(["#e9f5e1", "#fff4d6", "#fbe6d4", "#e3eef9"]):
                a = y0 + s_ * env.season_len
                ax.axvspan(a / env.tpd, (a + env.season_len) / env.tpd, color=col, lw=0, zorder=0)
        for sp, (name, col) in enumerate(zip(SPECIES, ["#8d6e63", "#4e342e", "#b71c1c"])):
            ax.plot(days, counts[:, sp], c=col, label=name)
            ax.axhline(env.fauna_max[sp], c=col, ls="--", lw=0.7)
            ax.axhline(max(env.fauna_min[sp], 1), c=col, ls=":", lw=0.7)
        ax.set_yscale("log")
        ax2 = ax.twinx()
        ax2.plot(days, plant, c="#2e7d32", lw=0.8, alpha=0.7)
        ax2.set_ylabel("plant biomass", color="#2e7d32")
        ax.set_ylabel("count (log)")
        ax.set_title(f"seed {env.seed}: fauna vs caps (dashed) and floors (dotted) — config '{args.tag}'")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    axs[-1, 0].set_xlabel("day")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig9_fauna_{args.tag}.png")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
