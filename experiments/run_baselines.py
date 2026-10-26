"""Run the non-learning baselines across seeds and summarise survival.

    python experiments/run_baselines.py --seeds 5 --years 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.baselines import make_baseline  # noqa: E402
from analysis.survival import bootstrap_ci, hedges_g, km_at, lifetimes  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.constants import DEATH_NAMES  # noqa: E402
from sim.layout import I_UID  # noqa: E402
from sim.logger import RunLogger, append_agent_log  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POLICIES = ["random", "greedy", "heuristic"]


def run_one(policy: str, seed: int, years: float, n_agents: int, cfg_path, out_root: Path) -> dict:
    cfg = load_config(cfg_path, overrides={"agents": {"n_initial": n_agents, "max_agents": n_agents}})
    env = GenesisEnv(cfg, seed=seed)
    pol = make_baseline(policy, env, seed)
    log = RunLogger(out_root / f"{policy}_s{seed}", env, meta=dict(policy=policy, experiment="stage1_baselines"))
    initial = env.I[env.alive, I_UID].copy()
    ticks = int(years * env.year_len)
    obs = env.observe()
    t0 = time.perf_counter()
    for k in range(ticks):
        a = pol.act(obs, env.alive)
        obs, r, died, info = env.step(a)
        if info["born"].any():
            pol.reset(info["born"])
        if k % cfg.logging.metrics_every == 0:
            log.log_metrics({"policy": policy})
        if k % 1000 == 0:
            log.collect_events()
    wall = time.perf_counter() - t0
    log.collect_events()
    ev = np.concatenate(log._events) if log._events else np.zeros((0, 7))
    dur, obsd, cause = lifetimes(ev, initial, env.tick, start_tick=env.spinup_ticks)
    exposure_years = dur.sum() / env.year_len
    deaths = int(obsd.sum())
    res = dict(policy=policy, seed=seed, ticks=ticks, wall_s=round(wall, 1),
               deaths_per_agent_year=deaths / exposure_years,
               km_survival_1yr=km_at(dur, obsd, env.year_len),
               median_lifetime_days=float(np.median(dur) / env.tpd),
               deaths_by_cause={DEATH_NAMES[c]: int((cause == c).sum()) for c in range(len(DEATH_NAMES)) if (cause == c).any()},
               mean_needs={k: env.metrics()[f"mean_{k}"] for k in ["energy", "hydration", "health", "warmth"]})
    log.close(summary=dict(result=res))
    np.savez_compressed(out_root / f"{policy}_s{seed}" / "lifetimes.npz", dur=dur, observed=obsd, cause=cause)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/stage1_baselines")
    args = ap.parse_args()
    out_root = ROOT / args.out
    results = []
    for seed in range(args.seeds):
        for p in POLICIES:
            r = run_one(p, seed, args.years, args.agents, args.config, out_root)
            results.append(r)
            print(f"{p:10s} seed={seed} deaths/agent-yr={r['deaths_per_agent_year']:.2f} "
                  f"S(1yr)={r['km_survival_1yr']:.3f} wall={r['wall_s']}s {r['deaths_by_cause']}", flush=True)
    table = {}
    for p in POLICIES:
        d = [r["deaths_per_agent_year"] for r in results if r["policy"] == p]
        s = [r["km_survival_1yr"] for r in results if r["policy"] == p]
        table[p] = dict(deaths_per_agent_year=bootstrap_ci(d), km_survival_1yr=bootstrap_ci(s), n_seeds=len(d))
    rand = [r["deaths_per_agent_year"] for r in results if r["policy"] == "random"]
    for p in ("greedy", "heuristic"):
        mine = [r["deaths_per_agent_year"] for r in results if r["policy"] == p]
        table[p]["hedges_g_vs_random_deaths"] = hedges_g(mine, rand)
    out = dict(args=vars(args), per_run=results, summary=table)
    rp = ROOT / "reports" / "stage1" / "baselines.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(out, indent=2))
    lines = [f"Baselines, {args.seeds} seeds x {args.years} years, {args.agents} agents, fixed population (respawn).",
             "", "| policy | deaths / agent-year (95% CI) | Kaplan-Meier survival at 1 yr (95% CI) |", "|---|---|---|"]
    for p in POLICIES:
        d, s = table[p]["deaths_per_agent_year"], table[p]["km_survival_1yr"]
        lines.append(f"| {p} | {d[0]:.2f} [{d[1]:.2f}, {d[2]:.2f}] | {s[0]:.3f} [{s[1]:.3f}, {s[2]:.3f}] |")
    causes = {p: {} for p in POLICIES}
    for r in results:
        for k, v in r["deaths_by_cause"].items():
            causes[r["policy"]][k] = causes[r["policy"]].get(k, 0) + v
    lines += ["", "Deaths by cause (summed over seeds): " + "; ".join(f"**{p}** {causes[p]}" for p in POLICIES)]
    append_agent_log("Stage 1 baseline survival", lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
