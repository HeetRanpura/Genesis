"""Population-density sweep for the non-learning baselines.

    python experiments/density_sweep.py --seeds 3 --years 1

Hypothesis (written 2026-10-26 before this script was run; prompted by ONE uncontrolled live-viewer
session, seed 3 with 1,000 heuristic agents, that showed ~5 deaths/agent-yr, almost all thirst, and
small prey crashing while large prey boomed):
  H1  heuristic mortality per agent-year rises with population size (256 -> 512 -> 1000) on a 256^2 world;
  H2  the rise is mostly thirst (fresh-water cells are scarce, one body per cell);
  H3  more agents -> fewer small prey and more large prey at the end of the year (agents hunt only small
      prey; large prey lose their main grazing competitor).
Random and greedy are included as baselines (HARD RULE 5). Numbers are per-seed; CIs by bootstrap.
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
from analysis.survival import bootstrap_ci, lifetimes  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.constants import DEATH_NAMES  # noqa: E402
from sim.layout import I_UID  # noqa: E402
from sim.logger import append_agent_log  # noqa: E402
from sim.recorder import TrajectoryRecorder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POLICIES = ["random", "greedy", "heuristic"]


def run_one(policy: str, seed: int, years: float, n: int) -> dict:
    env = GenesisEnv(load_config(overrides={"agents": {"n_initial": n, "max_agents": n}}), seed=seed)
    pol = make_baseline(policy, env, seed)
    initial = env.I[env.alive, I_UID].copy()
    obs = env.observe()
    rec = (TrajectoryRecorder(env, ROOT / "runs" / "density_sweep" / f"{policy}_n{n}_s{seed}", meta=dict(policy=policy))
           if env.cfg.logging.trajectory else None)
    sp, lp, pr = [], [], []
    t0 = time.perf_counter()
    for k in range(int(years * env.year_len)):
        if rec:
            rec.pre(obs)
        a = pol.act(obs, env.alive)
        obs, r, _, info = env.step(a)
        if rec:
            rec.post(a, r, info, policy_state=getattr(pol, "mode", None))
        if info["born"].any():
            pol.reset(info["born"])
        if k % 40 == 0:
            sp.append(env.fcount[0]); lp.append(env.fcount[1]); pr.append(env.fcount[2])
    if rec:
        rec.close()
    ev = np.concatenate(env.event_log) if env.event_log else np.zeros((0, 7))
    dur, obsd, cause = lifetimes(ev, initial, env.tick, start_tick=env.spinup_ticks)
    deaths = int(obsd.sum())
    return dict(policy=policy, seed=seed, n_agents=n, wall_s=round(time.perf_counter() - t0, 1),
                deaths_per_agent_year=deaths / (dur.sum() / env.year_len),
                thirst_share=float((cause[obsd] == DEATH_NAMES.index("thirst")).mean()) if deaths else 0.0,
                deaths_by_cause={DEATH_NAMES[c]: int((cause == c).sum()) for c in range(len(DEATH_NAMES)) if (cause == c).any()},
                small_prey_mean=float(np.mean(sp)), large_prey_mean=float(np.mean(lp)), predators_mean=float(np.mean(pr)),
                small_prey_end=int(env.fcount[0]), large_prey_end=int(env.fcount[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--years", type=float, default=1.0)
    ap.add_argument("--sizes", default="256,512,1000")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    res = []
    for n in sizes:
        for seed in range(a.seeds):
            for p in POLICIES:
                r = run_one(p, seed, a.years, n)
                res.append(r)
                print(f"N={n:5d} {p:9s} s{seed} deaths/agent-yr={r['deaths_per_agent_year']:.2f} thirst={r['thirst_share']:.2f} "
                      f"small={r['small_prey_mean']:.0f} large={r['large_prey_mean']:.0f} ({r['wall_s']}s)", flush=True)
    summ = {}
    for p in POLICIES:
        for n in sizes:
            rr = [r for r in res if r["policy"] == p and r["n_agents"] == n]
            summ[f"{p}_{n}"] = {k: bootstrap_ci([r[k] for r in rr]) for k in
                                ("deaths_per_agent_year", "thirst_share", "small_prey_mean", "large_prey_mean", "predators_mean")}
    out = ROOT / "reports" / "stage1" / "density_sweep.json"
    out.write_text(json.dumps(dict(args=vars(a), per_run=res, summary=summ), indent=2))
    lines = [f"Density sweep, {a.seeds} seeds x {a.years} yr per cell, 256x256 world, fixed population (respawn). "
             "Mean [95% bootstrap CI] over seeds. Hypotheses H1-H3 are in `experiments/density_sweep.py` (written before running).",
             "", "| policy | agents | deaths / agent-yr | thirst share of deaths | small prey (mean) | large prey (mean) | predators (mean) |",
             "|---|---|---|---|---|---|---|"]
    for p in POLICIES:
        for n in sizes:
            s = summ[f"{p}_{n}"]
            f = lambda k, d=2: f"{s[k][0]:.{d}f} [{s[k][1]:.{d}f}, {s[k][2]:.{d}f}]"  # noqa: E731
            lines.append(f"| {p} | {n} | {f('deaths_per_agent_year')} | {f('thirst_share')} | {f('small_prey_mean', 0)} | "
                         f"{f('large_prey_mean', 0)} | {f('predators_mean', 1)} |")
    append_agent_log("Baseline population-density sweep", lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
