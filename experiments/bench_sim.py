"""Headless throughput benchmark (random actions; median of repeated whole-day windows).

    python experiments/bench_sim.py            # full table -> reports/stage1/bench.json
    python experiments/bench_sim.py --quick
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim import GenesisEnv, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def bench(size: int, n_agents: int, steps: int, obs: bool, trials: int = 5, warmup: int = 240) -> dict:
    """Median over `trials` timed windows of `steps` env steps each (steps should span whole days:
    the view radius, and so observation cost, differs between day and night)."""
    cfg = load_config(overrides={"world": {"width": size, "height": size},
                                 "agents": {"n_initial": n_agents, "max_agents": n_agents}})
    t0 = time.perf_counter()
    env = GenesisEnv(cfg, seed=1)
    init_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    acts = [rng.integers(0, env.n_actions, env.N) for _ in range(16)]
    for k in range(warmup):
        env.step(acts[k % 16], compute_obs=obs)
    rates = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for k in range(steps):
            env.step(acts[k % 16], compute_obs=obs)
        rates.append(steps / (time.perf_counter() - t0))
    med = float(np.median(rates))
    return dict(world=f"{size}x{size}", agents=n_agents, alive_end=env.population(), obs=obs, steps=steps,
                trials=trials, steps_per_s=round(med, 1), steps_per_s_min=round(min(rates), 1),
                steps_per_s_max=round(max(rates), 1), steps_per_min=round(60 * med),
                agent_steps_per_s=round(med * env.population()), init_s_incl_spinup=round(init_s, 2),
                fauna=int(env.fcount[:3].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    steps = 1200                       # 10 simulated days per timed window
    grid = [(256, 256), (256, 1000), (256, 2000), (512, 1000)]
    if args.quick:
        grid = [(256, 1000)]
    rows = []
    for size, n in grid:
        for obs in (False, True):
            r = bench(size, n, steps, obs, trials=3 if args.quick else 5)
            rows.append(r)
            print(f"{r['world']:>8} agents={r['agents']:5d} obs={str(obs):5s}  {r['steps_per_s']:8.1f} steps/s"
                  f" [{r['steps_per_s_min']:.0f}-{r['steps_per_s_max']:.0f}]  {r['steps_per_min']:9d} steps/min (median)"
                  f"  {r['agent_steps_per_s']:10d} agent-steps/s", flush=True)
    out = dict(machine=platform.platform(), processor=platform.processor(), python=platform.python_version(),
               results=rows)
    if not args.quick:
        p = ROOT / "reports" / "stage1" / "bench.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print("wrote", p)


if __name__ == "__main__":
    main()
