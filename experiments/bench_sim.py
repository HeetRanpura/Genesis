"""Headless throughput benchmark (random actions).

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


def bench(size: int, n_agents: int, steps: int, obs: bool, warmup: int = 300) -> dict:
    cfg = load_config(overrides={"world": {"width": size, "height": size},
                                 "agents": {"n_initial": n_agents, "max_agents": n_agents}})
    t0 = time.perf_counter()
    env = GenesisEnv(cfg, seed=1)
    init_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    acts = [rng.integers(0, env.n_actions, env.N) for _ in range(16)]
    for k in range(warmup):
        env.step(acts[k % 16], compute_obs=obs)
    t0 = time.perf_counter()
    for k in range(steps):
        env.step(acts[k % 16], compute_obs=obs)
    dt = time.perf_counter() - t0
    return dict(world=f"{size}x{size}", agents=n_agents, alive_end=env.population(), obs=obs, steps=steps,
                steps_per_s=round(steps / dt, 1), steps_per_min=round(60 * steps / dt),
                agent_steps_per_s=round(steps * env.population() / dt), init_s=round(init_s, 2),
                fauna=int(env.fcount[:3].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    steps = 500 if args.quick else 3000
    grid = [(256, 256), (256, 1000), (256, 2000), (512, 1000)]
    if args.quick:
        grid = [(256, 1000)]
    rows = []
    for size, n in grid:
        for obs in (False, True):
            r = bench(size, n, steps, obs)
            rows.append(r)
            print(f"{r['world']:>8} agents={r['agents']:5d} obs={str(obs):5s}  {r['steps_per_s']:8.1f} steps/s"
                  f"  {r['steps_per_min']:9d} steps/min  {r['agent_steps_per_s']:10d} agent-steps/s", flush=True)
    out = dict(machine=platform.platform(), processor=platform.processor(), python=platform.python_version(),
               results=rows)
    if not args.quick:
        p = ROOT / "reports" / "stage1" / "bench.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print("wrote", p)


if __name__ == "__main__":
    main()
