"""Exploratory causal test of learner aggregation: evaluate the trained policies with the agent channel of the grid
(channel 12) and the neighbour features zeroed ("socially blind"), on held-out worlds 0-1 x 1 yr, and compare
aggregation (Clark-Evans R / water-matched null R) and mortality with the intact policy under the identical protocol.

    .venv/bin/python reports/stage2/pilot_preprint/blind_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from pilot_net import PilotPolicy as LearnedPolicy  # noqa: E402
from pilot_runs import checkpoint  # noqa: E402
from analysis_h6_h7 import SAMPLE, ce_with_null, world_geometry  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.layout import I_X, I_Y  # noqa: E402

C_AGENT = 12


def run(arm, seed, blind, years=1.0):
    cfg = load_config(None, overrides={"agents": {"n_initial": 256, "max_agents": 256}})
    env = GenesisEnv(cfg, seed=seed)
    pol = LearnedPolicy(checkpoint(arm), env, seed=seed)
    obs = env.observe()
    pos, deaths = [], 0
    for k in range(int(years * env.year_len)):
        if k % SAMPLE == 0:
            al = env.alive
            pos.append(np.c_[env.I[al, I_X], env.I[al, I_Y]].copy())
        o = obs
        if blind:
            o = dict(obs)
            o["grid"] = obs["grid"].copy()
            o["grid"][..., C_AGENT] = 0
            o["neighbors"] = np.zeros_like(obs["neighbors"])
        a = pol.act(o, env.alive)
        obs, _, died, info = env.step(a)
        deaths += int(np.asarray(died).sum())
        if info["born"].any():
            pol.reset(info["born"])
    agent_years = 256 * years
    return pos, deaths / agent_years


def main():
    rng = np.random.default_rng(0)
    out = {}
    for arm in ("gru", "ff"):
        for blind in (False, True):
            key = f"{arm}_{'blind' if blind else 'intact'}"
            ratios, mort = [], []
            for seed in (0, 1):
                _, passable, dist = world_geometry(seed)
                pos, m = run(arm, seed, blind)
                v = np.array([ce_with_null(xy.astype(np.int64), passable, dist, rng) for xy in pos])
                ratios.append(float(np.mean(v[:, 0] / v[:, 1])))
                mort.append(m)
            out[key] = dict(ratio_per_world=ratios, ratio=float(np.mean(ratios)), mortality_per_world=mort,
                            mortality=float(np.mean(mort)))
            print(key, out[key], flush=True)
    sp = HERE / "stats_h6_h7.json"
    s = json.loads(sp.read_text())
    s["blind_ablation"] = out
    sp.write_text(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
