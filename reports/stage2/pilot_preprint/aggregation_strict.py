"""H7a robustness: Clark-Evans R against two nulls -- (1) matched on distance to fresh water (as preregistered),
(2) matched on distance-to-water bin x terrain class x fertility tercile (stricter habitat null, exploratory).
Positions are cached in data/positions_w<seed>.npz. Updates stats_h6_h7.json["aggregation_strict"].

    .venv/bin/python reports/stage2/pilot_preprint/aggregation_strict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analysis_h6_h7 import (BINS, POLICIES, clark_evans, positions_baseline, positions_learner,  # noqa: E402
                            world_geometry)
from analysis.survival import bootstrap_ci  # noqa: E402


def null_r(xy, keys_agent, keys_cell, px, py, area, rng, reps=20):
    pools = {k: np.flatnonzero(keys_cell == k) for k in np.unique(keys_agent)}
    out = []
    for _ in range(reps):
        pick = np.empty(len(xy), np.int64)
        for k in np.unique(keys_agent):
            m = keys_agent == k
            pick[m] = rng.choice(pools[k], m.sum(), replace=pools[k].size < m.sum())
        jit = rng.uniform(-0.01, 0.01, (len(xy), 2))
        out.append(clark_evans(np.c_[px[pick], py[pick]] + jit, area))
    return float(np.mean(out))


def main():
    rng = np.random.default_rng(0)
    res = {p: dict(R_obs=[], R_water=[], R_habitat=[]) for p in POLICIES}
    for seed in range(5):
        env, passable, dist = world_geometry(seed)
        cache = HERE / "data" / f"positions_w{seed}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            pos = {p: list(z[p]) for p in POLICIES}
        else:
            pos = {p: positions_learner(p, seed) if p in ("gru", "ff") else positions_baseline(p, seed) for p in POLICIES}
            np.savez_compressed(cache, **{p: np.array(v, dtype=object) for p, v in pos.items()})
        py, px = np.nonzero(passable)
        area = passable.sum()
        fert = env.fertility
        q = np.quantile(fert[passable], [1 / 3, 2 / 3])
        wbin = np.digitize(dist, BINS)
        hab = wbin * 1000 + env.terrain.astype(np.int64) * 10 + np.digitize(fert, q)
        for p in POLICIES:
            ro, rw, rh = [], [], []
            for xy in pos[p]:
                xy = xy.astype(np.int64)
                ro.append(clark_evans(xy.astype(float), area))
                rw.append(null_r(xy, wbin[xy[:, 1], xy[:, 0]], wbin[py, px], px, py, area, rng))
                rh.append(null_r(xy, hab[xy[:, 1], xy[:, 0]], hab[py, px], px, py, area, rng))
            res[p]["R_obs"].append(float(np.mean(ro)))
            res[p]["R_water"].append(float(np.mean(np.array(ro) / np.array(rw))))
            res[p]["R_habitat"].append(float(np.mean(np.array(ro) / np.array(rh))))
        print(seed, {p: (round(res[p]["R_water"][-1], 3), round(res[p]["R_habitat"][-1], 3)) for p in POLICIES}, flush=True)
    for p in POLICIES:
        res[p]["ratio_water_ci"] = bootstrap_ci(res[p]["R_water"])
        res[p]["ratio_habitat_ci"] = bootstrap_ci(res[p]["R_habitat"])
    sp = HERE / "stats_h6_h7.json"
    s = json.loads(sp.read_text())
    s["aggregation_strict"] = res
    sp.write_text(json.dumps(s, indent=1, default=float))
    print({p: (res[p]["ratio_water_ci"], res[p]["ratio_habitat_ci"]) for p in POLICIES})


if __name__ == "__main__":
    main()
