"""Read flight-recorder trajectories (sim/recorder.py) and replay them exactly.

    python -m analysis.trajectories RUN_DIR                    # summary of what was recorded
    python -m analysis.trajectories RUN_DIR --uid 17 --csv a.csv   # one agent's complete tick-by-tick life
    python -m analysis.trajectories RUN_DIR --verify            # re-simulate and check every tick bit-exactly

`load(run_dir, "agents")` returns one structured array for the whole run (columns: sim.recorder.AGENT_FIELDS).
`replay(run_dir)` re-runs the simulation from (seed, config, recorded actions, manual interventions), checks the
state of every agent at every tick against the record, and yields (tick, env, obs) so anything not stored -- for
example each agent's exact observation -- can be reconstructed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _tdir(run_dir) -> Path:
    p = Path(run_dir)
    return p if p.name == "trajectory" else p / "trajectory"


def info(run_dir) -> dict:
    d = _tdir(run_dir)
    run = json.loads((d / "run.json").read_text())
    idx = json.loads((d / "index.json").read_text()) if (d / "index.json").exists() else None
    return dict(run=run, index=idx, chunks=sorted(d.glob("chunk_*.npz")))


def iter_chunks(run_dir, tables=("agents",)):
    for f in sorted(_tdir(run_dir).glob("chunk_*.npz")):
        with np.load(f) as z:
            out = {}
            for t in tables:
                if t == "manual":
                    out[t] = json.loads(bytes(z["manual"]).decode()) if "manual" in z.files else []
                elif t in z.files:
                    out[t] = z[t]
            yield out


def load(run_dir, table="agents"):
    """Concatenate one table (agents, world, fauna, events, births, births_genes, obs_grid, ...) over all chunks."""
    if table == "manual":
        return [m for c in iter_chunks(run_dir, ("manual",)) for m in c["manual"]]
    parts = [c[table] for c in iter_chunks(run_dir, (table,)) if table in c]
    return np.concatenate(parts) if parts else None


def agent_history(run_dir, uid: int):
    a = load(run_dir, "agents")
    return a[a["uid"] == uid]


# ---------------------------------------------------------------------------------------------
CHECK = ("x", "y", "age", "energy", "hydration", "health", "warmth", "carry_type", "carry_amt")


def replay(run_dir, until_tick: int | None = None, verify: bool = True, with_obs: bool = False):
    """Re-simulate a recorded run. Yields (tick, env, obs) BEFORE each recorded tick is stepped, where obs is the
    exact observation every agent acted on (only computed if with_obs). Raises AssertionError on any mismatch."""
    from sim import GenesisEnv
    from sim.config import Config
    from sim.layout import (A_CARRY_AMT, A_ENERGY, A_HEALTH, A_HYDRATION, A_WARMTH, I_AGE, I_CARRY_TYPE, I_UID,
                            I_X, I_Y)

    d = _tdir(run_dir)
    run = json.loads((d / "run.json").read_text())
    env = GenesisEnv(Config(run["config"]), seed=run["seed"])
    if verify and env.state_hash() != run["state_hash_start"]:
        raise AssertionError("initial state differs from the recording (different code or config?)")
    manual = {}
    for m in load(run_dir, "manual"):
        manual.setdefault(int(m["tick"]), []).append(m)
    obs = env.observe() if with_obs else None
    N = env.N
    for chunk in iter_chunks(run_dir, ("agents", "world")):
        a = chunk["agents"]
        ticks = chunk["world"]["tick"]
        starts = np.searchsorted(a["tick"], ticks, side="left")
        ends = np.searchsorted(a["tick"], ticks, side="right")
        for t, s0, s1 in zip(ticks, starts, ends):
            if until_tick is not None and t >= until_tick:
                return
            for m in manual.pop(int(t), []):
                _apply_manual(env, m)
                if with_obs:
                    obs = env.observe()
            if env.tick != t:
                raise AssertionError(f"tick mismatch: env {env.tick}, record {t}")
            rows = a[s0:s1]
            if verify:
                al = np.flatnonzero(env.alive)
                if not np.array_equal(al, rows["slot"].astype(np.int64)):
                    raise AssertionError(f"tick {t}: different set of live agents")
                if not np.array_equal(env.I[al, I_UID], rows["uid"].astype(np.int64)):
                    raise AssertionError(f"tick {t}: agent identities differ")
                cur = dict(x=env.I[al, I_X], y=env.I[al, I_Y], age=env.I[al, I_AGE], energy=env.A[al, A_ENERGY],
                           hydration=env.A[al, A_HYDRATION], health=env.A[al, A_HEALTH], warmth=env.A[al, A_WARMTH],
                           carry_type=env.I[al, I_CARRY_TYPE], carry_amt=env.A[al, A_CARRY_AMT])
                for k in CHECK:
                    if not np.array_equal(np.asarray(cur[k]).astype(rows[k].dtype), rows[k]):
                        raise AssertionError(f"tick {t}: field {k} differs")
            yield int(t), env, obs
            actions = np.zeros(N, np.int64)
            actions[rows["slot"].astype(np.int64)] = rows["action"]
            o, _, _, _ = env.step(actions, compute_obs=with_obs)
            if with_obs:
                obs = o
    idx = json.loads((d / "index.json").read_text()) if (d / "index.json").exists() else None
    if verify and idx and until_tick is None and env.state_hash() != idx["state_hash_end"]:
        raise AssertionError("final state hash differs from the recording")


def _apply_manual(env, m):
    k = m["kind"]
    if k == "fire":
        env.trigger_fire(m["x"], m["y"])
    elif k == "flood":
        env.trigger_flood(m["x"], m["y"], days=m["days"] if m.get("days_given", True) else None,
                          radius=m.get("radius", 30))
    elif k == "famine":
        env.trigger_famine(m["days"])
    elif k == "drought":
        env.trigger_drought(m["days"])
    elif k == "cold_snap":
        env.trigger_cold_snap(m["days"])
    else:
        raise ValueError(f"unknown manual intervention {k}")


def observation_of(run_dir, uid: int, tick: int):
    """Exact observation (grid, self, neighbors, heard) agent `uid` acted on at absolute `tick` (by replay)."""
    from sim.layout import I_UID
    for t, env, obs in replay(run_dir, until_tick=tick + 1, verify=True, with_obs=True):
        if t == tick:
            slot = np.flatnonzero((env.I[:, I_UID] == uid) & env.alive)
            if not slot.size:
                raise KeyError(f"agent {uid} not alive at tick {tick}")
            return {k: v[slot[0]].copy() for k, v in obs.items()}
    raise KeyError(f"tick {tick} not in recording")


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--uid", type=int)
    ap.add_argument("--csv")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    i = info(a.run_dir)
    run, idx = i["run"], i["index"]
    if a.uid is not None:
        h = agent_history(a.run_dir, a.uid)
        names = run["action_names"]
        print(f"agent {a.uid}: {len(h)} recorded ticks")
        if a.csv:
            import csv
            with open(a.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(list(h.dtype.names) + ["action_name"])
                for r in h:
                    w.writerow([r[n].item() for n in h.dtype.names] + [names[r["action"]]])
            print("wrote", a.csv)
        return
    print(f"seed {run['seed']}, start tick {run['start_tick']}, obs recorded: {run['obs_recorded']}")
    if idx:
        print(f"{idx['ticks']} ticks, {idx['agent_rows']:,} agent-tick rows, {len(idx['chunks'])} chunks, "
              f"{idx['bytes'] / 1e6:.1f} MB")
    else:
        print(f"{len(i['chunks'])} chunks on disk (run still open or not closed cleanly)")
    if a.verify:
        n = sum(1 for _ in replay(a.run_dir))
        print(f"replay verified: {n} ticks bit-exact")


if __name__ == "__main__":
    main()
