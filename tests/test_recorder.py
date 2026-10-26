"""Flight recorder: complete per-tick agent records, exact replay, and observation reconstruction."""
import json

import numpy as np

from agents.baselines import make_baseline
from analysis.trajectories import load, observation_of, replay
from sim.recorder import TrajectoryRecorder
from tests.helpers import SMALL, make_env


def _run(tmp_path, ticks=260, obs=False, doomed=0):
    env = make_env(seed=3, agents=dict(SMALL["agents"], respawn=True))
    al = np.flatnonzero(env.alive)[:doomed]      # (not replayable: only for the non-replay test)
    env.A[al, 1] = 0.0                            # hydration
    env.A[al, 2] = 0.002                          # health -> dies of thirst within a tick or two
    pol = make_baseline("heuristic", env, 3)
    rec = TrajectoryRecorder(env, tmp_path, chunk_ticks=100, obs=obs)
    o = env.observe()
    seen = {}
    for k in range(ticks):
        if k == 40:
            env.trigger_fire(20, 20)
        if k == 90:
            env.trigger_flood(30, 30)            # duration drawn from env.rng -> must replay exactly
        if k == 150:
            env.trigger_drought(2)
        rec.pre(o)
        if k == 120:
            seen = {u: o["grid"][s].copy() for s, u in zip(np.flatnonzero(env.alive), env.I[env.alive, 5])}
            seen_tick = env.tick
        a = pol.act(o, env.alive)
        o, r, d, info = env.step(a)
        if info["born"].any():
            pol.reset(info["born"])
        rec.post(a, r, info, policy_state=pol.mode)
    idx = rec.close()
    return env, idx, seen, seen_tick


def test_recorder_captures_every_agent_every_tick(tmp_path):
    env, idx, _, _ = _run(tmp_path, doomed=2)
    a = load(tmp_path, "agents")
    w = load(tmp_path, "world")
    assert idx["ticks"] == 260 and len(w) == 260 and len(idx["chunks"]) == 3
    assert len(a) == idx["agent_rows"] == int(w["population"].sum())
    for t in np.unique(a["tick"])[:50]:                 # one row per live agent per tick, unique ids
        rows = a[a["tick"] == t]
        assert len(np.unique(rows["uid"])) == len(rows)
    births = load(tmp_path, "births")
    assert set(np.unique(a["uid"])) <= set(births["uid"].tolist())
    assert load(tmp_path, "births_genes").shape == (len(births), env.G.shape[1])
    died = a[a["died"] == 1]
    assert len(died) >= 2 and (died["death_cause"] >= 0).all()
    assert (died["death_cause"] == 1).sum() >= 2             # thirst
    ev = load(tmp_path, "events")
    assert (ev[:, 1] == 1).sum() == len(died)           # EV_DEATH rows == died rows
    assert set(np.unique(a["policy_state"])) <= {0, 1, 2}
    assert [m["kind"] for m in load(tmp_path, "manual")] == ["fire", "flood", "drought"]
    assert load(tmp_path, "fauna") is not None


def test_replay_is_bit_exact_and_rebuilds_observations(tmp_path):
    env, idx, seen, seen_tick = _run(tmp_path, obs=True)
    n = sum(1 for _ in replay(tmp_path))                 # raises on any mismatch, checks final state hash
    assert n == idx["ticks"]
    uid = next(iter(seen))
    o = observation_of(tmp_path, int(uid), seen_tick)
    assert np.array_equal(o["grid"], seen[uid])
    # stored observations line up with agent rows
    a = load(tmp_path, "agents")
    g = load(tmp_path, "obs_grid")
    assert len(g) == len(a)
    i = np.flatnonzero((a["tick"] == seen_tick) & (a["uid"] == uid))[0]
    assert np.array_equal(g[i], seen[uid])
    assert json.loads((tmp_path / "trajectory" / "run.json").read_text())["obs_recorded"] is True
