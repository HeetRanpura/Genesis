"""Sanity check that the ecology is learnable: hand-written survival rules beat random."""
import numpy as np

from agents.baselines import make_baseline
from sim import GenesisEnv, load_config
from sim.constants import EV_DEATH


def deaths(policy, seed, ticks=1500):
    env = GenesisEnv(load_config(overrides={"agents": {"n_initial": 128, "max_agents": 128}}), seed=seed)
    pol = make_baseline(policy, env, seed)
    obs, n = env.observe(), 0
    for _ in range(ticks):
        obs, r, d, info = env.step(pol.act(obs, env.alive))
        n += int(d.sum())
        if info["born"].any():
            pol.reset(info["born"])
    return n


def test_heuristic_beats_random():
    h = [deaths("heuristic", s) for s in (0, 1)]
    r = [deaths("random", s) for s in (0, 1)]
    assert np.mean(h) < 0.6 * np.mean(r)


def test_baselines_emit_valid_actions():
    env = GenesisEnv(load_config(overrides={"agents": {"n_initial": 32, "max_agents": 32}}), seed=0)
    obs = env.observe()
    for name in ("random", "greedy", "heuristic"):
        a = make_baseline(name, env, 0).act(obs, env.alive)
        assert a.shape == (env.N,) and a.min() >= 0 and a.max() < env.n_actions
