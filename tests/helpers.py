"""Scenario helpers: small worlds with hand-placed agents/animals for rule tests."""
import numpy as np

from sim import GenesisEnv, load_config
from sim import kernels as K
from sim.constants import GRASS, NOOP
from sim.layout import A_ENERGY, A_HEALTH, A_HYDRATION, A_WARMTH, F_ALIVE, I_ALIVE, I_LIFESPAN

SMALL = {
    "world": {"width": 64, "height": 64},
    "agents": {"n_initial": 8, "max_agents": 16, "respawn": False},
    "fauna": {"small_prey": {"initial": 30, "min": 0, "max": 60},
              "large_prey": {"initial": 6, "min": 0, "max": 12},
              "predators": {"initial": 0, "min": 0, "max": 0}},
}


def deep(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = deep(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def make_env(seed=0, **overrides):
    return GenesisEnv(load_config(overrides=deep(SMALL, overrides)), seed=seed)


def clear_world(env, terrain=GRASS):
    """Empty, flat, uniform test arena: no agents, animals, water, trees, fire."""
    env.terrain[:] = terrain
    env.elev[:] = env.cfg.world.sea_level + 0.1
    env.fresh[:] = False
    env.fresh_now[:] = False
    env.world["stream"][:] = False
    env.plant[:] = 0.0
    env.fertility[:] = 0.5
    env.fire[:] = 0
    env.flood[:] = 0
    env.snow[:] = 0
    env.food_item[:] = 0
    env.tree_at[:] = -1
    env.I[:, I_ALIVE] = 0
    env.occ[:] = -1
    env.FI[:, F_ALIVE] = 0
    env.fidx[:] = -1
    env.fcount[:3] = 0
    env.weather = 0
    env.famine_left = env.drought_left = env.cold_snap_left = 0
    env.cfg.disturbances.fire_ignition_per_day = 0.0
    env.cfg.disturbances.famine_prob_per_year = 0.0


def put_agent(env, slot, x, y, energy=0.8, hydration=0.8, warmth=0.8, uid=None):
    genes = np.zeros(env.G.shape[1], np.float32)
    uid = uid if uid is not None else 1000 + slot
    K.init_agent(slot, x, y, uid, 0, 0, genes, energy, hydration, warmth, 1e6, 1e6, env.seed, env.tick,
                 env.A, env.I, env.S, env.G, env.HIST, env.occ)
    env.I[slot, I_LIFESPAN] = 10**9
    env._prev_drive = env._drive()
    return slot


def noop(env):
    return np.full(env.N, NOOP, np.int64)


def body(env, slot):
    return {k: float(env.A[slot, c]) for k, c in
            [("energy", A_ENERGY), ("hydration", A_HYDRATION), ("health", A_HEALTH), ("warmth", A_WARMTH)]}
