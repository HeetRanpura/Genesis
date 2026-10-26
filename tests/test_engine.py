import numpy as np
import pytest

from sim import GenesisEnv, load_config
from sim.constants import N_BASE_ACTIONS
from sim.layout import A_ENERGY, A_HEALTH, A_HYDRATION, A_WARMTH
from sim.worldgen import generate_world
from tests.helpers import SMALL, clear_world, deep, make_env, noop, put_agent


# ---------------------------------------------------------------- world generation
def test_worldgen_deterministic_and_seed_dependent():
    cfg = load_config()
    a, b, c = generate_world(cfg, 7), generate_world(cfg, 7), generate_world(cfg, 8)
    for k in ("terrain", "fertility", "tree_x", "fresh"):
        assert np.array_equal(a[k], b[k])
    assert not np.array_equal(a["terrain"], c["terrain"])


def test_worldgen_composition():
    cfg = load_config()
    W = generate_world(cfg, 0)
    T = W["terrain"]
    assert T.shape == (256, 256)
    frac = np.bincount(T.ravel(), minlength=10) / T.size
    assert 0.15 < frac[0] < 0.5          # ocean / deep water
    assert 0.02 < frac[8] < 0.12         # mountains
    assert frac[5] > 0.05                # forest
    assert (T == 9).sum() >= 1           # caves exist
    assert W["fresh"].mean() > 0.02      # drinkable water
    assert W["tree_x"].size > 50


def test_uniform_layout_flattens_fertility():
    cfg = load_config(overrides={"world": {"resource_layout": "uniform"}})
    W = generate_world(cfg, 0)
    veg = W["fertility"][W["fertility"] > 0]
    assert np.allclose(veg, veg[0])


# ---------------------------------------------------------------- determinism
def _run(seed, steps=300):
    env = make_env(seed=seed, agents={"n_initial": 12, "max_agents": 16})
    rng = np.random.default_rng(123)
    for _ in range(steps):
        env.step(rng.integers(0, env.n_actions, env.N), compute_obs=False)
    return env.state_hash()


def test_full_run_bit_reproducible():
    assert _run(3) == _run(3)


def test_different_seeds_diverge():
    assert _run(3) != _run(4)


def test_default_size_run_reproducible():
    def run():
        env = GenesisEnv(load_config(overrides={"agents": {"n_initial": 64, "max_agents": 64}}), seed=11)
        rng = np.random.default_rng(5)
        for _ in range(150):
            env.step(rng.integers(0, env.n_actions, env.N))
        return env.state_hash(), env.observe()["grid"].sum()
    assert run() == run()


# ---------------------------------------------------------------- capability switches
def test_capabilities_off_by_default():
    cap = load_config().capabilities
    for k in ("memory", "world_model", "communication", "social_learning", "niche_construction",
              "reproduction", "reputation", "aggression", "huddle_warmth"):
        assert cap[k] is False, k


def test_action_space_grows_with_capabilities():
    assert make_env().n_actions == N_BASE_ACTIONS
    env = make_env(capabilities={"niche_construction": True, "aggression": True, "communication": True,
                                 "n_symbols": 4})
    assert env.n_actions == N_BASE_ACTIONS + 1 + 1 + 4
    assert env.action_names[env.build_idx] == "build"
    assert env.action_names[env.signal_idx] == "signal_0"


def test_signal_is_heard_only_with_comms():
    env = make_env(capabilities={"communication": True, "n_symbols": 4})
    clear_world(env)
    put_agent(env, 0, 10, 10)
    put_agent(env, 1, 13, 10)
    a = noop(env)
    a[0] = env.signal_idx + 2
    obs, *_ = env.step(a)
    assert obs["heard"][1, 2] > 0 and obs["heard"][1].sum() == obs["heard"][1, 2]
    env2 = make_env()
    obs2 = env2.observe()
    assert obs2["heard"].sum() == 0


def test_build_requires_niche_construction_and_material():
    from sim.constants import FOREST, PICKUP
    env = make_env(capabilities={"niche_construction": True})
    clear_world(env)
    env.terrain[10, 10] = FOREST
    env.wood[10, 10] = 1.0
    put_agent(env, 0, 10, 10)
    a = noop(env)
    a[0] = PICKUP
    env.step(a)
    a[0] = env.build_idx
    env.step(a)
    assert env.built[10, 10] == 1


# ---------------------------------------------------------------- reward purity (HARD RULE 1)
def test_reward_is_own_homeostatic_drive_only():
    env = make_env(agents={"n_initial": 12, "max_agents": 16})
    rng = np.random.default_rng(0)
    sp = np.asarray(env.cfg.reward.setpoint)
    for _ in range(200):
        before = env.A[:, [A_ENERGY, A_HYDRATION, A_HEALTH, A_WARMTH]].copy()
        alive_before = env.alive.copy()
        _, r, died, info = env.step(rng.integers(0, env.n_actions, env.N), compute_obs=False)
        after = env.A[:, [A_ENERGY, A_HYDRATION, A_HEALTH, A_WARMTH]]
        d0 = np.sqrt((np.maximum(sp - before, 0) ** 2).sum(1))
        d1 = np.sqrt((np.maximum(sp - after, 0) ** 2).sum(1))
        expect = d0 - d1 + np.where(died, env.cfg.reward.death_penalty, 0.0)
        ok = alive_before & ~info["born"]
        np.testing.assert_allclose(r[ok], expect[ok], atol=1e-5)
        assert np.all(r[~alive_before & ~died] == 0)


# ---------------------------------------------------------------- observations
def test_observation_shapes_and_dead_slots_zero():
    env = make_env()
    obs = env.observe()
    R = env.R
    assert obs["grid"].shape == (env.N, 2 * R + 1, 2 * R + 1, 16)
    assert obs["self"].shape == (env.N, 13)
    dead = ~env.alive
    assert obs["grid"][dead].sum() == 0


def test_night_shrinks_view():
    env = make_env()
    clear_world(env)
    put_agent(env, 0, 20, 20)
    env.tick = env.tpd // 4      # noon
    env._climate_tick(False)
    day_vis = int((env.observe()["grid"][0, :, :, 0] > 0).sum())
    env.tick = 3 * env.tpd // 4  # midnight
    env._climate_tick(False)
    night_vis = int((env.observe()["grid"][0, :, :, 0] > 0).sum())
    assert night_vis < day_vis


# ---------------------------------------------------------------- PettingZoo
def test_pettingzoo_parallel_api():
    from pettingzoo.test import parallel_api_test
    from sim.pz_env import GenesisParallelEnv
    env = GenesisParallelEnv(load_config(overrides=deep(SMALL, {})), seed=0)
    parallel_api_test(env, num_cycles=300)


# ---------------------------------------------------------------- disturbances
@pytest.mark.parametrize("kind", ["flood", "famine", "drought", "cold_snap"])
def test_manual_disturbances_logged(kind):
    env = GenesisEnv(load_config(overrides={"agents": {"n_initial": 8, "max_agents": 8}}), seed=0)
    if kind == "flood":
        ry, rx = np.nonzero(env.world["river"])
        env.trigger_flood(int(rx[0]), int(ry[0]), days=2)
        assert (env.flood > 0).sum() > 0
    elif kind == "famine":
        env.trigger_famine(5)
        assert env.growth_mult() < env.cfg.resources.season_growth[0]
    elif kind == "drought":
        env.trigger_drought(5)
        assert env.fresh_now.sum() < env.fresh.sum()
    else:
        t0 = env.global_temp()
        env.trigger_cold_snap(3)
        assert env.global_temp() < t0 - 5
    assert env.manual_log[-1]["kind"] == kind


# ---------------------------------------------------------------- observation image invariant
def test_incremental_obs_image_matches_rebuild():
    """img is patched in place at every mutation site; it must equal a from-scratch rebuild."""
    cfg = load_config(overrides={"agents": {"n_initial": 300, "max_agents": 300},
                                 "capabilities": {"niche_construction": True, "aggression": True,
                                                  "communication": True, "reproduction": True}})
    env = GenesisEnv(cfg, seed=2)
    rng = np.random.default_rng(0)
    ry, rx = np.nonzero(env.world["river"])
    forest = np.argwhere(env.terrain == 5)
    for k in range(1600):
        if k == 100:
            env.trigger_fire(int(forest[len(forest) // 2][1]), int(forest[len(forest) // 2][0]))
        if k == 300:
            env.trigger_flood(int(rx[10]), int(ry[10]), days=2)
        env.step(rng.integers(0, env.n_actions, env.N), compute_obs=(k % 97 == 0))
        if k in (150, 333, 801, 1599):
            inc = env.img.copy()
            env.refresh_img()
            assert np.array_equal(inc, env.img), f"img drift at tick {k}: channels {np.unique(np.argwhere(inc != env.img)[:, 2])}"
    assert env.ev_counts[13] >= 1 and env.ev_counts[14] >= 1   # manual fire + flood were logged


def test_sequential_and_parallel_twins_agree():
    """env.step picks prey_proposals / fire_step or their sequential twins by workload size; the
    choice must not change results."""
    from sim import kernels as K
    env = GenesisEnv(load_config(overrides={"agents": {"n_initial": 200, "max_agents": 200}}), seed=4)
    rng = np.random.default_rng(1)
    for _ in range(50):
        env.step(rng.integers(0, env.n_actions, env.N), compute_obs=False)
    K.build_threat_grid(env.I, env.FI, env.tgrid)
    args = (env.P, env.FF, env.FI, env.fidx, env.occ, env.terrain, env.plant, env.built, env.fire, env.tgrid)
    a, b = env.fprop.copy(), env.fprop.copy()
    K.prey_proposals(*args, a)
    K.prey_proposals_seq(*args, b)
    assert np.array_equal(a, b)

    forest = np.argwhere(env.terrain == 5)
    y, x = forest[len(forest) // 2]
    env.plant[:] = np.maximum(env.plant, 0.5 * env.fertility)
    env.trigger_fire(int(x), int(y))
    states = []
    for kernel in (K.fire_step, K.fire_step_seq):
        arrs = [env.fire.copy(), env.ignite.copy(), env.plant.copy(), env.wood.copy(), env.built.copy(),
                env.img.copy(), env.fire_box.copy(), env._row_min.copy(), env._row_max.copy()]
        fire, ign, plant, wood, built, img, box, rmin, rmax = arrs
        for t in range(40):
            env.P[1] = 1000 + t
            kernel(env.P, 0.3, 30, 1.2, fire, ign, env.terrain, plant, wood, built, env.flood, img, box, rmin, rmax)
        states.append([fire, plant, wood, img, box])
    for u, v in zip(*states):
        assert np.array_equal(u, v)
