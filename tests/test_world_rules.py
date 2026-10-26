import numpy as np
import pytest

from sim import kernels as K
from sim.constants import (CAVE, DEEP_WATER, DRINK, EAT, EV_BIRTH, EV_DEATH, EV_KILL_LARGE, FOREST,
                           GRASS, HUNT, LARGE_PREY, MOUNTAIN, MOVE_E, MOVE_N, REST, SHALLOW,
                           DEATH_THIRST)
from sim.layout import A_ENERGY, A_HYDRATION, I_ALIVE, I_X, I_Y
from tests.helpers import body, clear_world, make_env, noop, put_agent


@pytest.fixture
def env():
    e = make_env()
    clear_world(e)
    return e


def test_move_into_free_cell(env):
    put_agent(env, 0, 10, 10)
    a = noop(env)
    a[0] = MOVE_E
    env.step(a)
    assert (env.I[0, I_X], env.I[0, I_Y]) == (11, 10)
    assert env.occ[10, 11] == 0 and env.occ[10, 10] == -1


@pytest.mark.parametrize("blocker", [MOUNTAIN, DEEP_WATER])
def test_blocking_terrain(env, blocker):
    put_agent(env, 0, 10, 10)
    env.terrain[9, 10] = blocker
    a = noop(env)
    a[0] = MOVE_N
    env.step(a)
    assert (env.I[0, I_X], env.I[0, I_Y]) == (10, 10)


def test_bodies_block_each_other(env):
    put_agent(env, 0, 10, 10)
    put_agent(env, 1, 11, 10)
    a = noop(env)
    a[0] = MOVE_E
    env.step(a)
    assert (env.I[0, I_X], env.I[0, I_Y]) == (10, 10)


def test_moving_costs_more_than_resting(env):
    put_agent(env, 0, 10, 10)
    put_agent(env, 1, 30, 30)
    for k in range(20):
        a = noop(env)
        a[0] = MOVE_E if k % 2 == 0 else 3 + 1  # E then W (MOVE_W == 4)
        a[1] = REST
        env.step(a)
    assert env.A[0, A_ENERGY] < env.A[1, A_ENERGY]


def test_drink_only_near_fresh_water(env):
    put_agent(env, 0, 10, 10, hydration=0.3)
    put_agent(env, 1, 30, 30, hydration=0.3)
    env.terrain[10, 11] = SHALLOW
    env.fresh[10, 11] = env.fresh_now[10, 11] = True
    a = noop(env)
    a[0] = a[1] = DRINK
    env.step(a)
    assert env.A[0, A_HYDRATION] > 0.45
    assert env.A[1, A_HYDRATION] < 0.3


def test_salt_water_is_not_drinkable(env):
    put_agent(env, 0, 10, 10, hydration=0.3)
    env.terrain[10, 11] = SHALLOW      # shore of the ocean: fresh=False
    a = noop(env)
    a[0] = DRINK
    env.step(a)
    assert env.A[0, A_HYDRATION] < 0.3


def test_eat_plant_raises_energy_and_depletes_plant(env):
    put_agent(env, 0, 10, 10, energy=0.3)
    env.plant[10, 10] = 0.5
    a = noop(env)
    a[0] = EAT
    env.step(a)
    assert env.A[0, A_ENERGY] > 0.3
    assert env.plant[10, 10] < 0.5


def test_thirst_kills_and_is_logged(env):
    put_agent(env, 0, 10, 10, hydration=0.0, energy=1.0)
    env.A[0, 2] = 0.002  # health: one tick of thirst damage kills
    env.step(noop(env))
    assert env.I[0, I_ALIVE] == 0
    deaths = env.last_events[env.last_events[:, 1] == EV_DEATH]
    assert len(deaths) == 1 and int(deaths[0, 6]) == DEATH_THIRST


def _put_large_prey(env, x, y):
    j = K._spawn_fauna(LARGE_PREY, x, y, 1.0, env.FF, env.FI, env.fidx, env.fcount)
    assert j >= 0
    return j


@pytest.mark.parametrize("coop", [True, False])
def test_large_prey_needs_two_hunters_when_cooperative(coop):
    env = make_env(capabilities={"cooperative_large_prey": coop})
    clear_world(env)
    put_agent(env, 0, 10, 10)
    _put_large_prey(env, 11, 10)
    kills = 0
    for _ in range(8):           # one hunter, many hits
        a = noop(env)
        a[0] = HUNT
        env.step(a)
        kills += int((env.last_events[:, 1] == EV_KILL_LARGE).sum())
    if coop:
        assert kills == 0
        put_agent(env, 1, 12, 11)
        # bring the prey back next to both hunters if it fled
        for _ in range(8):
            a = noop(env)
            a[0] = a[1] = HUNT
            env.step(a)
            kills += int((env.last_events[:, 1] == EV_KILL_LARGE).sum())
            if kills:
                break
        # the prey may flee out of reach; the rule itself is checked directly below
    else:
        assert kills == 1


def test_cooperative_kill_rule_direct():
    env = make_env()
    clear_world(env)
    put_agent(env, 0, 10, 10)
    put_agent(env, 1, 12, 10)
    env.terrain[9:12, 11] = GRASS
    # wall the prey in so it cannot flee
    for (y, x) in [(9, 11), (11, 11)]:
        env.terrain[y, x] = MOUNTAIN
    _put_large_prey(env, 11, 10)
    kills = 0
    for _ in range(4):
        a = noop(env)
        a[0] = a[1] = HUNT
        env.step(a)
        kills += int((env.last_events[:, 1] == EV_KILL_LARGE).sum())
    assert kills == 1
    assert env.food_item[10, 11] > 0


def test_seasonal_plant_growth():
    def grow(season_tick):
        env = make_env()
        clear_world(env)
        env.plant[:] = 0.1
        env.tick = season_tick
        for _ in range(40):
            env.step(noop(env))
        return float(env.plant.mean())
    sl = make_env().season_len
    spring, winter = grow(int(sl * 0.5)), grow(int(sl * 3.5))
    assert spring > 0.112 and winter < 0.103


def test_occlusion_by_mountain(env):
    put_agent(env, 0, 20, 20)
    env.plant[20, 23] = 1.0
    obs = env.observe()
    R = env.R
    assert obs["grid"][0, 0, R, R + 3] == 255          # visible
    assert obs["grid"][0, 6, R, R + 3] > 200          # plant seen
    env.terrain[20, 22] = MOUNTAIN
    obs = env.observe()
    assert obs["grid"][0, 0, R, R + 3] == 0           # hidden behind mountain
    assert obs["grid"][0, 6, R, R + 3] == 0


def test_dense_forest_occludes(env):
    put_agent(env, 0, 20, 20)
    env.terrain[20, 21:23] = FOREST
    obs = env.observe()
    R = env.R
    assert obs["grid"][0, 0, R, R + 2] == 255   # first forest cell after one forest cell: visible
    assert obs["grid"][0, 0, R, R + 4] == 0     # behind two forest cells: hidden


def test_fire_spreads_and_burns_out(env):
    env.plant[:] = 0.8
    env.trigger_fire(30, 30)
    burning = []
    for _ in range(400):
        env.step(noop(env))
        burning.append(int((env.fire > 0).sum()))
    assert max(burning) > 5            # it spread
    assert env.fire[30, 30] <= 0       # origin burnt out
    assert env.plant[30, 30] == 0


def test_cave_is_warmer(env):
    put_agent(env, 0, 10, 10, warmth=0.5)
    put_agent(env, 1, 30, 30, warmth=0.5)
    env.terrain[10, 10] = CAVE
    for _ in range(50):
        env.step(noop(env))
    assert env.A[0, 3] > env.A[1, 3]


def test_reproduction_inherits_genes():
    env = make_env(capabilities={"reproduction": True}, reproduction={"cooldown_years": 0.0})
    clear_world(env)
    put_agent(env, 0, 10, 10, energy=1.0, hydration=1.0)
    env.G[0, :] = 0.5
    env.I[0, 3] = env.maturity + 1   # age
    env.step(noop(env))
    births = env.last_events[env.last_events[:, 1] == EV_BIRTH]
    assert len(births) == 1
    child = int(np.flatnonzero((env.I[:, I_ALIVE] == 1) & (np.arange(env.N) != 0))[0])
    assert np.abs(env.G[child] - 0.5).max() < 0.4
    assert env.A[0, A_ENERGY] < 1.0
