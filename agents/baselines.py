"""Non-learning baselines (HARD RULE 5). All act on the same observations learners receive.

- RandomPolicy:     uniform over the action space.
- GreedyForager:    attends to its lowest of {energy, hydration}; eats/drinks if possible,
                    otherwise walks to the nearest visible resource for that need, else explores.
- HeuristicAgent:   hand-written survival rules: flee predators/fire, drink/eat with hysteresis,
                    hunt adjacent small prey, seek shelter when cold, rest at night when sated.
None of them has any social rule.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from sim.constants import DRINK, EAT, HUNT, MOVE_E, MOVE_N, MOVE_S, MOVE_W, NOOP, PICKUP, REST

# observation channel indices (see sim.constants.OBS_CHANNELS)
C_VIS, C_BLOCK, C_WATER, C_FOREST, C_ROCK, C_SHELTER, C_PLANT, C_FRUIT, C_FOOD, C_SMALL, C_LARGE, \
    C_PRED, C_AGENT, C_FIRE, C_SIGNAL, C_SNOW = range(16)
MOVES = np.array([MOVE_N, MOVE_E, MOVE_S, MOVE_W], dtype=np.int64)
MDX = np.array([0, 1, 0, -1], dtype=np.int64)
MDY = np.array([-1, 0, 1, 0], dtype=np.int64)


class RandomPolicy:
    name = "random"

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def reset(self, mask=None):
        pass

    def act(self, obs: dict, alive: np.ndarray) -> np.ndarray:
        return self.rng.integers(0, self.n_actions, alive.shape[0])


@njit(cache=True, inline="always")
def _free(grid, gy, gx):
    S = grid.shape[1]
    if gy < 0 or gx < 0 or gy >= S or gx >= S:
        return False
    return (grid[C_BLOCK, gy, gx] == 0 and grid[C_AGENT, gy, gx] == 0 and grid[C_SMALL, gy, gx] == 0
            and grid[C_LARGE, gy, gx] == 0 and grid[C_PRED, gy, gx] == 0)


@njit(cache=True)
def _step_to(grid, R, ty, tx, u, away):
    """move action toward (or away from) patch offset (ty, tx); -1 if no free step."""
    best, best_s = -1, 0.0
    for d in range(4):
        gy, gx = R + MDY[d], R + MDX[d]
        if not _free(grid, gy, gx):
            continue
        dist = abs(MDY[d] - ty) + abs(MDX[d] - tx)
        s = (dist if away else -dist) + 0.01 * ((d + int(u * 4)) % 4)
        if best < 0 or s > best_s:
            best, best_s = d, s
    return MOVES[best] if best >= 0 else -1


@njit(cache=True)
def _nearest(grid, R, ch, thr):
    """nearest visible cell with grid[ch] >= thr (Manhattan); returns (found, dy, dx)."""
    S = 2 * R + 1
    best, by, bx = 1 << 30, 0, 0
    for gy in range(S):
        for gx in range(S):
            if grid[C_VIS, gy, gx] == 0 or grid[ch, gy, gx] < thr:
                continue
            d = abs(gy - R) + abs(gx - R)
            if d < best:
                best, by, bx = d, gy - R, gx - R
    return best < (1 << 30), by, bx


@njit(cache=True)
def _adjacent(grid, R, ch, thr):
    for d in range(5):
        dy = 0 if d == 0 else MDY[d - 1]
        dx = 0 if d == 0 else MDX[d - 1]
        if grid[ch, R + dy, R + dx] >= thr:
            return True
    return False


@njit(cache=True)
def _explore(grid, R, heading, u):
    """persistent random walk: keep heading while free, else turn."""
    h = heading
    if u < 0.08:
        h = int(u * 50) % 4
    for k in range(4):
        d = (h + k) % 4
        if _free(grid, R + MDY[d], R + MDX[d]):
            return MOVES[d], d
    return NOOP, h


@njit(cache=True)
def _food_here(grid, R):
    return (_adjacent(grid, R, C_FOOD, 10) or _adjacent(grid, R, C_FRUIT, 80)
            or grid[C_PLANT, R, R] >= 40)


@njit(cache=True)
def _seek_food(grid, R, u):
    for ch, thr in ((C_FOOD, 10), (C_FRUIT, 80), (C_PLANT, 90)):
        ok, dy, dx = _nearest(grid, R, ch, thr)
        if ok:
            if ch == C_FRUIT or ch == C_FOOD:
                if abs(dy) + abs(dx) <= 1:
                    return EAT
            return _step_to(grid, R, dy, dx, u, False)
    return -1


@njit(cache=True, parallel=True)
def greedy_act(grid, selff, alive, U, heading, out):
    R = grid.shape[2] // 2
    for i in prange(grid.shape[0]):
        out[i] = NOOP
        if not alive[i]:
            continue
        g = grid[i]
        energy, hyd = selff[i, 0], selff[i, 1]
        a = -1
        if hyd < energy:
            if _adjacent(g, R, C_WATER, 200):
                a = DRINK
            else:
                ok, dy, dx = _nearest(g, R, C_WATER, 200)
                if ok:
                    a = _step_to(g, R, dy, dx, U[i, 0], False)
        else:
            if _food_here(g, R):
                a = EAT
            else:
                a = _seek_food(g, R, U[i, 0])
        if a < 0:
            a, heading[i] = _explore(g, R, heading[i], U[i, 1])
        out[i] = a


@njit(cache=True, parallel=True)
def heuristic_act(grid, selff, alive, U, heading, mode, out):
    """mode: per-agent hysteresis state (0 none, 1 drinking, 2 eating)."""
    R = grid.shape[2] // 2
    for i in prange(grid.shape[0]):
        out[i] = NOOP
        if not alive[i]:
            continue
        g = grid[i]
        energy, hyd, health, warmth = selff[i, 0], selff[i, 1], selff[i, 2], selff[i, 3]
        light = selff[i, 9]
        sheltered = selff[i, 11] > 0
        a = -1
        # 1. danger: predators or fire within 3 cells -> move away
        ok, dy, dx = _nearest(g, R, C_PRED, 1)
        if ok and abs(dy) + abs(dx) <= 3:
            a = _step_to(g, R, dy, dx, U[i, 0], True)
        if a < 0:
            ok, dy, dx = _nearest(g, R, C_FIRE, 200)
            if ok and abs(dy) + abs(dx) <= 3:
                a = _step_to(g, R, dy, dx, U[i, 0], True)
        if a >= 0:
            out[i] = a
            continue
        # 2. hysteresis on needs
        if mode[i] == 1 and hyd > 0.95:
            mode[i] = 0
        if mode[i] == 2 and energy > 0.9:
            mode[i] = 0
        if mode[i] == 0:
            if hyd < 0.55 and hyd <= energy:
                mode[i] = 1
            elif energy < 0.6:
                mode[i] = 2
        # 3. cold -> shelter
        if warmth < 0.45 and not sheltered:
            ok, dy, dx = _nearest(g, R, C_SHELTER, 100)
            if not ok:
                ok, dy, dx = _nearest(g, R, C_FOREST, 200)
            if ok and abs(dy) + abs(dx) > 0:
                a = _step_to(g, R, dy, dx, U[i, 0], False)
            elif ok or mode[i] == 0:
                a = REST
        if a < 0 and mode[i] == 1:
            if _adjacent(g, R, C_WATER, 200):
                a = DRINK
            else:
                ok, dy, dx = _nearest(g, R, C_WATER, 200)
                if ok:
                    a = _step_to(g, R, dy, dx, U[i, 0], False)
        if a < 0 and mode[i] == 2:
            if _food_here(g, R):
                a = EAT
            elif _adjacent(g, R, C_SMALL, 1):
                a = HUNT
            else:
                a = _seek_food(g, R, U[i, 0])
                if a < 0:
                    ok, dy, dx = _nearest(g, R, C_SMALL, 1)
                    if ok:
                        a = _step_to(g, R, dy, dx, U[i, 0], False)
        if a < 0 and mode[i] == 0:
            # sated: stockpile ground food, rest at night, otherwise drift toward water + food
            if g[C_FOOD, R, R] >= 10 and selff[i, 5] < 0.5:
                a = PICKUP
            elif light < 0.5 or health < 0.7:
                a = REST
        if a < 0:
            a, heading[i] = _explore(g, R, heading[i], U[i, 1])
        out[i] = a


class _StatefulBaseline:
    def __init__(self, n_slots: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.heading = self.rng.integers(0, 4, n_slots).astype(np.int64)
        self.mode = np.zeros(n_slots, np.int64)
        self.out = np.zeros(n_slots, np.int64)

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones_like(self.mode, bool)
        self.mode[mask] = 0
        self.heading[mask] = self.rng.integers(0, 4, int(mask.sum()))


class GreedyForager(_StatefulBaseline):
    name = "greedy"

    def act(self, obs, alive):
        U = self.rng.random((alive.shape[0], 2))
        greedy_act(obs["grid"], obs["self"], alive, U, self.heading, self.out)
        return self.out.copy()


class HeuristicAgent(_StatefulBaseline):
    name = "heuristic"

    def act(self, obs, alive):
        U = self.rng.random((alive.shape[0], 2))
        heuristic_act(obs["grid"], obs["self"], alive, U, self.heading, self.mode, self.out)
        return self.out.copy()


def make_baseline(name: str, env, seed: int = 0):
    if name == "random":
        return RandomPolicy(env.n_actions, seed)
    if name == "greedy":
        return GreedyForager(env.N, seed)
    if name == "heuristic":
        return HeuristicAgent(env.N, seed)
    raise ValueError(name)
