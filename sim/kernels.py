"""Numba kernels for one simulation tick.

Determinism: all randomness comes from sim.rng.uniform(seed, tick, index, stream); sequential
kernels iterate in a hash-derived permutation; parallel kernels never draw shared state.
Single occupancy: a cell holds at most one body (agent or animal).
"""
import numpy as np
from numba import njit, prange

from .constants import (BARREN, CARRY_FOOD, CARRY_NONE, CARRY_STONE, CARRY_WOOD, CAVE,
                        DEATH_COLD, DEATH_FIRE, DEATH_OLD_AGE, DEATH_OTHER, DEATH_PREDATOR,
                        DEATH_STARVATION, DEATH_STRIKE, DEATH_THIRST, DEEP_WATER, DRINK, DROP, EAT,
                        EV_ATTACK_LARGE, EV_BIRTH, EV_BUILD, EV_DEATH, EV_DRINK, EV_DROP_FOOD,
                        EV_EAT_FRUIT, EV_EAT_MEAT, EV_EAT_PLANT, EV_KILL_LARGE, EV_KILL_SMALL,
                        EV_PICKUP_FOOD, EV_PREDATOR_ATTACK, EV_SIGNAL, EV_STRIKE, FOREST, GRASS,
                        HUNT, LARGE_PREY, MEADOW, MOUNTAIN, MOVE_E, MOVE_N, MOVE_S, MOVE_W, PICKUP,
                        PREDATOR, REST, ROCK, SHALLOW, SMALL_PREY)
from .layout import *  # noqa: F401,F403  (column constants)
from .rng import randint, uniform

# RNG streams
ST_ORDER, ST_HUNT, ST_KICK, ST_FAUNA, ST_FAUNA_MOVE, ST_FAUNA_REPRO, ST_SPAWN, ST_FIRE, \
    ST_FIRE_OUT, ST_BUILT, ST_PRED, ST_GENE, ST_LIFESPAN, ST_RESPAWN = range(14)

DX4 = np.array([0, 0, 1, 0, -1], dtype=np.int64)   # self, N, E, S, W
DY4 = np.array([0, -1, 0, 1, 0], dtype=np.int64)


@njit(cache=True, inline="always")
def push_event(EV, n_ev, tick, typ, a, b, x, y, v):
    k = n_ev[0]
    if k < EV.shape[0]:
        EV[k, 0] = tick
        EV[k, 1] = typ
        EV[k, 2] = a
        EV[k, 3] = b
        EV[k, 4] = x
        EV[k, 5] = y
        EV[k, 6] = v
    n_ev[0] = k + 1


@njit(cache=True, inline="always")
def cell_temp(P, elev, y, x):
    return P[P_T_GLOBAL] - P[P_LAPSE] * max(elev[y, x] - P[P_SEA], 0.0)


@njit(cache=True, inline="always")
def body_free(terrain, occ, fidx, y, x):
    h, w = terrain.shape
    if y < 0 or x < 0 or y >= h or x >= w:
        return False
    t = terrain[y, x]
    return t != DEEP_WATER and t != MOUNTAIN and occ[y, x] < 0 and fidx[y, x] < 0


@njit(cache=True)
def kill_agent(i, cause, A, I, occ, food_item, item_owner, EV, n_ev, tick):
    x, y = I[i, I_X], I[i, I_Y]
    I[i, I_ALIVE] = 0
    I[i, I_DIED_NOW] = 1
    I[i, I_LAST_CAUSE] = cause
    occ[y, x] = -1
    if I[i, I_CARRY_TYPE] == CARRY_FOOD and A[i, A_CARRY_AMT] > 0:
        food_item[y, x] += A[i, A_CARRY_AMT]
        item_owner[y, x] = I[i, I_UID]
    I[i, I_CARRY_TYPE] = CARRY_NONE
    A[i, A_CARRY_AMT] = 0.0
    push_event(EV, n_ev, tick, EV_DEATH, I[i, I_UID], I[i, I_AGE], x, y, cause)


@njit(cache=True)
def kill_fauna(j, FI, fidx, fcount):
    FI[j, F_ALIVE] = 0
    fidx[FI[j, F_Y], FI[j, F_X]] = -1
    fcount[FI[j, F_SPECIES]] -= 1


@njit(cache=True)
def effective_temp(P, i, x, y, terrain, elev, built, snow, occ, resting):
    T = cell_temp(P, elev, y, x)
    t = terrain[y, x]
    sheltered = t == CAVE or built[y, x] >= 1
    if t == CAVE:
        T += 10.0
    T += 4.0 * built[y, x]
    if t == FOREST:
        T += 2.0
    if resting:
        T += 1.5
    if P[P_RAINING] > 0 and not sheltered:
        T -= 3.0
    T -= 2.0 * snow[y, x]
    if P[P_HUDDLE] > 0:
        h, w = occ.shape
        for d in range(1, 5):
            ny, nx = y + DY4[d], x + DX4[d]
            if 0 <= ny < h and 0 <= nx < w and occ[ny, nx] >= 0:
                T += 1.5
    return T, sheltered


# =====================================================================================
# agents
# =====================================================================================
@njit(cache=True)
def agents_step(P, actions, order, A, I, S, G, HIST, occ, terrain, elev, fresh_now, flood,
                plant, tree_at, tree_fruit, food_item, item_owner, wood, built, fire, snow,
                fidx, FF, FI, fcount, EV, n_ev):
    seed = np.int64(P[P_SEED])
    tick = np.int64(P[P_TICK])
    h, w = terrain.shape
    hist_len = np.int64(P[P_HIST_LEN])
    build_idx = np.int64(P[P_BUILD_IDX])
    strike_idx = np.int64(P[P_STRIKE_IDX])
    signal_idx = np.int64(P[P_SIGNAL_IDX])
    n_sym = np.int64(P[P_N_SYMBOLS])
    for oi in range(order.shape[0]):
        i = order[oi]
        if I[i, I_ALIVE] == 0:
            continue
        a = np.int64(actions[i])
        x, y = I[i, I_X], I[i, I_Y]
        uid = I[i, I_UID]
        I[i, I_SIGNAL] = -1
        moved = False
        resting = False
        move_mult = 1.0
        extra_cost = 0.0
        size = 1.0 + 0.3 * G[i, 1]

        if MOVE_N <= a <= MOVE_W:
            nx, ny = x + DX4[a], y + DY4[a]
            if body_free(terrain, occ, fidx, ny, nx):
                occ[y, x] = -1
                occ[ny, nx] = i
                x, y = nx, ny
                I[i, I_X], I[i, I_Y] = x, y
                moved = True
                t = terrain[y, x]
                if t == SHALLOW or flood[y, x] > 0:
                    move_mult = 2.0
                elif t == ROCK:
                    move_mult = 1.6
                elif t == FOREST:
                    move_mult = 1.25
                move_mult += 0.5 * snow[y, x]
                S[i, S_DISTANCE] += 1.0
        elif a == REST:
            resting = True
        elif a == EAT:
            gained = 0.0
            if I[i, I_CARRY_TYPE] == CARRY_FOOD and A[i, A_CARRY_AMT] > 0:
                bite = min(P[P_MEAT_BITE], A[i, A_CARRY_AMT])
                A[i, A_CARRY_AMT] -= bite
                gained = bite * P[P_MEAT_E]
                if A[i, A_CARRY_AMT] <= 1e-6:
                    A[i, A_CARRY_AMT] = 0.0
                    I[i, I_CARRY_TYPE] = CARRY_NONE
                push_event(EV, n_ev, tick, EV_EAT_MEAT, uid, uid, x, y, gained)
            else:
                # ground food (carcass / dropped) on own or adjacent cell
                for d in range(5):
                    cx, cy = x + DX4[d], y + DY4[d]
                    if 0 <= cx < w and 0 <= cy < h and food_item[cy, cx] > 1e-6:
                        bite = min(P[P_MEAT_BITE], food_item[cy, cx])
                        food_item[cy, cx] -= bite
                        gained = bite * P[P_MEAT_E]
                        owner = item_owner[cy, cx]
                        if owner != 0 and owner != uid:
                            S[i, S_FOOD_FROM_OTHERS] += gained
                        push_event(EV, n_ev, tick, EV_EAT_MEAT, uid, owner, cx, cy, gained)
                        break
                if gained == 0.0:
                    for d in range(5):
                        cx, cy = x + DX4[d], y + DY4[d]
                        if 0 <= cx < w and 0 <= cy < h:
                            tr = tree_at[cy, cx]
                            if tr >= 0 and tree_fruit[tr] >= 1.0:
                                tree_fruit[tr] -= 1.0
                                gained = P[P_FRUIT_E]
                                push_event(EV, n_ev, tick, EV_EAT_FRUIT, uid, 0, cx, cy, gained)
                                break
                if gained == 0.0 and plant[y, x] > 0.01:
                    bite = min(P[P_PLANT_BITE], plant[y, x])
                    plant[y, x] -= bite
                    gained = bite * P[P_PLANT_E]
                    push_event(EV, n_ev, tick, EV_EAT_PLANT, uid, 0, x, y, gained)
            A[i, A_ENERGY] = min(1.0, A[i, A_ENERGY] + gained)
            S[i, S_ENERGY_EATEN] += gained
        elif a == DRINK:
            ok = False
            for d in range(5):
                cx, cy = x + DX4[d], y + DY4[d]
                if 0 <= cx < w and 0 <= cy < h and (fresh_now[cy, cx] or flood[cy, cx] > 0):
                    ok = True
                    break
            amt = P[P_DRINK] if ok else (P[P_RAIN_DRINK] if P[P_RAINING] > 0 else 0.0)
            if amt > 0:
                A[i, A_HYDRATION] = min(1.0, A[i, A_HYDRATION] + amt)
                S[i, S_WATER_DRUNK] += amt
                push_event(EV, n_ev, tick, EV_DRINK, uid, 0, x, y, amt)
        elif a == HUNT:
            best, best_prio = -1, 0
            for d in range(5):
                cx, cy = x + DX4[d], y + DY4[d]
                if 0 <= cx < w and 0 <= cy < h and fidx[cy, cx] >= 0:
                    j = fidx[cy, cx]
                    sp = FI[j, F_SPECIES]
                    prio = 3 if sp == PREDATOR else (2 if sp == LARGE_PREY else 1)
                    if prio > best_prio:
                        best, best_prio = j, prio
            extra_cost += P[P_MOVE]
            if best >= 0:
                j = best
                sp = FI[j, F_SPECIES]
                fx, fy = FI[j, F_X], FI[j, F_Y]
                if sp == SMALL_PREY:
                    if uniform(seed, tick, i, ST_HUNT) < 0.6:
                        kill_fauna(j, FI, fidx, fcount)
                        food_item[fy, fx] += P[P_SMALL_E]
                        item_owner[fy, fx] = uid
                        S[i, S_KILL_SMALL] += 1
                        push_event(EV, n_ev, tick, EV_KILL_SMALL, uid, 0, fx, fy, P[P_SMALL_E])
                elif sp == LARGE_PREY:
                    # record attacker (distinct uids within the window)
                    slot, oldest_t = -1, tick + 1
                    for s in range(N_ATT_SLOTS):
                        if FI[j, F_ATT0 + s] == uid:
                            slot = s
                            break
                        if FI[j, F_ATT_T0 + s] < oldest_t:
                            oldest_t = FI[j, F_ATT_T0 + s]
                            slot = s
                    FI[j, F_ATT0 + slot] = uid
                    FI[j, F_ATT_T0 + slot] = tick
                    n_att = 0
                    for s in range(N_ATT_SLOTS):
                        if FI[j, F_ATT0 + s] > 0 and tick - FI[j, F_ATT_T0 + s] <= P[P_ATT_WINDOW]:
                            n_att += 1
                    FF[j, F_HEALTH] -= 0.35 * size
                    FI[j, F_FLEE] = 12
                    push_event(EV, n_ev, tick, EV_ATTACK_LARGE, uid, j, fx, fy, n_att)
                    if FF[j, F_HEALTH] <= 0.0:
                        if P[P_COOP_LARGE] > 0 and n_att < P[P_MIN_ATTACKERS]:
                            FF[j, F_HEALTH] = 0.05      # wounded but escapes a lone hunter
                        else:
                            kill_fauna(j, FI, fidx, fcount)
                            food_item[fy, fx] += P[P_LARGE_E]
                            item_owner[fy, fx] = uid
                            S[i, S_KILL_LARGE] += 1
                            push_event(EV, n_ev, tick, EV_KILL_LARGE, uid, n_att, fx, fy, P[P_LARGE_E])
                    if uniform(seed, tick, i, ST_KICK) < 0.1:   # kicked
                        A[i, A_HEALTH] -= 0.05
                        S[i, S_DAMAGE_TAKEN] += 0.05
                        I[i, I_LAST_CAUSE] = DEATH_OTHER
                else:  # predator
                    FF[j, F_HEALTH] -= 0.3 * size
                    FI[j, F_FLEE] = 25
                    if FF[j, F_HEALTH] <= 0.0:
                        kill_fauna(j, FI, fidx, fcount)
                        food_item[fy, fx] += 0.8
                        item_owner[fy, fx] = uid
        elif a == PICKUP:
            ct = I[i, I_CARRY_TYPE]
            room = P[P_CARRY_CAP] - (A[i, A_CARRY_AMT] if ct == CARRY_FOOD else 0.0)
            done = False
            if (ct == CARRY_NONE or ct == CARRY_FOOD) and room > 1e-6:
                if food_item[y, x] > 1e-6:
                    take = min(room, food_item[y, x])
                    food_item[y, x] -= take
                    A[i, A_CARRY_AMT] += take
                    I[i, I_CARRY_TYPE] = CARRY_FOOD
                    push_event(EV, n_ev, tick, EV_PICKUP_FOOD, uid, item_owner[y, x], x, y, take)
                    done = True
                else:
                    for d in range(5):
                        cx, cy = x + DX4[d], y + DY4[d]
                        if 0 <= cx < w and 0 <= cy < h:
                            tr = tree_at[cy, cx]
                            if tr >= 0 and tree_fruit[tr] >= 1.0 and room >= P[P_FRUIT_E]:
                                tree_fruit[tr] -= 1.0
                                A[i, A_CARRY_AMT] += P[P_FRUIT_E]
                                I[i, I_CARRY_TYPE] = CARRY_FOOD
                                push_event(EV, n_ev, tick, EV_PICKUP_FOOD, uid, 0, cx, cy, P[P_FRUIT_E])
                                done = True
                                break
            if not done and ct == CARRY_NONE and P[P_NICHE] > 0:
                if terrain[y, x] == FOREST and wood[y, x] >= 0.2:
                    wood[y, x] -= 0.2
                    I[i, I_CARRY_TYPE] = CARRY_WOOD
                    A[i, A_CARRY_AMT] = 1.0
                elif terrain[y, x] == ROCK or terrain[y, x] == CAVE:
                    I[i, I_CARRY_TYPE] = CARRY_STONE
                    A[i, A_CARRY_AMT] = 1.0
        elif a == DROP:
            if I[i, I_CARRY_TYPE] == CARRY_FOOD and A[i, A_CARRY_AMT] > 0:
                amt = A[i, A_CARRY_AMT]
                food_item[y, x] += amt
                item_owner[y, x] = uid
                S[i, S_FOOD_DROPPED] += amt
                push_event(EV, n_ev, tick, EV_DROP_FOOD, uid, 0, x, y, amt)
            I[i, I_CARRY_TYPE] = CARRY_NONE
            A[i, A_CARRY_AMT] = 0.0
        elif a == build_idx and build_idx >= 0:
            ct = I[i, I_CARRY_TYPE]
            t = terrain[y, x]
            if (ct == CARRY_WOOD or ct == CARRY_STONE) and t != SHALLOW and built[y, x] < 3:
                built[y, x] += 1
                I[i, I_CARRY_TYPE] = CARRY_NONE
                A[i, A_CARRY_AMT] = 0.0
                S[i, S_BUILDS] += 1
                push_event(EV, n_ev, tick, EV_BUILD, uid, ct, x, y, built[y, x])
        elif a == strike_idx and strike_idx >= 0:
            extra_cost += P[P_MOVE]
            for d in range(1, 5):
                cx, cy = x + DX4[d], y + DY4[d]
                if 0 <= cx < w and 0 <= cy < h and occ[cy, cx] >= 0:
                    v = occ[cy, cx]
                    A[v, A_HEALTH] -= P[P_STRIKE_DMG] * size
                    S[v, S_DAMAGE_TAKEN] += P[P_STRIKE_DMG] * size
                    I[v, I_LAST_CAUSE] = DEATH_STRIKE
                    S[i, S_STRIKES] += 1
                    if I[v, I_CARRY_TYPE] == CARRY_FOOD and A[v, A_CARRY_AMT] > 0:
                        food_item[cy, cx] += A[v, A_CARRY_AMT]
                        item_owner[cy, cx] = I[v, I_UID]
                        A[v, A_CARRY_AMT] = 0.0
                        I[v, I_CARRY_TYPE] = CARRY_NONE
                    push_event(EV, n_ev, tick, EV_STRIKE, uid, I[v, I_UID], cx, cy, P[P_STRIKE_DMG] * size)
                    if A[v, A_HEALTH] <= 0 and I[v, I_ALIVE] == 1:
                        kill_agent(v, DEATH_STRIKE, A, I, occ, food_item, item_owner, EV, n_ev, tick)
                    break
        elif signal_idx >= 0 and signal_idx <= a < signal_idx + n_sym:
            I[i, I_SIGNAL] = a - signal_idx
            extra_cost += P[P_SIGNAL_COST]
            S[i, S_SIGNALS] += 1
            push_event(EV, n_ev, tick, EV_SIGNAL, uid, a - signal_idx, x, y, 0.0)

        I[i, I_LAST_ACTION] = a
        HIST[i, I[i, I_HIST_PTR]] = a
        I[i, I_HIST_PTR] = (I[i, I_HIST_PTR] + 1) % hist_len

        # ---- physiology -------------------------------------------------------------------
        T, sheltered = effective_temp(P, i, x, y, terrain, elev, built, snow, occ, resting)
        A[i, A_TEMP_EFF] = T
        I[i, I_SHELTERED] = 1 if sheltered else 0
        basal = P[P_BASAL] * (1.0 + 0.2 * G[i, 1]) * (P[P_REST] if resting else 1.0)
        cost = basal + extra_cost + P[P_CARRY] * A[i, A_CARRY_AMT]
        if moved:
            cost += P[P_MOVE] * move_mult
        A[i, A_ENERGY] -= cost
        hyd = P[P_HYD] * (P[P_HEAT_MULT] if T > P[P_HEAT] else 1.0) * (0.7 if resting else 1.0)
        A[i, A_HYDRATION] -= hyd
        cold = P[P_COLD] - 3.0 * G[i, 2]
        target = min(1.0, max(0.0, (T - cold) / (P[P_COMFORT] - cold)))
        A[i, A_WARMTH] += P[P_WARMTH_RATE] * (target - A[i, A_WARMTH])

        dmg = 0.0
        cause = I[i, I_LAST_CAUSE]
        if A[i, A_ENERGY] <= 0.0:
            A[i, A_ENERGY] = 0.0
            dmg += P[P_HUNGER_DMG]
            cause = DEATH_STARVATION
        if A[i, A_HYDRATION] <= 0.0:
            A[i, A_HYDRATION] = 0.0
            dmg += P[P_HUNGER_DMG]
            cause = DEATH_THIRST
        if A[i, A_WARMTH] < 0.2:
            dmg += P[P_COLD_DMG] * (1.0 - A[i, A_WARMTH] / 0.2)
            cause = DEATH_COLD
        if fire[y, x] > 0:
            dmg += P[P_FIRE_DMG]
            cause = DEATH_FIRE
        if dmg > 0:
            A[i, A_HEALTH] -= dmg
            S[i, S_DAMAGE_TAKEN] += dmg
            I[i, I_LAST_CAUSE] = cause
        elif A[i, A_ENERGY] > 0.4 and A[i, A_HYDRATION] > 0.4 and A[i, A_WARMTH] > 0.35:
            A[i, A_HEALTH] = min(1.0, A[i, A_HEALTH] + P[P_HEAL] * (2.0 if resting else 1.0))
        I[i, I_AGE] += 1
        if I[i, I_REPRO_COOLDOWN] > 0:
            I[i, I_REPRO_COOLDOWN] -= 1
        if A[i, A_HEALTH] <= 0.0:
            A[i, A_HEALTH] = 0.0
            kill_agent(i, I[i, I_LAST_CAUSE], A, I, occ, food_item, item_owner, EV, n_ev, tick)
        elif I[i, I_AGE] >= I[i, I_LIFESPAN]:
            kill_agent(i, DEATH_OLD_AGE, A, I, occ, food_item, item_owner, EV, n_ev, tick)


# =====================================================================================
# fauna
# =====================================================================================
@njit(cache=True)
def _nearest_threat(x, y, r, occ, fidx, FI, include_pred, h, w):
    """nearest agent (or predator) within Chebyshev radius r; returns (dist2, tx, ty) or (-1,..)."""
    best_d, bx, by = -1, 0, 0
    for dy in range(-r, r + 1):
        cy = y + dy
        if cy < 0 or cy >= h:
            continue
        for dx in range(-r, r + 1):
            cx = x + dx
            if cx < 0 or cx >= w or (dx == 0 and dy == 0):
                continue
            hit = occ[cy, cx] >= 0
            if not hit and include_pred and fidx[cy, cx] >= 0 and FI[fidx[cy, cx], F_SPECIES] == PREDATOR:
                hit = True
            if hit:
                d = dx * dx + dy * dy
                if best_d < 0 or d < best_d:
                    best_d, bx, by = d, cx, cy
    return best_d, bx, by


@njit(cache=True, inline="always")
def _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, y, x):
    if not body_free(terrain, occ, fidx, y, x):
        return False
    if fire[y, x] > 0:
        return False
    if sp == PREDATOR and (terrain[y, x] == CAVE or built[y, x] >= 2):
        return False
    return True


@njit(cache=True, inline="always")
def _move_fauna(j, nx, ny, FI, fidx):
    fidx[FI[j, F_Y], FI[j, F_X]] = -1
    FI[j, F_X], FI[j, F_Y] = nx, ny
    fidx[ny, nx] = j


@njit(cache=True)
def _step_toward(j, sp, tx, ty, away, FI, terrain, occ, fidx, built, fire):
    """greedy 4-neighbour step toward (or away from) a target."""
    x, y = FI[j, F_X], FI[j, F_Y]
    best_d, bx, by = -1.0, -1, -1
    for d in range(1, 5):
        nx, ny = x + DX4[d], y + DY4[d]
        if not _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, ny, nx):
            continue
        dd = float((nx - tx) ** 2 + (ny - ty) ** 2)
        score = dd if away else -dd
        if bx < 0 or score > best_d:
            best_d, bx, by = score, nx, ny
    if bx >= 0:
        _move_fauna(j, bx, by, FI, fidx)


@njit(cache=True)
def _spawn_fauna(sp, x, y, energy, FF, FI, fidx, fcount):
    """fcount[3] is a rotating free-slot hint so spawning is O(1) amortised."""
    M = FI.shape[0]
    for k in range(M):
        j = (fcount[3] + k) % M
        if FI[j, F_ALIVE] == 0:
            fcount[3] = (j + 1) % M
            FI[j, :] = 0
            FI[j, F_ALIVE] = 1
            FI[j, F_X], FI[j, F_Y], FI[j, F_SPECIES] = x, y, sp
            FF[j, F_ENERGY] = energy
            FF[j, F_HEALTH] = 1.0
            fidx[y, x] = j
            fcount[sp] += 1
            return j
    return -1


@njit(cache=True)
def _nearest_threat_blocks(x, y, r, occ, fidx, FI, tgrid, h, w):
    """_nearest_threat restricted to 4x4 blocks that contain an agent or predator."""
    gh, gw = tgrid.shape
    best_d, bx, by = -1, 0, 0
    for gy in range(max(0, (y - r) >> 2), min(gh - 1, (y + r) >> 2) + 1):
        for gx in range(max(0, (x - r) >> 2), min(gw - 1, (x + r) >> 2) + 1):
            if tgrid[gy, gx] == 0:
                continue
            for cy in range(max(gy * 4, y - r), min(gy * 4 + 4, y + r + 1, h)):
                for cx in range(max(gx * 4, x - r), min(gx * 4 + 4, x + r + 1, w)):
                    if cx == x and cy == y:
                        continue
                    hit = occ[cy, cx] >= 0
                    if not hit and fidx[cy, cx] >= 0 and FI[fidx[cy, cx], F_SPECIES] == PREDATOR:
                        hit = True
                    if hit:
                        d = (cx - x) * (cx - x) + (cy - y) * (cy - y)
                        if best_d < 0 or d < best_d:
                            best_d, bx, by = d, cx, cy
    return best_d, bx, by


@njit(cache=True)
def build_threat_grid(I, FI, tgrid):
    """4x4-block counts of agents and predators (prey use it to skip empty neighbourhoods)."""
    tgrid[:, :] = 0
    for i in range(I.shape[0]):
        if I[i, I_ALIVE] == 1:
            tgrid[I[i, I_Y] >> 2, I[i, I_X] >> 2] += 1
    for j in range(FI.shape[0]):
        if FI[j, F_ALIVE] == 1 and FI[j, F_SPECIES] == PREDATOR:
            tgrid[FI[j, F_Y] >> 2, FI[j, F_X] >> 2] += 1


@njit(cache=True, inline="always")
def _best_step(sp, x, y, tx, ty, away, terrain, occ, fidx, built, fire):
    """free 4-neighbour of (x, y) closest to (or farthest from) (tx, ty); (-1, -1) if none."""
    best_d, bx, by = -1.0, -1, -1
    for d in range(1, 5):
        nx, ny = x + DX4[d], y + DY4[d]
        if not _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, ny, nx):
            continue
        dd = float((nx - tx) ** 2 + (ny - ty) ** 2)
        score = dd if away else -dd
        if bx < 0 or score > best_d:
            best_d, bx, by = score, nx, ny
    return bx, by


@njit(cache=True, parallel=True)
def prey_proposals(P, FF, FI, fidx, occ, terrain, plant, built, fire, tgrid, prop):
    """Parallel phase: every prey animal decides where it wants to step, reading only the
    start-of-phase state (simultaneous update); conflicts are resolved sequentially in fauna_step."""
    seed = np.int64(P[P_SEED])
    tick = np.int64(P[P_TICK])
    h, w = terrain.shape
    flee_r = np.int64(P[P_FLEE_R])
    for j in prange(FI.shape[0]):
        prop[j, 0] = -1
        prop[j, 1] = -1
        if FI[j, F_ALIVE] == 0:
            continue
        sp = FI[j, F_SPECIES]
        if sp == PREDATOR:
            continue
        x, y = FI[j, F_X], FI[j, F_Y]
        u = uniform(seed, tick, j, ST_FAUNA_MOVE)
        r = flee_r if sp == SMALL_PREY else 3
        p_move = 0.6 if sp == SMALL_PREY else 0.25 + 0.4 * FF[j, F_HEALTH]
        d2, tx, ty = -1, 0, 0
        if u < p_move:
            d2, tx, ty = _nearest_threat_blocks(x, y, r, occ, fidx, FI, tgrid, h, w)
        bx, by = -1, -1
        if d2 >= 0:
            bx, by = _best_step(sp, x, y, tx, ty, True, terrain, occ, fidx, built, fire)
        elif sp == LARGE_PREY and u < 0.12:
            # loose herding toward local conspecific centroid
            sx, sy, n = 0, 0, 0
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    cy, cx = y + dy, x + dx
                    if 0 <= cy < h and 0 <= cx < w and fidx[cy, cx] >= 0 and fidx[cy, cx] != j \
                            and FI[fidx[cy, cx], F_SPECIES] == LARGE_PREY:
                        sx += cx
                        sy += cy
                        n += 1
            if n > 0:
                bx, by = _best_step(sp, x, y, sx // n, sy // n, False, terrain, occ, fidx, built, fire)
        elif u < 0.35:
            # graze toward the best neighbouring plant cell (random tie-break)
            best = plant[y, x] + 0.02
            k0 = randint(seed, tick, j, ST_FAUNA, 4)
            for k in range(4):
                d = 1 + (k0 + k) % 4
                nx, ny = x + DX4[d], y + DY4[d]
                if _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, ny, nx) and plant[ny, nx] > best:
                    best, bx, by = plant[ny, nx], nx, ny
            if bx < 0 and u < 0.12:
                d = 1 + randint(seed, tick, j, ST_FAUNA + 100, 4)
                nx, ny = x + DX4[d], y + DY4[d]
                if _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, ny, nx):
                    bx, by = nx, ny
        prop[j, 0] = bx
        prop[j, 1] = by


@njit(cache=True)
def fauna_step(P, order, FF, FI, fidx, fcount, occ, terrain, plant, food_item, item_owner, built,
               fire, A, I, S, EV, n_ev, prop):
    """Sequential phase: bookkeeping, prey moves proposed by prey_proposals, predators, births."""
    seed = np.int64(P[P_SEED])
    tick = np.int64(P[P_TICK])
    h, w = terrain.shape
    night = P[P_NIGHT] > 0.5
    flee_r = np.int64(P[P_FLEE_R])
    sense = np.int64(P[P_PRED_SENSE])
    max_age = np.array([12000, 20000, 20000])
    decay = np.array([0.0015, 0.0012, 0.0016])
    repro_rate = np.array([0.008, 0.0015, 0.0012])
    for oi in range(order.shape[0]):
        j = order[oi]
        if FI[j, F_ALIVE] == 0:
            continue
        sp = FI[j, F_SPECIES]
        x, y = FI[j, F_X], FI[j, F_Y]
        FI[j, F_AGE] += 1
        FF[j, F_ENERGY] -= decay[sp]
        if fire[y, x] > 0 or FF[j, F_ENERGY] <= 0 or FI[j, F_AGE] > max_age[sp]:
            kill_fauna(j, FI, fidx, fcount)
            continue
        u = uniform(seed, tick, j, ST_FAUNA_MOVE)
        if sp == SMALL_PREY or sp == LARGE_PREY:
            if sp == LARGE_PREY and FF[j, F_HEALTH] < 1.0:
                FF[j, F_HEALTH] = min(1.0, FF[j, F_HEALTH] + 0.004)
            if FI[j, F_FLEE] > 0:
                FI[j, F_FLEE] -= 1
            bx, by = prop[j, 0], prop[j, 1]      # decided in parallel by prey_proposals
            if bx >= 0 and _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, by, bx):
                _move_fauna(j, bx, by, FI, fidx)
            x, y = FI[j, F_X], FI[j, F_Y]
            bite = 0.03 if sp == SMALL_PREY else 0.06
            if FF[j, F_ENERGY] < 0.85 and plant[y, x] > bite:   # graze only when not sated
                plant[y, x] -= bite
                FF[j, F_ENERGY] = min(1.0, FF[j, F_ENERGY] + bite * (1.4 if sp == SMALL_PREY else 0.9))
        else:  # predator
            if FF[j, F_HEALTH] < 1.0:
                FF[j, F_HEALTH] = min(1.0, FF[j, F_HEALTH] + 0.002)
            if FI[j, F_FLEE] > 0:
                FI[j, F_FLEE] -= 1
                d2, tx, ty = _nearest_threat(x, y, sense, occ, fidx, FI, False, h, w)
                if d2 >= 0:
                    _step_toward(j, sp, tx, ty, True, FI, terrain, occ, fidx, built, fire)
            else:
                # nearest target: agent or prey
                best_d, tx, ty, tk, tj = -1, 0, 0, -1, -1
                for dy in range(-sense, sense + 1):
                    cy = y + dy
                    if cy < 0 or cy >= h:
                        continue
                    for dx in range(-sense, sense + 1):
                        cx = x + dx
                        if cx < 0 or cx >= w or (dx == 0 and dy == 0):
                            continue
                        kind, idx = -1, -1
                        if occ[cy, cx] >= 0:
                            v = occ[cy, cx]
                            if not (terrain[cy, cx] == CAVE or built[cy, cx] >= 2):
                                kind, idx = 0, v
                        elif fidx[cy, cx] >= 0 and FI[fidx[cy, cx], F_SPECIES] != PREDATOR:
                            kind, idx = 1, fidx[cy, cx]
                        if kind >= 0:
                            d = dx * dx + dy * dy
                            if best_d < 0 or d < best_d:
                                best_d, tx, ty, tk, tj = d, cx, cy, kind, idx
                if best_d == 1:  # adjacent: attack
                    if tk == 0:
                        if uniform(seed, tick, j, ST_PRED) < P[P_PRED_ATTACK]:
                            dmg = P[P_PRED_DMG]
                            A[tj, A_HEALTH] -= dmg
                            S[tj, S_DAMAGE_TAKEN] += dmg
                            I[tj, I_LAST_CAUSE] = DEATH_PREDATOR
                            push_event(EV, n_ev, tick, EV_PREDATOR_ATTACK, I[tj, I_UID], j, tx, ty, dmg)
                            if A[tj, A_HEALTH] <= 0.0:
                                A[tj, A_HEALTH] = 0.0
                                kill_agent(tj, DEATH_PREDATOR, A, I, occ, food_item, item_owner, EV, n_ev, tick)
                                FF[j, F_ENERGY] = min(1.0, FF[j, F_ENERGY] + 0.6)
                    else:
                        if FI[tj, F_SPECIES] == SMALL_PREY:
                            if uniform(seed, tick, j, ST_PRED) < 0.5:
                                kill_fauna(tj, FI, fidx, fcount)
                                FF[j, F_ENERGY] = min(1.0, FF[j, F_ENERGY] + 0.45)
                        else:
                            FF[tj, F_HEALTH] -= 0.15
                            FI[tj, F_FLEE] = 10
                            if FF[tj, F_HEALTH] <= 0:
                                kill_fauna(tj, FI, fidx, fcount)
                                FF[j, F_ENERGY] = 1.0
                                food_item[ty, tx] += 0.6 * P[P_LARGE_E]   # leftovers
                elif best_d > 1:
                    p_move = 0.85 if night else 0.55
                    if u < p_move:
                        _step_toward(j, sp, tx, ty, False, FI, terrain, occ, fidx, built, fire)
                elif u < 0.4:
                    d = 1 + randint(seed, tick, j, ST_FAUNA, 4)
                    nx, ny = x + DX4[d], y + DY4[d]
                    if _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, ny, nx):
                        _move_fauna(j, nx, ny, FI, fidx)
        # reproduction
        if FI[j, F_ALIVE] == 1 and FF[j, F_ENERGY] > 0.75 and fcount[sp] < P[P_FAUNA_MAX0 + sp]:
            rate = repro_rate[sp]
            if uniform(seed, tick, j, ST_FAUNA_REPRO) < rate:
                x, y = FI[j, F_X], FI[j, F_Y]
                d = 1 + randint(seed, tick, j, ST_FAUNA_REPRO + 100, 4)
                nx, ny = x + DX4[d], y + DY4[d]
                if _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, ny, nx):
                    FF[j, F_ENERGY] *= 0.5
                    _spawn_fauna(sp, nx, ny, FF[j, F_ENERGY], FF, FI, fidx, fcount)
    # immigration keeps every species above its floor
    for sp in range(3):
        k = 0
        while fcount[sp] < P[P_FAUNA_MIN0 + sp] and k < 50:
            k += 1
            x = randint(seed, tick, sp * 1000 + k, ST_SPAWN, w)
            y = randint(seed, tick, sp * 1000 + k, ST_SPAWN + 1, h)
            t = terrain[y, x]
            good = t == GRASS or t == MEADOW or (sp == PREDATOR and t == FOREST) or (sp == SMALL_PREY and t == FOREST)
            if good and _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, y, x):
                _spawn_fauna(sp, x, y, 0.8, FF, FI, fidx, fcount)


# =====================================================================================
# vegetation, fire, floods, snow
# =====================================================================================
@njit(cache=True, parallel=True)
def vegetation_step(P, dt, growth_mult, rate, seed_floor, wood_regrow, meat_decay, plant, fertility,
                    terrain, elev, fire, flood, snow, wood, food_item, item_owner, built):
    """Advances slow cell dynamics by `dt` ticks in one pass (called every dt ticks)."""
    seed = np.int64(P[P_SEED])
    tick = np.int64(P[P_TICK])
    h, w = plant.shape
    raining = P[P_RAINING] > 0
    for y in prange(h):
        for x in range(w):
            T = cell_temp(P, elev, y, x)
            K = fertility[y, x]
            f = fire[y, x]
            if f < 0:
                fire[y, x] = min(0, f + dt)   # burn scar recovering: no growth
            elif f == 0 and K > 0:
                tf = min(1.0, max(0.0, (T - 2.0) / 12.0))
                if tf > 0:
                    p = plant[y, x]
                    g = dt * rate * growth_mult * tf * max(p, seed_floor * K) * (1.0 - p / K)
                    plant[y, x] = min(K, p + g)
            if flood[y, x] > 0:
                flood[y, x] = max(0, flood[y, x] - dt)
                plant[y, x] *= 0.998 ** dt
            if terrain[y, x] == FOREST and wood[y, x] < 1.0 and f == 0:
                wood[y, x] = min(1.0, wood[y, x] + dt * wood_regrow)
            fi = food_item[y, x]
            if fi > 0:
                fi -= dt * meat_decay
                if fi <= 0:
                    fi = 0.0
                    item_owner[y, x] = 0
                food_item[y, x] = fi
            s = snow[y, x]
            if raining and T < 0.0 and terrain[y, x] != DEEP_WATER and terrain[y, x] != SHALLOW:
                snow[y, x] = min(1.0, s + dt * 0.004)
            elif s > 0 and T > 0.5:
                snow[y, x] = max(0.0, s - dt * 0.0015 * T)
            if built[y, x] > 0 and uniform(seed, tick, y * w + x, ST_BUILT) < dt * 2e-5:
                built[y, x] -= 1


@njit(cache=True)
def trees_step(tree_x, tree_y, tree_season, tree_fruit, tree_state, tree_at, fire, season, fruit_ok,
               fruit_rate, fruit_rot, fruit_max):
    for k in range(tree_x.shape[0]):
        x, y = tree_x[k], tree_y[k]
        if tree_state[k] < 0:              # dead (burnt): regrow countdown
            tree_state[k] += 1
            if tree_state[k] == 0:
                tree_at[y, x] = k
            continue
        if fire[y, x] > 0:
            tree_state[k] = -2 * 3840          # ~2 years to regrow
            tree_fruit[k] = 0.0
            tree_at[y, x] = -1
            continue
        if fruit_ok and tree_season[k] == season:
            tree_fruit[k] = min(fruit_max, tree_fruit[k] + fruit_rate)
        else:
            tree_fruit[k] = max(0.0, tree_fruit[k] - fruit_rot)


@njit(cache=True, inline="always")
def _flammable(t):
    return t == GRASS or t == MEADOW or t == FOREST or t == BARREN


@njit(cache=True, parallel=True)
def fire_step(P, spread, burn_ticks, dryness, fire, ignite, terrain, plant, wood, built, flood):
    """Returns number of burning cells after the update."""
    seed = np.int64(P[P_SEED])
    tick = np.int64(P[P_TICK])
    raining = P[P_RAINING] > 0
    h, w = fire.shape
    for y in prange(h):
        for x in range(w):
            ignite[y, x] = 0
            if fire[y, x] != 0 or not _flammable(terrain[y, x]) or flood[y, x] > 0:
                continue
            fuel = min(1.0, plant[y, x] + wood[y, x])
            if fuel < 0.05:
                continue
            k = 0
            for d in range(1, 5):
                ny, nx = y + DY4[d], x + DX4[d]
                if 0 <= ny < h and 0 <= nx < w and fire[ny, nx] > 0:
                    k += 1
            if k > 0:
                p = 1.0 - (1.0 - spread * dryness * fuel) ** k
                if raining:
                    p *= 0.2
                if uniform(seed, tick, y * w + x, ST_FIRE) < p:
                    ignite[y, x] = 1
    n = 0
    for y in prange(h):
        row = 0
        for x in range(w):
            f = fire[y, x]
            if f > 0:
                plant[y, x] = 0.0
                wood[y, x] = max(0.0, wood[y, x] - 0.03)
                built[y, x] = 0
                f -= 1
                if raining and uniform(seed, tick, y * w + x, ST_FIRE_OUT) < 0.08:
                    f = 0
                fire[y, x] = -burn_ticks * 40 if f == 0 else f
            if ignite[y, x] == 1:
                fire[y, x] = burn_ticks
            if fire[y, x] > 0:
                row += 1
        n += row
    return n


# =====================================================================================
# births (reproduction capability) and respawn (fixed-population mode)
# =====================================================================================
@njit(cache=True)
def _gauss(seed, tick, idx, stream):
    u1 = max(uniform(seed, tick, idx, stream), 1e-12)
    u2 = uniform(seed, tick, idx, stream + 7919)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


@njit(cache=True)
def init_agent(k, x, y, uid, parent, gen, genes, energy, hydration, warmth, lifespan_lo, lifespan_hi,
               seed, tick, A, I, S, G, HIST, occ):
    A[k, :] = 0.0
    I[k, :] = 0
    S[k, :] = 0.0
    HIST[k, :] = -1
    G[k, :] = genes
    A[k, A_ENERGY] = energy
    A[k, A_HYDRATION] = hydration
    A[k, A_HEALTH] = 1.0
    A[k, A_WARMTH] = warmth
    I[k, I_ALIVE] = 1
    I[k, I_X], I[k, I_Y] = x, y
    I[k, I_UID] = uid
    I[k, I_PARENT] = parent
    I[k, I_GEN] = gen
    I[k, I_SIGNAL] = -1
    I[k, I_LAST_ACTION] = -1
    I[k, I_BORN] = tick
    I[k, I_BORN_NOW] = 1
    base = lifespan_lo + (lifespan_hi - lifespan_lo) * uniform(seed, tick, uid, ST_LIFESPAN)
    I[k, I_LIFESPAN] = np.int64(base * (1.0 + 0.15 * genes[0]))
    occ[y, x] = k


@njit(cache=True)
def births_step(P, next_uid, maturity, energy_thr, hyd_thr, cost, cooldown, mut_std, lifespan_lo,
                lifespan_hi, order, A, I, S, G, HIST, occ, terrain, fidx, EV, n_ev):
    """Asexual reproduction with mutation. Returns updated next_uid."""
    seed = np.int64(P[P_SEED])
    tick = np.int64(P[P_TICK])
    N = A.shape[0]
    free = 0
    for oi in range(order.shape[0]):
        i = order[oi]
        if I[i, I_ALIVE] == 0 or I[i, I_BORN_NOW] == 1:
            continue
        if I[i, I_AGE] < maturity or I[i, I_REPRO_COOLDOWN] > 0:
            continue
        if A[i, A_ENERGY] < energy_thr or A[i, A_HYDRATION] < hyd_thr:
            continue
        x, y = I[i, I_X], I[i, I_Y]
        cx, cy = -1, -1
        d0 = randint(seed, tick, i, ST_SPAWN, 4)
        for k in range(4):
            d = 1 + (d0 + k) % 4
            nx, ny = x + DX4[d], y + DY4[d]
            if body_free(terrain, occ, fidx, ny, nx):
                cx, cy = nx, ny
                break
        if cx < 0:
            continue
        while free < N and I[free, I_ALIVE] == 1:
            free += 1
        if free >= N:
            break
        genes = G[i, :].copy()
        for g in range(genes.shape[0]):
            genes[g] = min(1.0, max(-1.0, genes[g] + mut_std * _gauss(seed, tick, next_uid * 64 + g, ST_GENE)))
        A[i, A_ENERGY] -= cost
        I[i, I_REPRO_COOLDOWN] = cooldown
        I[i, I_REPRODUCED_NOW] = 1
        S[i, S_CHILDREN] += 1
        init_agent(free, cx, cy, next_uid, I[i, I_UID], I[i, I_GEN] + 1, genes, cost * 0.9,
                   0.8, A[i, A_WARMTH], lifespan_lo, lifespan_hi, seed, tick, A, I, S, G, HIST, occ)
        push_event(EV, n_ev, tick, EV_BIRTH, next_uid, I[i, I_UID], cx, cy, I[i, I_GEN] + 1)
        next_uid += 1
    return next_uid


@njit(cache=True)
def respawn_step(P, next_uid, n_genes, lifespan_lo, lifespan_hi, slots, A, I, S, G, HIST, occ,
                 terrain, fidx, spawn_ok, EV, n_ev):
    """Fixed-population mode: dead slots get fresh agents (new uid, random genes) at random land cells."""
    seed = np.int64(P[P_SEED])
    tick = np.int64(P[P_TICK])
    h, w = terrain.shape
    for k in range(slots.shape[0]):
        i = slots[k]
        for attempt in range(200):
            x = randint(seed, tick, next_uid * 256 + attempt, ST_RESPAWN, w)
            y = randint(seed, tick, next_uid * 256 + attempt, ST_RESPAWN + 1, h)
            if spawn_ok[y, x] and body_free(terrain, occ, fidx, y, x):
                genes = np.empty(n_genes, dtype=np.float32)
                for g in range(n_genes):
                    genes[g] = 2.0 * uniform(seed, tick, next_uid * 64 + g, ST_GENE) - 1.0
                init_agent(i, x, y, next_uid, 0, 0, genes, 0.8, 0.8, 0.8, lifespan_lo, lifespan_hi,
                           seed, tick, A, I, S, G, HIST, occ)
                push_event(EV, n_ev, tick, EV_BIRTH, next_uid, 0, x, y, 0)
                next_uid += 1
                break
    return next_uid


# =====================================================================================
# observations
# =====================================================================================
@njit(cache=True, parallel=True)
def build_obs(P, R, r_eff, OFF, ODIST, RAY, RAYLEN, maturity, n_actions, social_on, rep_on, comms_on,
              hear_r, strike_idx, A, I, G, HIST, occ, terrain, fresh_now, flood, plant, tree_at, tree_fruit,
              fruit_max, food_item, fidx, FI, FF, fire, snow, built, grid_out, self_out, nb_out, heard_out):
    N = A.shape[0]
    h, w = terrain.shape
    K = nb_out.shape[1]
    n_genes = G.shape[1]
    hist_len = HIST.shape[1]
    n_sym = heard_out.shape[1]
    for i in prange(N):
        grid_out[i] = 0
        self_out[i] = 0.0
        nb_out[i] = 0.0
        heard_out[i] = 0.0
        if I[i, I_ALIVE] == 0:
            continue
        x, y = I[i, I_X], I[i, I_Y]
        kn = 0
        for o in range(OFF.shape[0]):
            if ODIST[o] > r_eff + 0.5:
                break
            dy, dx = OFF[o, 0], OFF[o, 1]
            cy, cx = y + dy, x + dx
            gy, gx = dy + R, dx + R
            if cy < 0 or cx < 0 or cy >= h or cx >= w:
                grid_out[i, 1, gy, gx] = 255
                continue
            blocked = False
            nf = 0
            for s in range(RAYLEN[o]):
                t = terrain[y + RAY[o, s, 0], x + RAY[o, s, 1]]
                if t == MOUNTAIN:
                    blocked = True
                    break
                if t == FOREST:
                    nf += 1
                    if nf >= 2:
                        blocked = True
                        break
            if blocked:
                continue
            t = terrain[cy, cx]
            grid_out[i, 0, gy, gx] = 255
            if t == DEEP_WATER or t == MOUNTAIN:
                grid_out[i, 1, gy, gx] = 255
            if fresh_now[cy, cx] or flood[cy, cx] > 0:
                grid_out[i, 2, gy, gx] = 255
            elif t == DEEP_WATER or t == SHALLOW:
                grid_out[i, 2, gy, gx] = 100
            if t == FOREST:
                grid_out[i, 3, gy, gx] = 255
            if t == ROCK or t == MOUNTAIN:
                grid_out[i, 4, gy, gx] = 255
            sh = 255 if t == CAVE else min(255, 80 * built[cy, cx])
            grid_out[i, 5, gy, gx] = sh
            grid_out[i, 6, gy, gx] = np.uint8(min(1.0, plant[cy, cx]) * 255)
            tr = tree_at[cy, cx]
            if tr >= 0:
                grid_out[i, 7, gy, gx] = np.uint8(40 + 215 * min(1.0, tree_fruit[tr] / fruit_max))
            grid_out[i, 8, gy, gx] = np.uint8(min(1.0, food_item[cy, cx] / 2.0) * 255)
            j = fidx[cy, cx]
            if j >= 0:
                sp = FI[j, F_SPECIES]
                grid_out[i, 9 + sp, gy, gx] = np.uint8(60 + 195 * max(0.0, min(1.0, FF[j, F_HEALTH])))
            f = fire[cy, cx]
            if f > 0:
                grid_out[i, 13, gy, gx] = 255
            elif f < 0:
                grid_out[i, 13, gy, gx] = 60
            grid_out[i, 15, gy, gx] = np.uint8(snow[cy, cx] * 255)
            v = occ[cy, cx]
            if v >= 0 and v != i:
                diff = 0.0
                for g in range(n_genes):
                    diff += abs(G[i, g] - G[v, g])
                kin = 1.0 - diff / (2.0 * n_genes)
                grid_out[i, 12, gy, gx] = np.uint8(100 + 155 * kin)
                if comms_on and I[v, I_SIGNAL] >= 0:
                    grid_out[i, 14, gy, gx] = 255
                if kn < K:
                    nb_out[i, kn, 0] = 1.0
                    nb_out[i, kn, 1] = dx / R
                    nb_out[i, kn, 2] = dy / R
                    nb_out[i, kn, 3] = A[v, A_HEALTH]
                    nb_out[i, kn, 4] = A[v, A_CARRY_AMT] if I[v, I_CARRY_TYPE] == CARRY_FOOD else 0.0
                    nb_out[i, kn, 5] = I[v, I_LAST_ACTION] if (social_on or rep_on) else -1.0
                    if rep_on:
                        ns, nk = 0, 0
                        for hh in range(hist_len):
                            if HIST[v, hh] == DROP:
                                ns += 1
                            elif strike_idx >= 0 and HIST[v, hh] == strike_idx:
                                nk += 1
                        nb_out[i, kn, 6] = ns / hist_len
                        nb_out[i, kn, 7] = nk / hist_len
                    nb_out[i, kn, 8] = kin
                    kn += 1
        # proprioception / interoception
        self_out[i, 0] = A[i, A_ENERGY]
        self_out[i, 1] = A[i, A_HYDRATION]
        self_out[i, 2] = A[i, A_HEALTH]
        self_out[i, 3] = A[i, A_WARMTH]
        self_out[i, 4] = I[i, I_AGE] / max(1, I[i, I_LIFESPAN])
        ct = I[i, I_CARRY_TYPE]
        if ct == CARRY_FOOD:
            self_out[i, 5] = A[i, A_CARRY_AMT]
        elif ct == CARRY_WOOD:
            self_out[i, 6] = 1.0
        elif ct == CARRY_STONE:
            self_out[i, 7] = 1.0
        self_out[i, 8] = A[i, A_TEMP_EFF] / 30.0
        self_out[i, 9] = 1.0 - P[P_NIGHT]
        self_out[i, 10] = P[P_RAINING]
        self_out[i, 11] = I[i, I_SHELTERED]
        self_out[i, 12] = 1.0 if I[i, I_AGE] >= maturity else 0.0
        if comms_on:
            for dy in range(-hear_r, hear_r + 1):
                cy = y + dy
                if cy < 0 or cy >= h:
                    continue
                for dx in range(-hear_r, hear_r + 1):
                    cx = x + dx
                    if cx < 0 or cx >= w or (dx == 0 and dy == 0):
                        continue
                    v = occ[cy, cx]
                    if v >= 0 and I[v, I_SIGNAL] >= 0:
                        d = np.sqrt(dx * dx + dy * dy)
                        if d <= hear_r:
                            s = I[v, I_SIGNAL]
                            if s < n_sym:
                                heard_out[i, s] += 1.0 / (1.0 + d)


@njit(cache=True)
def homeostatic_drive(A, setpoint, out):
    """D(h) = ||max(setpoint - h, 0)||_2 over (energy, hydration, health, warmth), per slot."""
    for i in range(A.shape[0]):
        s = 0.0
        for k in range(4):
            d = setpoint[k] - A[i, k]
            if d > 0:
                s += d * d
        out[i] = np.sqrt(s)
    return out
