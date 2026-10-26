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


# =====================================================================================
# observation image: img[y, x, c] holds every obs channel that does not depend on the observer
# (all except visible/agent/signal). It is updated in place at every mutation site so that an
# observation is a masked copy of 16-byte cells. tests/test_engine.py checks it against
# img_rebuild after long random runs.
# =====================================================================================
@njit(cache=True, inline="always")
def _u8(v):
    return np.uint8(min(255.0, max(0.0, v)))


@njit(cache=True, inline="always")
def img_plant(img, plant, y, x):
    img[y, x, 6] = _u8(plant[y, x] * 255.0)


@njit(cache=True, inline="always")
def img_food(img, food_item, y, x):
    img[y, x, 8] = _u8(food_item[y, x] * 127.5)


@njit(cache=True, inline="always")
def img_fruit(img, tree_at, tree_fruit, fruit_max, y, x):
    tr = tree_at[y, x]
    img[y, x, 7] = _u8(40.0 + 215.0 * min(1.0, tree_fruit[tr] / fruit_max)) if tr >= 0 else 0


@njit(cache=True, inline="always")
def img_fauna(img, FI, FF, j):
    """write animal j at its cell (value encodes health)."""
    y, x = FI[j, F_Y], FI[j, F_X]
    img[y, x, 9] = 0
    img[y, x, 10] = 0
    img[y, x, 11] = 0
    img[y, x, 9 + FI[j, F_SPECIES]] = _u8(60.0 + 195.0 * min(1.0, max(0.0, FF[j, F_HEALTH])))


@njit(cache=True, inline="always")
def img_no_fauna(img, y, x):
    img[y, x, 9] = 0
    img[y, x, 10] = 0
    img[y, x, 11] = 0


@njit(cache=True, inline="always")
def img_fire(img, fire, y, x):
    f = fire[y, x]
    img[y, x, 13] = 255 if f > 0 else (60 if f < 0 else 0)


@njit(cache=True, inline="always")
def img_ground(img, terrain, fresh_now, flood, built, y, x):
    """blocked, water, forest, rock, shelter."""
    t = terrain[y, x]
    img[y, x, 1] = 255 if (t == DEEP_WATER or t == MOUNTAIN) else 0
    img[y, x, 2] = 255 if (fresh_now[y, x] or flood[y, x] > 0) else (100 if (t == DEEP_WATER or t == SHALLOW) else 0)
    img[y, x, 3] = 255 if t == FOREST else 0
    img[y, x, 4] = 255 if (t == ROCK or t == MOUNTAIN) else 0
    img[y, x, 5] = 255 if t == CAVE else min(255, 80 * built[y, x])


@njit(cache=True, inline="always")
def img_cell(img, y, x, terrain, fresh_now, flood, built, plant, tree_at, tree_fruit, fruit_max, food_item,
             fidx, FI, FF, fire, snow):
    img_ground(img, terrain, fresh_now, flood, built, y, x)
    img_plant(img, plant, y, x)
    img_fruit(img, tree_at, tree_fruit, fruit_max, y, x)
    img_food(img, food_item, y, x)
    j = fidx[y, x]
    if j >= 0:
        img_fauna(img, FI, FF, j)
    else:
        img_no_fauna(img, y, x)
    img_fire(img, fire, y, x)
    img[y, x, 15] = _u8(snow[y, x] * 255.0)


@njit(cache=True, parallel=True)
def img_rebuild(img, terrain, fresh_now, flood, built, plant, tree_at, tree_fruit, fruit_max, food_item,
                fidx, FI, FF, fire, snow):
    h, w = terrain.shape
    for y in prange(h):
        for x in range(w):
            img[y, x, 0] = 255     # "visible": copied only for visible cells, so constant here
            img[y, x, 12] = 0
            img[y, x, 14] = 0
            img_cell(img, y, x, terrain, fresh_now, flood, built, plant, tree_at, tree_fruit, fruit_max,
                     food_item, fidx, FI, FF, fire, snow)


@njit(cache=True)
def kill_agent(i, cause, A, I, occ, food_item, item_owner, EV, n_ev, tick, img):
    x, y = I[i, I_X], I[i, I_Y]
    I[i, I_ALIVE] = 0
    I[i, I_DIED_NOW] = 1
    I[i, I_LAST_CAUSE] = cause
    occ[y, x] = -1
    if I[i, I_CARRY_TYPE] == CARRY_FOOD and A[i, A_CARRY_AMT] > 0:
        food_item[y, x] += A[i, A_CARRY_AMT]
        item_owner[y, x] = I[i, I_UID]
        img_food(img, food_item, y, x)
    I[i, I_CARRY_TYPE] = CARRY_NONE
    A[i, A_CARRY_AMT] = 0.0
    push_event(EV, n_ev, tick, EV_DEATH, I[i, I_UID], I[i, I_AGE], x, y, cause)


@njit(cache=True)
def kill_fauna(j, FI, fidx, fcount, img, fstat, cause):
    fstat[FI[j, F_SPECIES], cause] += 1
    img_no_fauna(img, FI[j, F_Y], FI[j, F_X])
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
                fidx, FF, FI, fcount, EV, n_ev, img, fstat):
    seed = np.int64(P[P_SEED])
    fruit_max = P[P_FRUIT_MAX]
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
                        img_food(img, food_item, cy, cx)
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
                                img_fruit(img, tree_at, tree_fruit, fruit_max, cy, cx)
                                gained = P[P_FRUIT_E]
                                push_event(EV, n_ev, tick, EV_EAT_FRUIT, uid, 0, cx, cy, gained)
                                break
                if gained == 0.0 and plant[y, x] > 0.01:
                    bite = min(P[P_PLANT_BITE], plant[y, x])
                    plant[y, x] -= bite
                    img_plant(img, plant, y, x)
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
                        kill_fauna(j, FI, fidx, fcount, img, fstat, FS_HUNTED)
                        food_item[fy, fx] += P[P_SMALL_E]
                        item_owner[fy, fx] = uid
                        img_food(img, food_item, fy, fx)
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
                    img_fauna(img, FI, FF, j)
                    push_event(EV, n_ev, tick, EV_ATTACK_LARGE, uid, j, fx, fy, n_att)
                    if FF[j, F_HEALTH] <= 0.0:
                        if P[P_COOP_LARGE] > 0 and n_att < P[P_MIN_ATTACKERS]:
                            FF[j, F_HEALTH] = 0.05      # wounded but escapes a lone hunter
                            img_fauna(img, FI, FF, j)
                        else:
                            kill_fauna(j, FI, fidx, fcount, img, fstat, FS_HUNTED)
                            food_item[fy, fx] += P[P_LARGE_E]
                            item_owner[fy, fx] = uid
                            img_food(img, food_item, fy, fx)
                            S[i, S_KILL_LARGE] += 1
                            push_event(EV, n_ev, tick, EV_KILL_LARGE, uid, n_att, fx, fy, P[P_LARGE_E])
                    if uniform(seed, tick, i, ST_KICK) < 0.1:   # kicked
                        A[i, A_HEALTH] -= 0.05
                        S[i, S_DAMAGE_TAKEN] += 0.05
                        I[i, I_LAST_CAUSE] = DEATH_OTHER
                else:  # predator
                    FF[j, F_HEALTH] -= 0.3 * size
                    FI[j, F_FLEE] = 25
                    img_fauna(img, FI, FF, j)
                    if FF[j, F_HEALTH] <= 0.0:
                        kill_fauna(j, FI, fidx, fcount, img, fstat, FS_HUNTED)
                        food_item[fy, fx] += 0.8
                        item_owner[fy, fx] = uid
                        img_food(img, food_item, fy, fx)
        elif a == PICKUP:
            ct = I[i, I_CARRY_TYPE]
            room = P[P_CARRY_CAP] - (A[i, A_CARRY_AMT] if ct == CARRY_FOOD else 0.0)
            done = False
            if (ct == CARRY_NONE or ct == CARRY_FOOD) and room > 1e-6:
                if food_item[y, x] > 1e-6:
                    take = min(room, food_item[y, x])
                    food_item[y, x] -= take
                    img_food(img, food_item, y, x)
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
                                img_fruit(img, tree_at, tree_fruit, fruit_max, cy, cx)
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
                img_food(img, food_item, y, x)
                S[i, S_FOOD_DROPPED] += amt
                push_event(EV, n_ev, tick, EV_DROP_FOOD, uid, 0, x, y, amt)
            I[i, I_CARRY_TYPE] = CARRY_NONE
            A[i, A_CARRY_AMT] = 0.0
        elif a == build_idx and build_idx >= 0:
            ct = I[i, I_CARRY_TYPE]
            t = terrain[y, x]
            if (ct == CARRY_WOOD or ct == CARRY_STONE) and t != SHALLOW and built[y, x] < 3:
                built[y, x] += 1
                img[y, x, 5] = 255 if t == CAVE else min(255, 80 * built[y, x])
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
                        img_food(img, food_item, cy, cx)
                        A[v, A_CARRY_AMT] = 0.0
                        I[v, I_CARRY_TYPE] = CARRY_NONE
                    push_event(EV, n_ev, tick, EV_STRIKE, uid, I[v, I_UID], cx, cy, P[P_STRIKE_DMG] * size)
                    if A[v, A_HEALTH] <= 0 and I[v, I_ALIVE] == 1:
                        kill_agent(v, DEATH_STRIKE, A, I, occ, food_item, item_owner, EV, n_ev, tick, img)
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
            kill_agent(i, I[i, I_LAST_CAUSE], A, I, occ, food_item, item_owner, EV, n_ev, tick, img)
        elif I[i, I_AGE] >= I[i, I_LIFESPAN]:
            kill_agent(i, DEATH_OLD_AGE, A, I, occ, food_item, item_owner, EV, n_ev, tick, img)


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
def _move_fauna(j, nx, ny, FI, FF, fidx, img):
    img_no_fauna(img, FI[j, F_Y], FI[j, F_X])
    fidx[FI[j, F_Y], FI[j, F_X]] = -1
    FI[j, F_X], FI[j, F_Y] = nx, ny
    fidx[ny, nx] = j
    img_fauna(img, FI, FF, j)


@njit(cache=True)
def _step_toward(j, sp, tx, ty, away, FI, FF, terrain, occ, fidx, built, fire, img):
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
        _move_fauna(j, bx, by, FI, FF, fidx, img)


@njit(cache=True)
def _spawn_fauna(sp, x, y, energy, FF, FI, fidx, fcount, img):
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
            img_fauna(img, FI, FF, j)
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


@njit(cache=True, inline="always")
def _prey_decide(j, seed, tick, flee_r, h, w, FF, FI, fidx, occ, terrain, plant, built, fire, tgrid, prop):
    """One prey animal's movement decision from start-of-phase state (no shared RNG, no writes
    except prop[j]), so the drivers below may run it in any order or in parallel."""
    prop[j, 0] = -1
    prop[j, 1] = -1
    if FI[j, F_ALIVE] == 0:
        return
    sp = FI[j, F_SPECIES]
    if sp == PREDATOR:
        return
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
        # wander when the neighbourhood is grazed out (probability higher on bare ground)
        if bx < 0 and (u < 0.12 or plant[y, x] < 0.1):
            d = 1 + randint(seed, tick, j, ST_FAUNA + 100, 4)
            nx, ny = x + DX4[d], y + DY4[d]
            if _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, ny, nx):
                bx, by = nx, ny
    prop[j, 0] = bx
    prop[j, 1] = by


@njit(cache=True, parallel=True)
def prey_proposals(P, FF, FI, fidx, occ, terrain, plant, built, fire, tgrid, prop):
    """Every prey animal decides where it wants to step, reading only the start-of-phase state
    (simultaneous update); conflicts are resolved sequentially in fauna_step."""
    h, w = terrain.shape
    for j in prange(FI.shape[0]):
        if FI[j, F_ALIVE] == 0 or FI[j, F_SPECIES] == PREDATOR:
            prop[j, 0] = -1
            continue
        _prey_decide(j, np.int64(P[P_SEED]), np.int64(P[P_TICK]), np.int64(P[P_FLEE_R]), h, w, FF, FI, fidx,
                     occ, terrain, plant, built, fire, tgrid, prop)


@njit(cache=True)
def prey_proposals_seq(P, FF, FI, fidx, occ, terrain, plant, built, fire, tgrid, prop):
    """Sequential twin of prey_proposals (identical output; cheaper below ~700 prey, where a
    parallel launch costs more than the work)."""
    h, w = terrain.shape
    for j in range(FI.shape[0]):
        if FI[j, F_ALIVE] == 0 or FI[j, F_SPECIES] == PREDATOR:
            prop[j, 0] = -1
            continue
        _prey_decide(j, np.int64(P[P_SEED]), np.int64(P[P_TICK]), np.int64(P[P_FLEE_R]), h, w, FF, FI, fidx,
                     occ, terrain, plant, built, fire, tgrid, prop)


@njit(cache=True)
def fauna_step(P, order, FF, FI, fidx, fcount, occ, terrain, plant, food_item, item_owner, built,
               fire, A, I, S, EV, n_ev, prop, img, FP, fstat, wood):
    """Sequential phase: bookkeeping, prey moves proposed by prey_proposals, predators, births."""
    seed = np.int64(P[P_SEED])
    tick = np.int64(P[P_TICK])
    h, w = terrain.shape
    night = P[P_NIGHT] > 0.5
    flee_r = np.int64(P[P_FLEE_R])
    sense = np.int64(P[P_PRED_SENSE])
    cold = P[P_FAUNA_COLD] if P[P_T_GLOBAL] < 5.0 else 1.0     # torpor: lower metabolism when cold
    for oi in range(order.shape[0]):
        j = order[oi]
        if FI[j, F_ALIVE] == 0:
            continue
        sp = FI[j, F_SPECIES]
        x, y = FI[j, F_X], FI[j, F_Y]
        FI[j, F_AGE] += 1
        FF[j, F_ENERGY] -= FP[sp, FP_METAB] * cold
        if fire[y, x] > 0 or FF[j, F_ENERGY] <= 0 or FI[j, F_AGE] > FP[sp, FP_MAX_AGE]:
            cause = FS_FIRE if fire[y, x] > 0 else (FS_STARVED if FF[j, F_ENERGY] <= 0 else FS_OLD)
            kill_fauna(j, FI, fidx, fcount, img, fstat, cause)
            continue
        u = uniform(seed, tick, j, ST_FAUNA_MOVE)
        if sp == SMALL_PREY or sp == LARGE_PREY:
            if sp == LARGE_PREY and FF[j, F_HEALTH] < 1.0:
                FF[j, F_HEALTH] = min(1.0, FF[j, F_HEALTH] + 0.004)
                img_fauna(img, FI, FF, j)
            if FI[j, F_FLEE] > 0:
                FI[j, F_FLEE] -= 1
            bx, by = prop[j, 0], prop[j, 1]      # decided in parallel by prey_proposals
            if bx >= 0 and _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, by, bx):
                _move_fauna(j, bx, by, FI, FF, fidx, img)
            x, y = FI[j, F_X], FI[j, F_Y]
            bite = min(FP[sp, FP_BITE], plant[y, x])
            if FF[j, F_ENERGY] < FP[sp, FP_SAT] and bite > 0.005:   # graze only when not sated
                plant[y, x] -= bite
                img_plant(img, plant, y, x)
                FF[j, F_ENERGY] = min(1.0, FF[j, F_ENERGY] + bite * FP[sp, FP_CONV])
            elif FP[sp, FP_BROWSE] > 0 and FF[j, F_ENERGY] < FP[sp, FP_SAT] and terrain[y, x] == FOREST \
                    and wood[y, x] > 0.05:
                # browsers (large prey) also eat woody forage in forests, which grazers cannot use
                b = min(FP[sp, FP_BITE], wood[y, x])
                wood[y, x] -= b
                FF[j, F_ENERGY] = min(1.0, FF[j, F_ENERGY] + b * FP[sp, FP_BROWSE])
        else:  # predator
            if FF[j, F_HEALTH] < 1.0:
                FF[j, F_HEALTH] = min(1.0, FF[j, F_HEALTH] + 0.002)
                img_fauna(img, FI, FF, j)
            if FI[j, F_FLEE] > 0:
                FI[j, F_FLEE] -= 1
                d2, tx, ty = _nearest_threat(x, y, sense, occ, fidx, FI, False, h, w)
                if d2 >= 0:
                    _step_toward(j, sp, tx, ty, True, FI, FF, terrain, occ, fidx, built, fire, img)
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
                                kill_agent(tj, DEATH_PREDATOR, A, I, occ, food_item, item_owner, EV, n_ev, tick, img)
                                FF[j, F_ENERGY] = min(1.0, FF[j, F_ENERGY] + FP[sp, FP_GAIN_AGENT])
                    else:
                        if FI[tj, F_SPECIES] == SMALL_PREY:
                            if uniform(seed, tick, j, ST_PRED) < 0.5:
                                kill_fauna(tj, FI, fidx, fcount, img, fstat, FS_EATEN)
                                FF[j, F_ENERGY] = min(1.0, FF[j, F_ENERGY] + FP[sp, FP_GAIN_SMALL])
                        else:
                            FF[tj, F_HEALTH] -= 0.15
                            FI[tj, F_FLEE] = 10
                            img_fauna(img, FI, FF, tj)
                            if FF[tj, F_HEALTH] <= 0:
                                kill_fauna(tj, FI, fidx, fcount, img, fstat, FS_EATEN)
                                FF[j, F_ENERGY] = min(1.0, FF[j, F_ENERGY] + FP[sp, FP_GAIN_LARGE])
                                food_item[ty, tx] += 0.6 * P[P_LARGE_E]   # leftovers
                                img_food(img, food_item, ty, tx)
                elif best_d > 1:
                    p_move = 0.85 if night else 0.55
                    if u < p_move:
                        _step_toward(j, sp, tx, ty, False, FI, FF, terrain, occ, fidx, built, fire, img)
                elif u < 0.4:
                    d = 1 + randint(seed, tick, j, ST_FAUNA, 4)
                    nx, ny = x + DX4[d], y + DY4[d]
                    if _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, ny, nx):
                        _move_fauna(j, nx, ny, FI, FF, fidx, img)
        # reproduction
        # births need stored energy AND standing forage where the animal is (density dependence via food)
        if P[P_FAUNA_BREED] > 0 and FI[j, F_ALIVE] == 1 and FF[j, F_ENERGY] > FP[sp, FP_REPRO_THR] \
                and plant[FI[j, F_Y], FI[j, F_X]] >= FP[sp, FP_BREED_FORAGE] and fcount[sp] < P[P_FAUNA_MAX0 + sp]:
            rate = FP[sp, FP_REPRO_RATE]
            if uniform(seed, tick, j, ST_FAUNA_REPRO) < rate:
                x, y = FI[j, F_X], FI[j, F_Y]
                d = 1 + randint(seed, tick, j, ST_FAUNA_REPRO + 100, 4)
                nx, ny = x + DX4[d], y + DY4[d]
                if _fauna_cell_ok(sp, terrain, occ, fidx, built, fire, ny, nx):
                    FF[j, F_ENERGY] *= 0.5
                    if _spawn_fauna(sp, nx, ny, FF[j, F_ENERGY], FF, FI, fidx, fcount, img) >= 0:
                        fstat[sp, FS_BIRTH] += 1
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
                if _spawn_fauna(sp, x, y, 0.8, FF, FI, fidx, fcount, img) >= 0:
                    fstat[sp, FS_IMMIGRANT] += 1


# =====================================================================================
# vegetation, fire, floods, snow
# =====================================================================================
@njit(cache=True, parallel=True)
def vegetation_step(P, dt, growth_mult, rate, seed_floor, wood_regrow, meat_decay, plant, fertility,
                    terrain, elev, fire, flood, snow, wood, food_item, item_owner, built, img, fresh_now):
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
            img_ground(img, terrain, fresh_now, flood, built, y, x)
            img_plant(img, plant, y, x)
            img_food(img, food_item, y, x)
            img_fire(img, fire, y, x)
            img[y, x, 15] = _u8(snow[y, x] * 255.0)


@njit(cache=True)
def trees_step(tree_x, tree_y, tree_season, tree_fruit, tree_state, tree_at, fire, season, fruit_ok,
               fruit_rate, fruit_rot, fruit_max, img):
    for k in range(tree_x.shape[0]):
        x, y = tree_x[k], tree_y[k]
        if tree_state[k] < 0:              # dead (burnt): regrow countdown
            tree_state[k] += 1
            if tree_state[k] == 0:
                tree_at[y, x] = k
                img_fruit(img, tree_at, tree_fruit, fruit_max, y, x)
            continue
        if fire[y, x] > 0:
            tree_state[k] = -2 * 3840          # ~2 years to regrow
            tree_fruit[k] = 0.0
            tree_at[y, x] = -1
            img[y, x, 7] = 0
            continue
        if fruit_ok and tree_season[k] == season:
            tree_fruit[k] = min(fruit_max, tree_fruit[k] + fruit_rate)
        else:
            tree_fruit[k] = max(0.0, tree_fruit[k] - fruit_rot)
        img_fruit(img, tree_at, tree_fruit, fruit_max, y, x)


@njit(cache=True, inline="always")
def _flammable(t):
    return t == GRASS or t == MEADOW or t == FOREST or t == BARREN


@njit(cache=True)
def _fire_ignite_row(y, x0, x1, seed, tick, raining, spread, dryness, fire, ignite, terrain, plant, wood, flood):
    h, w = fire.shape
    for x in range(x0, x1 + 1):
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


@njit(cache=True)
def _fire_update_row(y, x0, x1, seed, tick, raining, burn_ticks, fire, ignite, terrain, plant, wood, built, img,
                     row_min, row_max):
    w = fire.shape[1]
    row_min[y] = w
    row_max[y] = -1
    for x in range(x0, x1 + 1):
        f = fire[y, x]
        burning = f > 0
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
        if burning or ignite[y, x] == 1:
            img[y, x, 6] = _u8(plant[y, x] * 255.0)
            img[y, x, 5] = 255 if terrain[y, x] == CAVE else min(255, 80 * built[y, x])
            img_fire(img, fire, y, x)
        if fire[y, x] > 0:
            row_min[y] = min(row_min[y], x)
            row_max[y] = max(row_max[y], x)


@njit(cache=True)
def _fire_box(box, y0, y1, row_min, row_max, h, w):
    n = 0
    ny0, ny1, nx0, nx1 = h, -1, w, -1
    for y in range(y0, y1 + 1):
        if row_max[y] >= 0:
            n += 1
            ny0, ny1 = min(ny0, y), max(ny1, y)
            nx0, nx1 = min(nx0, row_min[y]), max(nx1, row_max[y])
    box[0], box[1], box[2], box[3] = ny0 - 1, ny1 + 1, nx0 - 1, nx1 + 1
    return n     # number of rows with fire (0 = extinguished)


@njit(cache=True, parallel=True)
def fire_step(P, spread, burn_ticks, dryness, fire, ignite, terrain, plant, wood, built, flood, img, box,
              row_min, row_max):
    """Advance burning cells inside box = [y0, y1, x0, x1] (burning cells' bounding box grown by one
    cell: ignition only spreads to 4-neighbours). Updates box in place; returns rows still burning."""
    seed, tick, raining = np.int64(P[P_SEED]), np.int64(P[P_TICK]), P[P_RAINING] > 0
    h, w = fire.shape
    y0, y1, x0, x1 = max(0, box[0]), min(h - 1, box[1]), max(0, box[2]), min(w - 1, box[3])
    for y in prange(y0, y1 + 1):
        _fire_ignite_row(y, x0, x1, seed, tick, raining, spread, dryness, fire, ignite, terrain, plant, wood, flood)
    for y in prange(y0, y1 + 1):
        _fire_update_row(y, x0, x1, seed, tick, raining, burn_ticks, fire, ignite, terrain, plant, wood, built,
                         img, row_min, row_max)
    return _fire_box(box, y0, y1, row_min, row_max, h, w)


@njit(cache=True)
def fire_step_seq(P, spread, burn_ticks, dryness, fire, ignite, terrain, plant, wood, built, flood, img, box,
                  row_min, row_max):
    """Sequential twin of fire_step (identical output; cheaper for small fires)."""
    seed, tick, raining = np.int64(P[P_SEED]), np.int64(P[P_TICK]), P[P_RAINING] > 0
    h, w = fire.shape
    y0, y1, x0, x1 = max(0, box[0]), min(h - 1, box[1]), max(0, box[2]), min(w - 1, box[3])
    for y in range(y0, y1 + 1):
        _fire_ignite_row(y, x0, x1, seed, tick, raining, spread, dryness, fire, ignite, terrain, plant, wood, flood)
    for y in range(y0, y1 + 1):
        _fire_update_row(y, x0, x1, seed, tick, raining, burn_ticks, fire, ignite, terrain, plant, wood, built,
                         img, row_min, row_max)
    return _fire_box(box, y0, y1, row_min, row_max, h, w)


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
def build_vismask(terrain, OFF, PARENT, vismask):
    """Static line of sight: bit o of vismask[y, x] (two uint64 words) is set when patch offset o is
    visible from (y, x). Visibility propagates outward along each offset's Bresenham parent (the last
    cell before it on the line): hidden if any ancestor is a mountain or two ancestors are forest.
    Terrain never changes during a run, so this is computed once per world."""
    h, w = terrain.shape
    n_off = OFF.shape[0]
    for y in prange(h):
        st = np.empty(n_off, np.int64)
        for x in range(w):
            m0 = np.uint64(0)
            m1 = np.uint64(0)
            for o in range(n_off):
                cy, cx = y + OFF[o, 0], x + OFF[o, 1]
                if cy < 0 or cx < 0 or cy >= h or cx >= w:
                    st[o] = -1
                    continue
                p = PARENT[o]
                v = 0
                if p >= 0:
                    v = st[p]
                    if v >= 0:
                        pt = terrain[y + OFF[p, 0], x + OFF[p, 1]]
                        if pt == MOUNTAIN:
                            v = -1
                        elif pt == FOREST:
                            v += 1
                            if v >= 2:
                                v = -1
                st[o] = v
                if v >= 0:
                    if o < 64:
                        m0 |= np.uint64(1) << np.uint64(o)
                    else:
                        m1 |= np.uint64(1) << np.uint64(o - 64)
            vismask[y, x, 0] = m0
            vismask[y, x, 1] = m1


@njit(cache=True, parallel=True)
def build_obs(P, R, r_eff, OFF, ODIST, maturity, social_on, rep_on, comms_on, hear_r, strike_idx,
              A, I, G, HIST, occ, img, img64, vismask, grid_out, grid64, self_out, nb_out, heard_out):
    """Egocentric observations, channel-last: grid_out[i, gy, gx, c] = img cell if visible
    (vismask, distance <= r_eff), plus observer-specific channels (visible, agent/kin, signal)."""
    N = A.shape[0]
    h, w = occ.shape
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
        m0 = vismask[y, x, 0]
        m1 = vismask[y, x, 1]
        kn = 0
        for o in range(OFF.shape[0]):
            if ODIST[o] > r_eff + 0.5:
                break
            dy, dx = OFF[o, 0], OFF[o, 1]
            cy, cx = y + dy, x + dx
            gy, gx = dy + R, dx + R
            if cy < 0 or cx < 0 or cy >= h or cx >= w:
                grid_out[i, gy, gx, 1] = 255
                continue
            if o < 64:
                bit = (m0 >> np.uint64(o)) & np.uint64(1)
            else:
                bit = (m1 >> np.uint64(o - 64)) & np.uint64(1)
            if bit == 0:
                continue
            grid64[i, gy, gx, 0] = img64[cy, cx, 0]     # 16 channels = two 64-bit words
            grid64[i, gy, gx, 1] = img64[cy, cx, 1]
            grid_out[i, gy, gx, 0] = 255
            v = occ[cy, cx]
            if v >= 0 and v != i:
                diff = 0.0
                for g in range(n_genes):
                    diff += abs(G[i, g] - G[v, g])
                kin = 1.0 - diff / (2.0 * n_genes)
                grid_out[i, gy, gx, 12] = np.uint8(100 + 155 * kin)
                if comms_on and I[v, I_SIGNAL] >= 0:
                    grid_out[i, gy, gx, 14] = 255
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
                            sg = I[v, I_SIGNAL]
                            if sg < n_sym:
                                heard_out[i, sg] += 1.0 / (1.0 + d)


@njit(cache=True)
def alive_permutation(alive_col, seed, tick, stream):
    """Hash-shuffled indices of rows whose alive flag is set (deterministic)."""
    n = 0
    for i in range(alive_col.shape[0]):
        if alive_col[i] == 1:
            n += 1
    p = np.empty(n, np.int64)
    k = 0
    for i in range(alive_col.shape[0]):
        if alive_col[i] == 1:
            p[k] = i
            k += 1
    for i in range(n - 1, 0, -1):
        j = randint(seed, tick, i, stream, i + 1)
        p[i], p[j] = p[j], p[i]
    return p


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


@njit(cache=True)
def begin_tick(I):
    for i in range(I.shape[0]):
        I[i, I_DIED_NOW] = 0
        I[i, I_BORN_NOW] = 0
        I[i, I_REPRODUCED_NOW] = 0


@njit(cache=True)
def tally_events(EV, n, ev_counts, deaths_by_cause):
    for k in range(n):
        t = np.int64(EV[k, 1])
        ev_counts[t] += 1
        if t == EV_DEATH:
            deaths_by_cause[np.int64(EV[k, 6])] += 1



@njit(cache=True)
def tick_rewards(prev_drive, setpoint, A, I, death_penalty, rewards, died):
    """r = D(before) - D(after) + death penalty; also returns the new drive in prev_drive's place."""
    for i in range(A.shape[0]):
        s = 0.0
        for k in range(4):
            d = setpoint[k] - A[i, k]
            if d > 0:
                s += d * d
        dr = np.sqrt(s)
        died[i] = I[i, I_DIED_NOW] == 1
        rewards[i] = prev_drive[i] - dr
        if died[i]:
            rewards[i] += death_penalty


@njit(cache=True)
def tick_masks(I, A, setpoint, died, rewards, born, alive, prev_drive):
    """After births/respawns: zero rewards of slots that held no agent this tick (a slot that died
    AND was refilled keeps the dead agent's transition), and store the drive for the next tick."""
    for i in range(I.shape[0]):
        born[i] = I[i, I_BORN_NOW] == 1
        alive[i] = I[i, I_ALIVE] == 1
        if not ((alive[i] and not born[i]) or died[i]):
            rewards[i] = 0.0
        s = 0.0
        for k in range(4):
            d = setpoint[k] - A[i, k]
            if d > 0:
                s += d * d
        prev_drive[i] = np.sqrt(s)


@njit(cache=True)
def alive_rotation(alive_col, seed, tick, stream):
    """Indices of alive rows in index order starting from a hash-chosen offset (cheaper than a full
    shuffle; used for fauna, where order only breaks ties between rare conflicting moves)."""
    M = alive_col.shape[0]
    n = 0
    for i in range(M):
        if alive_col[i] == 1:
            n += 1
    p = np.empty(n, np.int64)
    start = randint(seed, tick, 0, stream, M)
    k = 0
    for q in range(M):
        i = (start + q) % M
        if alive_col[i] == 1:
            p[k] = i
            k += 1
    return p


@njit(cache=True)
def world_tick(P, actions, A, I, S, G, HIST, occ, terrain, elev, fresh_now, flood, plant, fertility, tree_x,
               tree_y, tree_season, tree_fruit, tree_state, tree_at, food_item, item_owner, wood, built, fire,
               ignite, snow, fidx, FF, FI, FP, fcount, fstat, tgrid, fprop, EV, n_ev, img, fire_box, row_min,
               row_max, prev_drive, setpoint, rewards, died, born, alive,
               par_prey, do_veg, veg_dt, growth_mult, growth_rate, seed_floor, wood_regrow, meat_decay,
               season, fruit_ok, fruit_rate, fruit_rot, fruit_max, n_fire, par_fire, fire_spread, burn_ticks,
               dryness, death_penalty,
               spawn_mode, next_uid, n_genes, lifespan_lo, lifespan_hi, spawn_ok, maturity, rp_energy, rp_hyd,
               rp_cost, rp_cooldown, mut_std, repro_bonus, ev_counts, deaths_by_cause,
               compute_obs, R, r_eff, OFF, ODIST, social_on, rep_on, comms_on, hear_r, strike_idx, img64, vismask,
               grid_out, grid64, self_out, nb_out, heard_out):
    """One world tick in a single dispatch: agents -> fauna -> vegetation/trees -> fire -> rewards ->
    births/respawns (spawn_mode 1 = respawn dead slots, 2 = reproduction) -> reward masks -> event
    tally -> observations. Returns (fire row count, next uid, number of events)."""
    seed = np.int64(P[P_SEED])
    tick = np.int64(P[P_TICK])
    begin_tick(I)
    order = alive_permutation(I[:, I_ALIVE], seed, tick, ST_ORDER)
    agents_step(P, actions, order, A, I, S, G, HIST, occ, terrain, elev, fresh_now, flood, plant, tree_at,
                tree_fruit, food_item, item_owner, wood, built, fire, snow, fidx, FF, FI, fcount, EV, n_ev, img,
                fstat)
    forder = alive_rotation(FI[:, F_ALIVE], seed, tick, ST_ORDER + 50)
    build_threat_grid(I, FI, tgrid)
    if par_prey:
        prey_proposals(P, FF, FI, fidx, occ, terrain, plant, built, fire, tgrid, fprop)
    else:
        prey_proposals_seq(P, FF, FI, fidx, occ, terrain, plant, built, fire, tgrid, fprop)
    fauna_step(P, forder, FF, FI, fidx, fcount, occ, terrain, plant, food_item, item_owner, built, fire, A, I, S,
               EV, n_ev, fprop, img, FP, fstat, wood)
    if do_veg:
        vegetation_step(P, veg_dt, growth_mult, growth_rate, seed_floor, wood_regrow, meat_decay, plant,
                        fertility, terrain, elev, fire, flood, snow, wood, food_item, item_owner, built, img,
                        fresh_now)
    trees_step(tree_x, tree_y, tree_season, tree_fruit, tree_state, tree_at, fire, season, fruit_ok, fruit_rate,
               fruit_rot, fruit_max, img)
    if n_fire > 0:
        if par_fire:
            n_fire = fire_step(P, fire_spread, burn_ticks, dryness, fire, ignite, terrain, plant, wood, built,
                               flood, img, fire_box, row_min, row_max)
        else:
            n_fire = fire_step_seq(P, fire_spread, burn_ticks, dryness, fire, ignite, terrain, plant, wood,
                                   built, flood, img, fire_box, row_min, row_max)
    tick_rewards(prev_drive, setpoint, A, I, death_penalty, rewards, died)
    if spawn_mode == 2:
        border = alive_permutation(I[:, I_ALIVE], seed, tick, ST_ORDER + 99)
        next_uid = births_step(P, next_uid, maturity, rp_energy, rp_hyd, rp_cost, rp_cooldown, mut_std,
                               lifespan_lo, lifespan_hi, border, A, I, S, G, HIST, occ, terrain, fidx, EV, n_ev)
        for i in range(I.shape[0]):
            if I[i, I_REPRODUCED_NOW] == 1:
                rewards[i] += repro_bonus
    elif spawn_mode == 1:
        nd = 0
        for i in range(I.shape[0]):
            if died[i]:
                nd += 1
        if nd > 0:
            slots = np.empty(nd, np.int64)
            k = 0
            for i in range(I.shape[0]):
                if died[i]:
                    slots[k] = i
                    k += 1
            next_uid = respawn_step(P, next_uid, n_genes, lifespan_lo, lifespan_hi, slots, A, I, S, G, HIST, occ,
                                    terrain, fidx, spawn_ok, EV, n_ev)
    tick_masks(I, A, setpoint, died, rewards, born, alive, prev_drive)
    n = min(n_ev[0], EV.shape[0])
    tally_events(EV, n, ev_counts, deaths_by_cause)
    if compute_obs:
        build_obs(P, R, r_eff, OFF, ODIST, maturity, social_on, rep_on, comms_on, hear_r, strike_idx, A, I, G,
                  HIST, occ, img, img64, vismask, grid_out, grid64, self_out, nb_out, heard_out)
    return n_fire, next_uid, n_ev[0]
