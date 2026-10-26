"""GenesisEnv: batched, slot-based multi-agent environment.

Agents live in a fixed array of `max_agents` slots. A slot is either alive or empty; when an
agent dies its slot can later hold a newborn (new uid). `info["born"]` marks slots that hold
a new agent this tick, so learners must reset recurrent state for those slots.

Reward (HARD RULE 1) depends only on the agent's own body: homeostatic drive reduction
(Keramati & Gutkin 2014) + death penalty (+ optional reproduction bonus).
"""
from __future__ import annotations

import math

import numpy as np

from . import kernels as K
from .config import Config, load_config
from .constants import (BASE_ACTION_NAMES, BARREN, CLEAR, DEATH_NAMES, EV_COLD_SNAP_START,
                        EV_DROUGHT_START, EV_FAMINE_START, EV_FIRE_START, EV_FLOOD_START,
                        EVENT_NAMES, FOG, FOREST, GRASS, LARGE_PREY, MEADOW, N_BASE_ACTIONS,
                        N_OBS_CHANNELS, NEIGHBOR_FEATURES, PREDATOR, RAIN, ROCK, SAND,
                        SELF_FEATURES, SMALL_PREY)
from .layout import *  # noqa: F401,F403
from .rng import permutation
from .worldgen import generate_world

SEASON_NAMES = ["spring", "summer", "autumn", "winter"]


def make_rays(R: int):
    """Offsets of a (2R+1)^2 patch sorted by distance, with the Bresenham cells strictly between
    the centre and each offset (used for line-of-sight occlusion)."""
    offs = [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]
    offs.sort(key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    n = len(offs)
    lmax = 2 * R + 1
    ray = np.zeros((n, lmax, 2), dtype=np.int64)
    raylen = np.zeros(n, dtype=np.int64)
    for k, (dy, dx) in enumerate(offs):
        cells = []
        x0, y0, x1, y1 = 0, 0, dx, dy
        sx, sy = (1 if x1 > 0 else -1), (1 if y1 > 0 else -1)
        ddx, ddy = abs(x1), -abs(y1)
        err = ddx + ddy
        x, y = x0, y0
        while (x, y) != (x1, y1):
            e2 = 2 * err
            if e2 >= ddy:
                err += ddy
                x += sx
            if e2 <= ddx:
                err += ddx
                y += sy
            if (x, y) != (x1, y1):
                cells.append((y, x))
        raylen[k] = len(cells)
        for s, c in enumerate(cells):
            ray[k, s] = c
    off = np.array(offs, dtype=np.int64)
    dist = np.sqrt((off ** 2).sum(1)).astype(np.float64)
    return off, dist, ray, raylen


class GenesisEnv:
    def __init__(self, cfg: Config | None = None, seed: int | None = None):
        self.cfg = cfg if cfg is not None else load_config()
        self.seed = int(self.cfg.seed if seed is None else seed)
        c = self.cfg
        cap = c.capabilities
        # action space layout
        names = list(BASE_ACTION_NAMES)
        self.build_idx = self.strike_idx = self.signal_idx = -1
        if cap.niche_construction:
            self.build_idx = len(names)
            names.append("build")
        if cap.aggression:
            self.strike_idx = len(names)
            names.append("strike")
        if cap.communication:
            self.signal_idx = len(names)
            names += [f"signal_{s}" for s in range(cap.n_symbols)]
        self.action_names = names
        self.n_actions = len(names)
        ac = c.agents
        self.N = ac.max_agents
        self.R = ac.view_radius
        self.n_symbols = cap.n_symbols
        self.obs_shapes = {
            "grid": (N_OBS_CHANNELS, 2 * self.R + 1, 2 * self.R + 1),
            "self": (len(SELF_FEATURES),),
            "neighbors": (ac.n_neighbors_obs, len(NEIGHBOR_FEATURES)),
            "heard": (self.n_symbols,),
        }
        self.OFF, self.ODIST, self.RAY, self.RAYLEN = make_rays(self.R)
        tc = c.time
        self.tpd = tc.ticks_per_day
        self.season_len = tc.ticks_per_day * tc.days_per_season
        self.year_len = 4 * self.season_len
        self.maturity = int(ac.maturity_years * self.year_len)
        self.lifespan_lo = ac.lifespan_years[0] * self.year_len
        self.lifespan_hi = ac.lifespan_years[1] * self.year_len
        self.reset()

    # ------------------------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
        c = self.cfg
        W = generate_world(c, self.seed)
        self.world = W
        self.terrain = W["terrain"]
        self.elev = W["elevation"]
        self.fertility = W["fertility"]
        h, w = self.terrain.shape
        self.H, self.W = h, w
        self.plant = (self.fertility * 0.6).astype(np.float32)
        self.wood = W["wood"].copy()
        self.food_item = np.zeros((h, w), np.float32)
        self.item_owner = np.zeros((h, w), np.int64)
        self.built = np.zeros((h, w), np.int64)
        self.fire = np.zeros((h, w), np.int64)
        self.ignite = np.zeros((h, w), np.int8)
        self.flood = np.zeros((h, w), np.int64)
        self.snow = np.zeros((h, w), np.float32)
        self.occ = -np.ones((h, w), np.int64)
        self.fidx = -np.ones((h, w), np.int64)
        self.fresh = W["fresh"]
        self.fresh_now = self.fresh.copy()
        land = np.isin(self.terrain, [GRASS, MEADOW, FOREST, SAND, BARREN])
        self.spawn_ok = land
        # floodplain: low land near big rivers
        from scipy import ndimage
        near_river = ndimage.binary_dilation(W["river"], iterations=c.disturbances.flood_radius)
        self.floodplain = near_river & np.isin(self.terrain, [GRASS, MEADOW, FOREST, SAND, BARREN])
        # trees
        self.tree_x, self.tree_y = W["tree_x"], W["tree_y"]
        self.tree_season = W["tree_season"]
        nt = self.tree_x.size
        self.tree_fruit = np.zeros(nt, np.float32)
        self.tree_state = np.zeros(nt, np.int64)
        self.tree_at = -np.ones((h, w), np.int64)
        self.tree_at[self.tree_y, self.tree_x] = np.arange(nt)
        # agents
        N, ac = self.N, c.agents
        self.A = np.zeros((N, NA), np.float32)
        self.I = np.zeros((N, NI), np.int64)
        self.S = np.zeros((N, NS), np.float64)
        self.G = np.zeros((N, ac.n_genes), np.float32)
        self.HIST = -np.ones((N, ac.history_len), np.int64)
        # fauna
        fc = c.fauna
        self.fauna_max = np.array([fc.small_prey.max, fc.large_prey.max, fc.predators.max])
        self.fauna_min = np.array([fc.small_prey.min, fc.large_prey.min, fc.predators.min])
        M = int(self.fauna_max.sum()) + 64
        self.FF = np.zeros((M, NFF), np.float32)
        self.FI = np.zeros((M, NFI), np.int64)
        self.fcount = np.zeros(4, np.int64)          # 3 species counts + free-slot hint
        self.tgrid = np.zeros(((h + 3) // 4, (w + 3) // 4), np.int64)
        self.fprop = np.zeros((M, 2), np.int64)
        # events
        self.EV = np.zeros((200_000, EV_COLS), np.float64)
        self.n_ev = np.zeros(1, np.int64)
        self.event_log: list[np.ndarray] = []
        self.keep_events = bool(c.logging.events)
        self.ev_counts = np.zeros(len(EVENT_NAMES), np.int64)
        self.deaths_by_cause = np.zeros(len(DEATH_NAMES), np.int64)
        # climate state
        self.tick = 0
        self.next_uid = 1
        self.rng = np.random.default_rng([self.seed, 0xC11A7E])
        self.weather = CLEAR
        self.drought_left = 0
        self.cold_snap_left = 0
        self.famine_left = 0
        self.n_fire = 0
        self.manual_log: list[dict] = []
        self.P = np.zeros(NP, np.float64)
        self._fill_static_params()
        self._climate_tick(new_day=True)

        # initial population (deterministic, uses the respawn kernel)
        slots = np.arange(ac.n_initial, dtype=np.int64)
        self.next_uid = K.respawn_step(self.P, self.next_uid, ac.n_genes, self.lifespan_lo, self.lifespan_hi,
                                       slots, self.A, self.I, self.S, self.G, self.HIST, self.occ,
                                       self.terrain, self.fidx, self.spawn_ok, self.EV, self.n_ev)
        # random initial ages so the population is not synchronised
        alive = self.I[:, I_ALIVE] == 1
        self.I[alive, I_AGE] = (self.rng.random(alive.sum()) * 0.5 * self.I[alive, I_LIFESPAN]).astype(np.int64)
        self._spawn_initial_fauna()
        self.n_ev[0] = 0
        self.I[:, I_BORN_NOW] = 0
        self._prev_drive = self._drive()
        return self.observe()

    def _spawn_initial_fauna(self):
        fc = self.cfg.fauna
        t = self.terrain
        habitats = {
            SMALL_PREY: np.isin(t, [GRASS, MEADOW, FOREST]),
            LARGE_PREY: np.isin(t, [GRASS, MEADOW]),
            PREDATOR: t == FOREST,
        }
        n0 = {SMALL_PREY: fc.small_prey.initial, LARGE_PREY: fc.large_prey.initial, PREDATOR: fc.predators.initial}
        for sp in (SMALL_PREY, LARGE_PREY, PREDATOR):
            cand = np.flatnonzero(habitats[sp] & (self.occ < 0) & (self.fidx < 0))
            if sp == LARGE_PREY:   # herds: cluster around a few centres
                centres = self.rng.choice(cand, size=max(1, n0[sp] // 12), replace=False)
                picks = []
                for cidx in centres:
                    cy, cx = divmod(int(cidx), self.W)
                    for _ in range(12):
                        y = int(np.clip(cy + self.rng.integers(-4, 5), 0, self.H - 1))
                        x = int(np.clip(cx + self.rng.integers(-4, 5), 0, self.W - 1))
                        if habitats[sp][y, x]:
                            picks.append(y * self.W + x)
                picks = np.array(picks)
            else:
                picks = self.rng.choice(cand, size=min(n0[sp], cand.size), replace=False)
            for p in picks:
                y, x = divmod(int(p), self.W)
                if self.fidx[y, x] < 0 and self.occ[y, x] < 0:
                    K._spawn_fauna(sp, x, y, 0.8, self.FF, self.FI, self.fidx, self.fcount)

    # ------------------------------------------------------------------------------------
    def _fill_static_params(self):
        c, P = self.cfg, self.P
        ac, cl, rc, cap, fc = c.agents, c.climate, c.resources, c.capabilities, c.fauna
        P[P_SEED] = self.seed
        P[P_LAPSE] = cl.lapse_rate
        P[P_SEA] = c.world.sea_level
        P[P_COMFORT] = cl.comfort_temp
        P[P_COLD] = cl.cold_temp
        P[P_HEAT] = cl.heat_temp
        P[P_BASAL] = ac.basal_cost
        P[P_MOVE] = ac.move_cost
        P[P_CARRY] = ac.carry_cost
        P[P_REST] = ac.rest_factor
        P[P_HYD] = ac.hydration_cost
        P[P_HEAT_MULT] = ac.heat_hydration_mult
        P[P_WARMTH_RATE] = ac.warmth_rate
        P[P_HUNGER_DMG] = ac.hunger_damage
        P[P_COLD_DMG] = ac.cold_damage
        P[P_FIRE_DMG] = ac.fire_damage
        P[P_HEAL] = ac.heal_rate
        P[P_DRINK] = ac.drink_amount
        P[P_CARRY_CAP] = ac.carry_capacity
        P[P_STRIKE_DMG] = ac.strike_damage
        P[P_PLANT_BITE] = rc.plant_bite
        P[P_PLANT_E] = rc.plant_energy
        P[P_FRUIT_E] = rc.fruit_energy
        P[P_MEAT_BITE] = rc.meat_bite
        P[P_MEAT_E] = rc.meat_energy
        P[P_RAIN_DRINK] = rc.rain_drink
        P[P_N_SYMBOLS] = cap.n_symbols
        P[P_SIGNAL_COST] = cap.signal_cost
        P[P_NICHE] = float(cap.niche_construction)
        P[P_AGGRESSION] = float(cap.aggression)
        P[P_COMMS] = float(cap.communication)
        P[P_HUDDLE] = float(cap.huddle_warmth)
        P[P_COOP_LARGE] = float(cap.cooperative_large_prey)
        P[P_MIN_ATTACKERS] = fc.large_prey.min_attackers
        P[P_ATT_WINDOW] = fc.large_prey.attack_window
        P[P_FRUIT_MAX] = rc.fruit_max
        P[P_HIST_LEN] = ac.history_len
        P[P_SMALL_E] = fc.small_prey.energy
        P[P_LARGE_E] = fc.large_prey.energy
        P[P_PRED_DMG] = fc.predators.damage
        P[P_PRED_ATTACK] = fc.predators.attack_prob
        P[P_FLEE_R] = fc.flee_radius
        P[P_PRED_SENSE] = fc.predator_sense
        for s in range(3):
            P[P_FAUNA_MAX0 + s] = self.fauna_max[s]
            P[P_FAUNA_MIN0 + s] = self.fauna_min[s]
        P[P_BUILD_IDX] = self.build_idx
        P[P_STRIKE_IDX] = self.strike_idx
        P[P_SIGNAL_IDX] = self.signal_idx

    # ------------------------------------------------------------------------------------
    # time & climate
    @property
    def time_of_day(self) -> float:
        return ((self.tick + self.tpd // 4) % self.tpd) / self.tpd

    @property
    def season(self) -> int:
        return (self.tick // self.season_len) % 4

    @property
    def year(self) -> int:
        return self.tick // self.year_len

    @property
    def day(self) -> int:
        return self.tick // self.tpd

    def sun(self) -> float:
        return math.sin(2 * math.pi * (self.time_of_day - 0.25))

    def night_level(self) -> float:
        return float(1.0 - np.clip((self.sun() + 0.2) / 0.4, 0.0, 1.0))

    def global_temp(self) -> float:
        cl = self.cfg.climate
        phase = (self.tick % self.year_len) / self.year_len
        T = cl.base_temp + cl.seasonal_amplitude * math.cos(2 * math.pi * (phase - 0.375))
        T += cl.diurnal_amplitude * self.sun()
        if self.cold_snap_left > 0:
            T += self.cfg.weather.cold_snap_delta
        return T

    def growth_mult(self) -> float:
        rc = self.cfg.resources
        m = rc.season_growth[self.season]
        if self.weather == RAIN:
            m *= 1.25
        if self.drought_left > 0:
            m *= 0.4
        if self.famine_left > 0:
            m *= self.cfg.disturbances.famine_growth_mult
        return m

    def _log_env_event(self, typ, x=0, y=0, value=0.0):
        K.push_event(self.EV, self.n_ev, self.tick, typ, 0, 0, x, y, value)

    def _climate_tick(self, new_day: bool):
        wc, dc = self.cfg.weather, self.cfg.disturbances
        rng = self.rng
        if new_day:
            s = self.season
            for attr in ("drought_left", "cold_snap_left", "famine_left"):
                if getattr(self, attr) > 0:
                    setattr(self, attr, getattr(self, attr) - 1)
            if self.tick % self.year_len == 0 and rng.random() < dc.famine_prob_per_year:
                self.famine_left = 2 * self.cfg.time.days_per_season
                self._log_env_event(EV_FAMINE_START, value=self.famine_left)
            if s == 1 and self.drought_left == 0 and rng.random() < wc.drought_prob_summer:
                self.drought_left = int(rng.integers(wc.drought_days[0], wc.drought_days[1] + 1))
                self._log_env_event(EV_DROUGHT_START, value=self.drought_left)
            if s == 3 and self.cold_snap_left == 0 and rng.random() < wc.cold_snap_prob_winter:
                self.cold_snap_left = int(rng.integers(wc.cold_snap_days[0], wc.cold_snap_days[1] + 1))
                self._log_env_event(EV_COLD_SNAP_START, value=self.cold_snap_left)
            # weather Markov chain
            if self.drought_left > 0:
                self.weather = CLEAR
            elif self.weather == RAIN and rng.random() < wc.rain_persist:
                self.weather = RAIN
            elif rng.random() < wc.rain_prob[s]:
                self.weather = RAIN
            elif rng.random() < wc.fog_prob[s]:
                self.weather = FOG
            else:
                self.weather = CLEAR
            self.fresh_now = self.fresh & ~(self.world["stream"] & (self.drought_left > 0))
            # fire ignition
            season_f = [0.5, 1.5, 1.0, 0.2][s]
            p_fire = dc.fire_ignition_per_day * season_f * (4.0 if self.drought_left > 0 else 1.0)
            if self.weather != RAIN and rng.random() < p_fire:
                cand = np.flatnonzero(np.isin(self.terrain, [GRASS, FOREST, BARREN, MEADOW]).ravel()
                                      & ((self.plant + self.wood) > 0.3).ravel())
                if cand.size:
                    y, x = divmod(int(rng.choice(cand)), self.W)
                    self.trigger_fire(x, y, source="natural")
            # flood
            if s == 0 and self.weather == RAIN and rng.random() < dc.flood_prob_per_spring_rain_day:
                ry, rx = np.nonzero(self.world["river"])
                if ry.size:
                    k = int(rng.integers(ry.size))
                    self.trigger_flood(int(rx[k]), int(ry[k]), source="natural")
        P = self.P
        P[P_TICK] = self.tick
        P[P_T_GLOBAL] = self.global_temp()
        P[P_RAINING] = 1.0 if self.weather == RAIN else 0.0
        P[P_NIGHT] = self.night_level()
        P[P_DROUGHT] = 1.0 if self.drought_left > 0 else 0.0

    # ------------------------------------------------------------------------------------
    # disturbances (natural or manual; all logged)
    def trigger_fire(self, x: int, y: int, source="manual"):
        self.fire[y, x] = self.cfg.disturbances.fire_burn_ticks
        self._log_env_event(EV_FIRE_START, x, y, 1.0 if source == "manual" else 0.0)
        if source == "manual":
            self.manual_log.append(dict(tick=self.tick, kind="fire", x=x, y=y))

    def trigger_flood(self, x: int, y: int, days: int | None = None, radius: int = 30, source="manual"):
        dc = self.cfg.disturbances
        if days is None:
            days = int(self.rng.integers(dc.flood_days[0], dc.flood_days[1] + 1))
        yy, xx = np.ogrid[:self.H, :self.W]
        region = ((yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2) & self.floodplain
        self.flood[region] = np.maximum(self.flood[region], days * self.tpd)
        self._log_env_event(EV_FLOOD_START, x, y, float(region.sum()))
        if source == "manual":
            self.manual_log.append(dict(tick=self.tick, kind="flood", x=x, y=y, days=days))

    def trigger_famine(self, days: int):
        self.famine_left = days
        self._log_env_event(EV_FAMINE_START, value=days)
        self.manual_log.append(dict(tick=self.tick, kind="famine", days=days))

    def trigger_drought(self, days: int):
        self.drought_left = days
        self.weather = CLEAR
        self.fresh_now = self.fresh & ~self.world["stream"]
        self._log_env_event(EV_DROUGHT_START, value=days)
        self.manual_log.append(dict(tick=self.tick, kind="drought", days=days))

    def trigger_cold_snap(self, days: int):
        self.cold_snap_left = days
        self._log_env_event(EV_COLD_SNAP_START, value=days)
        self.manual_log.append(dict(tick=self.tick, kind="cold_snap", days=days))

    # ------------------------------------------------------------------------------------
    def _drive(self) -> np.ndarray:
        # columns 0..3 of A are (energy, hydration, health, warmth)
        assert (A_ENERGY, A_HYDRATION, A_HEALTH, A_WARMTH) == (0, 1, 2, 3)
        if not hasattr(self, "_setpoint"):
            self._setpoint = np.asarray(self.cfg.reward.setpoint, np.float64)
        return K.homeostatic_drive(self.A, self._setpoint, np.empty(self.N, np.float64))

    def step(self, actions: np.ndarray, compute_obs: bool = True):
        c = self.cfg
        rc, ac, cap = c.resources, c.agents, c.capabilities
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.N,), f"actions must have shape ({self.N},)"
        self.I[:, I_DIED_NOW] = 0
        self.I[:, I_BORN_NOW] = 0
        self.I[:, I_REPRODUCED_NOW] = 0
        self.n_ev[0] = 0
        new_day = self.tick % self.tpd == 0 and self.tick > 0
        self._climate_tick(new_day)
        P = self.P

        order = permutation(self.seed, self.tick, K.ST_ORDER, self.N)
        K.agents_step(P, actions, order, self.A, self.I, self.S, self.G, self.HIST, self.occ, self.terrain,
                      self.elev, self.fresh_now, self.flood, self.plant, self.tree_at, self.tree_fruit,
                      self.food_item, self.item_owner, self.wood, self.built, self.fire, self.snow,
                      self.fidx, self.FF, self.FI, self.fcount, self.EV, self.n_ev)
        forder = permutation(self.seed, self.tick, K.ST_ORDER + 50, self.FI.shape[0])
        K.build_threat_grid(self.I, self.FI, self.tgrid)
        K.prey_proposals(P, self.FF, self.FI, self.fidx, self.occ, self.terrain, self.plant, self.built,
                         self.fire, self.tgrid, self.fprop)
        K.fauna_step(P, forder, self.FF, self.FI, self.fidx, self.fcount, self.occ, self.terrain, self.plant,
                     self.food_item, self.item_owner, self.built, self.fire, self.A, self.I, self.S,
                     self.EV, self.n_ev, self.fprop)
        dt = c.time.veg_every
        if self.tick % dt == 0:
            K.vegetation_step(P, dt, self.growth_mult(), rc.plant_growth_rate, rc.plant_seed_floor, rc.wood_regrow,
                              rc.meat_decay, self.plant, self.fertility, self.terrain, self.elev, self.fire,
                              self.flood, self.snow, self.wood, self.food_item, self.item_owner, self.built)
        fruit_ok = self.famine_left == 0
        K.trees_step(self.tree_x, self.tree_y, self.tree_season, self.tree_fruit, self.tree_state, self.tree_at,
                     self.fire, self.season, fruit_ok, rc.fruit_rate, rc.fruit_rot, float(rc.fruit_max))
        if self.n_fire > 0 or (self.fire > 0).any():
            dryness = 1.6 if self.drought_left > 0 else [0.7, 1.2, 1.0, 0.5][self.season]
            self.n_fire = K.fire_step(P, c.disturbances.fire_spread_prob, c.disturbances.fire_burn_ticks, dryness,
                                      self.fire, self.ignite, self.terrain, self.plant, self.wood, self.built,
                                      self.flood)

        # rewards for agents alive at the start of this tick (incl. those who just died)
        drive = self._drive()
        died = self.I[:, I_DIED_NOW] == 1
        rewards = (self._prev_drive - drive).astype(np.float32)
        rewards[died] += c.reward.death_penalty

        # births / respawns
        border = permutation(self.seed, self.tick, K.ST_ORDER + 99, self.N)
        if cap.reproduction:
            rp = c.reproduction
            self.next_uid = K.births_step(P, self.next_uid, self.maturity, rp.energy_threshold,
                                          rp.hydration_threshold, rp.cost, int(rp.cooldown_years * self.year_len),
                                          ac.mutation_std, self.lifespan_lo, self.lifespan_hi, border, self.A,
                                          self.I, self.S, self.G, self.HIST, self.occ, self.terrain, self.fidx,
                                          self.EV, self.n_ev)
            rewards[self.I[:, I_REPRODUCED_NOW] == 1] += c.reward.reproduction_bonus
        elif ac.respawn:
            dead_slots = np.flatnonzero(died).astype(np.int64)
            if dead_slots.size:
                self.next_uid = K.respawn_step(P, self.next_uid, ac.n_genes, self.lifespan_lo, self.lifespan_hi,
                                               dead_slots, self.A, self.I, self.S, self.G, self.HIST, self.occ,
                                               self.terrain, self.fidx, self.spawn_ok, self.EV, self.n_ev)
        # a slot that died AND was refilled this tick still reports the dead agent's transition
        born = self.I[:, I_BORN_NOW] == 1
        alive_before = ((self.I[:, I_ALIVE] == 1) & ~born) | died
        rewards[~alive_before] = 0.0
        self._prev_drive = self._drive()

        # events
        n = min(int(self.n_ev[0]), self.EV.shape[0])
        if self.n_ev[0] > self.EV.shape[0]:
            raise RuntimeError("event buffer overflow; increase EV capacity")
        ev = self.EV[:n]
        if n:
            types = ev[:, 1].astype(np.int64)
            self.ev_counts += np.bincount(types, minlength=len(EVENT_NAMES))
            dmask = types == 1
            if dmask.any():
                self.deaths_by_cause += np.bincount(ev[dmask, 6].astype(np.int64), minlength=len(DEATH_NAMES))
            if self.keep_events:
                self.event_log.append(ev.copy())
        self.last_events = ev.copy()
        self.tick += 1
        info = {"died": died, "born": born, "alive": self.I[:, I_ALIVE] == 1, "uid": self.I[:, I_UID].copy()}
        obs = self.observe() if compute_obs else None
        return obs, rewards, died, info

    # ------------------------------------------------------------------------------------
    def view_radius_now(self) -> int:
        ac = self.cfg.agents
        r = self.R
        if self.night_level() > 0.5:
            r = min(r, ac.night_view_radius)
        if self.weather == FOG:
            r = min(r, ac.fog_view_radius)
        return r

    def observe(self) -> dict:
        if not hasattr(self, "_obs"):
            self._obs = {k: np.zeros((self.N,) + s, np.uint8 if k == "grid" else np.float32)
                         for k, s in self.obs_shapes.items()}
        o, cap, ac = self._obs, self.cfg.capabilities, self.cfg.agents
        K.build_obs(self.P, self.R, self.view_radius_now(), self.OFF, self.ODIST, self.RAY, self.RAYLEN,
                    self.maturity, self.n_actions, bool(cap.social_learning), bool(cap.reputation),
                    bool(cap.communication), ac.hearing_radius, self.strike_idx, self.A, self.I, self.G,
                    self.HIST, self.occ, self.terrain, self.fresh_now, self.flood, self.plant, self.tree_at,
                    self.tree_fruit, float(self.cfg.resources.fruit_max), self.food_item, self.fidx, self.FI,
                    self.FF, self.fire, self.snow, self.built, o["grid"], o["self"], o["neighbors"], o["heard"])
        return o

    # ------------------------------------------------------------------------------------
    @property
    def alive(self) -> np.ndarray:
        return self.I[:, I_ALIVE] == 1

    def population(self) -> int:
        return int(self.alive.sum())

    def metrics(self) -> dict:
        al = self.alive
        A = self.A[al]
        m = {
            "tick": self.tick, "day": self.day, "year": self.year, "season": self.season,
            "time_of_day": round(self.time_of_day, 4), "temp": round(self.global_temp(), 3),
            "weather": int(self.weather), "drought": int(self.drought_left > 0),
            "cold_snap": int(self.cold_snap_left > 0), "famine": int(self.famine_left > 0),
            "population": int(al.sum()),
            "small_prey": int(self.fcount[0]), "large_prey": int(self.fcount[1]), "predators": int(self.fcount[2]),
            "plant_total": float(self.plant.sum()), "fruit_total": float(self.tree_fruit.sum()),
            "fire_cells": int((self.fire > 0).sum()), "flood_cells": int((self.flood > 0).sum()),
            "snow_cells": int((self.snow > 0.2).sum()), "built_cells": int((self.built > 0).sum()),
            "food_items": float(self.food_item.sum()),
        }
        for k, name in zip([A_ENERGY, A_HYDRATION, A_HEALTH, A_WARMTH], ["energy", "hydration", "health", "warmth"]):
            m[f"mean_{name}"] = float(A[:, k].mean()) if len(A) else float("nan")
        m["mean_age_days"] = float(self.I[al, I_AGE].mean() / self.tpd) if al.any() else float("nan")
        m["max_generation"] = int(self.I[al, I_GEN].max()) if al.any() else 0
        for k, name in enumerate(DEATH_NAMES):
            m[f"deaths_{name}"] = int(self.deaths_by_cause[k])
        for k, name in enumerate(EVENT_NAMES):
            m[f"ev_{name}"] = int(self.ev_counts[k])
        return m

    def state_hash(self) -> str:
        import hashlib
        hsh = hashlib.sha256()
        for arr in (self.A, self.I, self.S, self.G, self.FF, self.FI, self.plant, self.food_item, self.fire,
                    self.flood, self.snow, self.tree_fruit, self.built, self.wood):
            hsh.update(np.ascontiguousarray(arr).tobytes())
        return hsh.hexdigest()

    def describe_actions(self) -> list[str]:
        return list(self.action_names)


__all__ = ["GenesisEnv", "SEASON_NAMES", "make_rays", "N_BASE_ACTIONS"]
