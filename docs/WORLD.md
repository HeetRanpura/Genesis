# GENESIS world specification (Stage 1)

Everything an agent can experience is listed here. Anything that could *bias* social outcomes is
flagged **[affordance]** so results can be interpreted honestly: an affordance makes a behaviour
*possible or useful*; it never *rewards* it.

## Time
- 1 tick = one action per agent. `ticks_per_day = 120`, `days_per_season = 8`, year = 3,840 ticks.
- Seasons in order spring, summer, autumn, winter. Tick 0 is dawn on the first day of spring.
- Sun: `sin(2π(time_of_day − 0.25))`; night when the sun is below the horizon (view radius 5 → 2).

## Terrain (256×256 by default, procedurally generated from the seed)
Domain-warped fBm elevation with continental falloff → rank-normalised heights → priority-flood
depression filling (Barnes et al. 2014; filled depressions become lakes) → D8 flow accumulation
(rivers ≥ 180 upstream cells, streams ≥ 90) → moisture (noise + distance to fresh water, land
quantile) → biomes.

| class | passable | notes |
|---|---|---|
| deep water | no | ocean is **salt** (not drinkable); deep lake cores are fresh (drink from the shore) |
| shallow | yes (2× move cost) | rivers, streams, lake rims are fresh; ocean shoreline is salt |
| sand, grass, meadow, barren | yes | fertility differs (meadow highest) |
| forest | yes (1.25×) | wood, most fruit trees, +2 °C, two forest cells block line of sight |
| rock | yes (1.6×) | stone; hills |
| mountain | no | blocks line of sight |
| cave | yes | shelter: +10 °C, rain cover, predators never enter |

## Resources
- **Plants**: logistic regrowth toward per-cell fertility, scaled by season (1.2/1.0/0.55/0.05),
  rain (+25 %), drought (×0.4), famine (×0.25) and temperature (none below 2 °C). EAT bites 0.25
  biomass → 0.03 energy. `resource_layout: uniform` flattens fertility (Stage 7 A ablation).
- **Fruit trees**: each fruits in one season (spring/summer/autumn), up to 6 fruit, rot otherwise;
  EAT or PICKUP from an adjacent tree. Fire kills trees for ~2 years. Famine stops fruiting.
- **Carcasses / dropped food** (`food_item` grid): spoil linearly. Each cell remembers the uid of
  whoever dropped/killed it, so eating it is logged with the provider (sharing analysis).
- **Wood / stone** (only with `niche_construction`): PICKUP in forest/rock, BUILD on a cell
  (level 0–3; +4 °C per level, rain cover, level ≥ 2 repels predators; decays slowly; fire destroys).

## Fauna (scripted animals = the environment, never the agents)
- **Small prey** graze, flee agents/predators within 4 cells, reproduce when fed. HUNT adjacent:
  60 % kill → 0.30 energy carcass. One agent can do this.
- **Large prey** loosely herd, flee within 3, heal over time. Each HUNT hit does 0.35 damage.
  **[affordance] cooperative large prey** (`capabilities.cooperative_large_prey`, on by default):
  a kill needs ≥ 2 distinct attackers within 6 ticks; a lone hunter only wounds. Carcass = 1.8
  energy (several meals → co-feeding is possible). Switch off for the ablation.
- **Predators** hunt the nearest agent or prey within 6 cells (faster at night), attack adjacent
  agents (p = 0.4, 0.08 health), flee for 25 ticks when hit, never enter caves/shelters.
- Population floors (immigration) and caps keep species from going extinct or exploding.

## Weather & disturbances (daily Markov chain from a seeded RNG; all logged as events)
rain (season-dependent, persistent), fog (view radius 3), drought (summer; streams dry, growth ×0.4,
fire ×4), cold snap (winter; −10 °C), famine year (two seasons of ×0.25 growth, no fruit), wildfire
(ignition per day, spreads by fuel and dryness, rain extinguishes, leaves a scar), spring floods
(floodplain cells near rivers become shallow water for days). Snow accumulates where it rains
below 0 °C (+move cost, −warmth). All can also be triggered manually (`trigger_*`), which is logged
in `manual_log` and the event stream.

## Agents
- **Body**: energy, hydration, health, warmth ∈ [0, 1]; age; heritable lifespan (3–4 years).
  Costs: basal metabolism, movement × terrain, carrying, signalling. Dehydration accelerates
  above 31 °C. Warmth relaxes toward a target set by effective temperature (altitude, shelter,
  forest, rain, snow, resting). Health drops when energy or hydration hit 0, warmth < 0.2, fire,
  predators, kicks from large prey, or strikes; it regenerates when all needs are met.
- **Genes** (8 floats in [−1, 1]): g0 lifespan, g1 body size (hunting damage ↑, metabolism ↑),
  g2 cold tolerance, g3–g7 neutral markers (used for appearance/kin recognition).
- **Actions**: noop, move N/E/S/W, rest, eat, drink, hunt, pickup, drop; + build (niche
  construction), + strike (aggression), + `n_symbols` signals (communication).
- **Perception**: 11×11 egocentric patch, 16 uint8 channels (visible, blocked, water, forest, rock,
  shelter, plant, fruit, food item, small prey, large prey, predator, agent [value encodes
  **[affordance]** genetic similarity], fire, signal, snow); line of sight blocked by mountains and
  by two forest cells; radius shrinks at night/fog. Self vector (13 interoceptive features), up to 4
  nearest visible neighbours (position, health, carried food; last action only with
  `social_learning`/`reputation`; share/strike rates only with `reputation`), heard signals
  (communication only, radius 9, not occluded).
- **Food transfer**: PICKUP/DROP are always available, so food can be carried and dropped where
  another agent can take it. **[affordance]** This is what makes sharing *possible*; nothing makes
  it rewarding.
- **[affordance, off by default] huddle warmth**: each adjacent agent adds +1.5 °C.

## Reward (HARD RULE 1)
`r_t = D(h_t) − D(h_{t+1}) + death_penalty·died (+ reproduction_bonus·reproduced)`, where
`D(h) = ‖max(setpoint − h, 0)‖₂` over (energy, hydration, health, warmth), setpoint
(1, 1, 1, 0.8). This is homeostatic reinforcement learning (Keramati & Gutkin 2014, eLife).
It depends only on the agent's own body — enforced by `tests/test_engine.py::test_reward_is_own_homeostatic_drive_only`.

## Population modes
- Fixed population (`agents.respawn: true`, default): a dead agent's slot is refilled by a fresh
  agent (new uid, random genes) at a random land cell.
- Evolution (`capabilities.reproduction: true`): asexual births when mature and well-fed, genes
  mutate (σ = 0.05); no respawn.

## Determinism
All in-kernel randomness is `hash(seed, tick, index, stream)` (SplitMix64), sequential kernels iterate in
hash-derived permutations, parallel kernels draw no shared state. A run is a pure function of
(config, seed, actions) — `tests/test_engine.py::test_full_run_bit_reproducible`.
