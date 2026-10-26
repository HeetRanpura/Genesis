# CLAUDE.md — working memory for GENESIS

Read this first in every session. Keep it current: update it whenever a stage lands, a design
decision is made, or a gotcha is found. `PLAN.md` holds milestones/acceptance criteria;
`logs/AGENT_LOG.md` is the running record of how agents are doing (append, never rewrite history).

## What this is
Artificial-life research testbed: survival-only RL agents in a seeded foraging ecology, plus (later)
a live PixiJS viewer. Research question: what social organisation/culture emerges with no social
reward, and how does its order of emergence compare with human prehistory (as structural analogy).

## Non-negotiable rules
1. Reward = own homeostasis only (`GenesisEnv._drive`), + death penalty (+ optional reproduction bonus).
   Never add a social term. `tests/test_engine.py::test_reward_is_own_homeostatic_drive_only` guards it.
2. Capabilities are config switches, default **off** (`configs/default.yaml: capabilities`).
3. Every run seeded + reproducible. Randomness inside kernels only via `sim/rng.py` hash RNG
   (`uniform(seed, tick, idx, stream)`); never `np.random` inside kernels; never draw shared RNG in `prange`.
4. No scripted agent behaviour. Animals are scripted (they are the environment).
5. Baselines (random, greedy, heuristic in `agents/baselines.py`) in every comparison.
6. Never claim emergence that was not measured; log negative results in AGENT_LOG.md.
7. Ecology features that make social behaviour *possible/useful* are tagged **[affordance]** in
   `docs/WORLD.md` (cooperative large prey, food drop/pickup, kin-similarity channel, huddle warmth).

## Layout
- `sim/` engine: `config.py` (YAML deep-merge, unknown keys raise), `constants.py` (enums = log format),
  `layout.py` (column indices of packed state arrays A/I/S/FF/FI and param vector P), `worldgen.py`,
  `kernels.py` (all Numba kernels), `env.py` (`GenesisEnv`, slot-based batched API), `pz_env.py`
  (PettingZoo ParallelEnv, agents named `agent_<uid>`), `logger.py` (RunLogger + `append_agent_log`).
- `agents/baselines.py`, `analysis/survival.py`, `experiments/*.py`, `tests/`, `reports/stageN/`, `docs/`.
- `runs/` is git-ignored (per-run config, manifest, metrics.csv, events.npy, summary.json).

## Commands
`make setup` · `make test` · `make bench` · `make baselines` · `make evidence` (Python is `.venv/bin/python`, 3.11).

## Engine facts / gotchas
- One tick = `GenesisEnv._climate_tick` (Python) + ONE Numba call `kernels.world_tick` that runs agents ->
  fauna -> vegetation/trees -> fire -> rewards -> births/respawns -> reward masks -> event tally -> obs.
  Keep per-tick work inside it: each extra Python-side kernel dispatch with many arrays costs ~5-10 us.
- Observations are channel-last `grid[agent, y, x, c]`. They are masked copies of `env.img[y, x, 16]`,
  an image of every observer-independent channel that is **patched in place at every world mutation
  site** (`img_*` helpers in kernels.py). If you add a mutation of plant/food/fruit/fauna/fire/snow/
  flood/built/fresh water, update img there too. `tests/test_engine.py::test_incremental_obs_image_matches_rebuild`
  catches drift. After editing world arrays directly (tests), call `env.refresh_img()` (also rebuilds
  the static line-of-sight masks `env.vismask`).
- Line of sight is precomputed per cell (`build_vismask`, terrain is static): parent-chain propagation.
- Sequential/parallel twins (`prey_proposals[_seq]`, `fire_step[_seq]`): Numba's macOS "workqueue" layer
  costs ~60 us per parallel launch, so small workloads run sequentially; outputs must stay identical
  (`test_sequential_and_parallel_twins_agree`). NEVER build a twin with `njit(cache=True)(f.py_func)`:
  it shares f's on-disk cache entry and silently overwrites it (this produced misleading benchmarks once).
- OpenMP threading layer was tried and rejected (slower and erratic here); stay on workqueue.
- Runs start after `time.spinup_years` (default 1) of ecology without agents; events/stats are reset.
- Agents occupy fixed slots (`max_agents`). `info["born"]` marks slots holding a NEW agent this tick
  (reset recurrent state there). A slot that died and was refilled in the same tick still reports
  the dead agent's reward/done (death penalty is not lost).
- One body per cell (agents and animals share the constraint).
- Adding a scalar param: append to the `P_*` tuple in `layout.py`, bump `NP`, set it in
  `GenesisEnv._fill_static_params` (or `_climate_tick` if dynamic).
- Adding an event type: append to `EVENT_NAMES` and the tuple in `constants.py` (log format!).
- Fauna update = `build_threat_grid` → `prey_proposals` (simultaneous decisions) → `fauna_step`
  (sequential resolve + predators + births). All fauna parameters live in `configs/default.yaml: fauna`
  (packed into `env.FP`, columns `FP_*` in layout.py); demographic counters in `env.fstat` (`FS_*`).
  Vegetation advances every `time.veg_every` (8) ticks with dt scaling.
- Fauna tuning check: `experiments/fauna_diagnostics.py` (caps should never bind; report floor time).
- Numba compile on first run ~20–40 s (cached in `__pycache__` afterwards).
- `fcount` has 4 entries: 3 species counts + a rotating free-slot hint.
- This machine is a fanless MacBook Air M4: sustained runs heat-throttle (~15-30% slower). Compare
  performance only with interleaved A/B runs in the same thermal state; quote medians.
- Test arenas: `tests/helpers.py` (`make_env`, `clear_world`, `put_agent`) — 64×64, no respawn,
  disturbances neutralised.

## Status (update as stages land)
- Stage 1 world engine: built, tested, benchmarked, evidence in `reports/stage1/`.
- Stage 1.1 (2026-10-26): fauna made food-limited (caps never bind), observation path rewritten
  (incremental obs image, static line-of-sight masks, fused world_tick) -> >=100k steps/min with obs at
  1,000 agents even when throttled. 41 tests. Awaiting user review before Stage 2.
- Known limitations: predators sit at their immigration floor ~30% of the time when 256 heuristic
  agents compete with them for small prey; greedy baseline ≈ random (water usually out of view → needs
  memory); wildfires can burn ~10% of land.

## Decisions log
- 2026-10-26: Numba (not JAX) for the engine — no CUDA on this M4; Numba multi-core gives >100k steps/min.
  PyTorch (MPS/CPU) planned for Stage 2 learners.
- 2026-10-26: Ocean is salt water; fresh water only from rivers/streams/lakes (makes water spatially scarce).
- 2026-10-26: Large prey require ≥2 attackers by default (stag-hunt pressure); ablation switch exists.
- 2026-10-26: Fauna regulation by food: seasonal breeding (spring-autumn), torpor (x0.2 metabolism below
  5 C), grazing-tolerant plant regrowth (seed floor 0.1), large-prey browsing niche (else small prey
  exclude them), 1-year ecological spin-up. Caps/floors are safety rails only.
