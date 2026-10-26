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
- Agents occupy fixed slots (`max_agents`). `info["born"]` marks slots holding a NEW agent this tick
  (reset recurrent state there). A slot that died and was refilled in the same tick still reports
  the dead agent's reward/done (death penalty is not lost).
- One body per cell (agents and animals share the constraint).
- Adding a scalar param: append to the `P_*` tuple in `layout.py`, bump `NP`, set it in
  `GenesisEnv._fill_static_params` (or `_climate_tick` if dynamic).
- Adding an event type: append to `EVENT_NAMES` and the tuple in `constants.py` (log format!).
- Fauna update = `build_threat_grid` → `prey_proposals` (parallel, simultaneous decisions) →
  `fauna_step` (sequential resolve + predators + births). Vegetation advances every `time.veg_every` ticks.
- Numba compile on first run ~20–40 s (cached in `__pycache__` afterwards).
- `fcount` has 4 entries: 3 species counts + a rotating free-slot hint.
- Test arenas: `tests/helpers.py` (`make_env`, `clear_world`, `put_agent`) — 64×64, no respawn,
  disturbances neutralised.

## Status (update as stages land)
- Stage 1 world engine: built, tested (38 tests), benchmarked, evidence in `reports/stage1/`.
  Awaiting user review before Stage 2.
- Known limitations: fauna populations often sit at their caps (cap-limited rather than food-limited);
  greedy baseline ≈ random (water usually out of view → needs memory); wildfires can burn ~10% of land.

## Decisions log
- 2026-10-26: Numba (not JAX) for the engine — no CUDA on this M4; Numba multi-core gives >100k steps/min.
  PyTorch (MPS/CPU) planned for Stage 2 learners.
- 2026-10-26: Ocean is salt water; fresh water only from rivers/streams/lakes (makes water spatially scarce).
- 2026-10-26: Large prey require ≥2 attackers by default (stag-hunt pressure); ablation switch exists.
