# Stage 1 report — world engine

Date: 2026-10-26 · Machine: Apple M4 (10 cores, 16 GB), Python 3.11, Numba 0.68, NumPy 2.4.
Reproduce everything: `make test && make baselines && make evidence && make bench`.

## Acceptance criteria

| Criterion (PLAN.md) | Result |
|---|---|
| Same seed ⇒ bit-identical world + trajectory; different seeds differ | ✅ `test_full_run_bit_reproducible`, `test_default_size_run_reproducible`, `test_different_seeds_diverge` |
| World rules unit-tested | ✅ 38 tests pass (movement/blocking, salt vs fresh water, eating, thirst death + cause logging, cooperative large-prey rule, seasonal growth, occlusion by mountains and dense forest, night view, fire spread/burn-out, cave warmth, reproduction + inheritance, capability switches, signals, building, disturbances, reward purity) |
| `pettingzoo.test.parallel_api_test` passes | ✅ `test_pettingzoo_parallel_api` |
| ≥1,000 agents on 256², ≥100k env steps/min (world dynamics) | ✅ 124k steps/min (2,072 steps/s). With observations: 72k steps/min — **below 100k** (see below) |
| Greedy and heuristic survive significantly longer than random | ⚠️ Partly. Heuristic: yes, by a wide margin. Greedy: **no** (≈ random) — explained below |
| Evidence figures | ✅ `fig1`–`fig8` in this folder |

## Throughput (random actions, 3,000 steps after 300 warm-up, `bench.json`)

| world | agents | world dynamics only | + observations for every agent |
|---|---|---|---|
| 256² | 256 | 2,998 steps/s (180k/min) | 1,956 steps/s (117k/min) |
| 256² | 1,000 | 2,072 steps/s (124k/min) | 1,198 steps/s (72k/min) |
| 256² | 2,000 | 1,572 steps/s (94k/min) | 565 steps/s (34k/min) |
| 512² | 1,000 | 1,853 steps/s (111k/min) | 774 steps/s (46k/min) |

Optimisations made during this stage: coarse 4×4 threat grid so prey skip empty
neighbourhoods; prey decisions computed in parallel against start-of-tick state, then resolved
sequentially (deterministic simultaneous update); vegetation advanced every 4 ticks with a 4× step;
reward drive in Numba. Fauna went from 1.0 ms to ~0.15 ms per tick. Observation building
(16×11×11 uint8 per agent, line-of-sight rays) is now the main cost; with neural policies in Stage 2,
network inference will dominate anyway. Observation throughput is reported honestly as below target.

## Baselines (5 seeds × 2 years × 256 agents, fixed population, `baselines.json`)

| policy | deaths / agent-year (95% bootstrap CI) | Kaplan-Meier survival at 1 year |
|---|---|---|
| random | 3.07 [2.66, 3.53] | 0.016 [0.007, 0.024] |
| greedy forager | 3.25 [2.62, 3.95] | 0.013 [0.004, 0.025] |
| fixed heuristic | **0.87 [0.69, 1.05]** | **0.426 [0.329, 0.516]** |

Heuristic vs random: Hedges' g on per-seed mortality is large and negative (see `baselines.json`).
**Negative result:** the greedy forager is no better than random. 64% of its deaths are thirst.
Fresh water covers ~5% of the map (the ocean is salt), so it is usually outside the 5-cell view,
and a memoryless agent that leaves water to look for food cannot find its way back. Random agents
mostly stay where they spawn, so those born near water survive. Nothing in the world was tuned to
produce this; it suggests memory or a world model should help most with finding water. That
prediction is logged in `logs/AGENT_LOG.md` before any training.

## What the world does (figures)
- `fig1_terrain_seeds.png` — four seeds: coastlines, mountain massifs ringed by caves, river
  networks from flow accumulation, lakes from filled depressions, forests along wet valleys, fruit trees.
- `fig2_layers.png` — elevation, moisture, fertility (patchy vs uniform ablation), hydrology.
- `fig3_timeline.png` — two years: seasonal and day/night temperature; rain, fog, drought, cold snap
  and famine episodes; plant and fruit cycles; boom-and-bust fauna; heuristic agents' body
  variables; snow cover; cumulative deaths (mortality accelerates during the year-1 famine and cold snap).
- `fig4_seasons.png` — biomass by season, winter snow, autumn wildfire scar.
- `fig5_baseline_survival.png` — Kaplan-Meier curves, mortality with CIs, causes of death.
- `fig6_paths.png` — tracked heuristic lives. `fig7_observation.png` — one agent's 16 channels in a
  forest (line of sight cut by dense forest). `fig8_disturbances.png` — triggered fire and flood.

## Known limitations / deviations (carried forward)
1. Fauna populations often sit at their configured caps or floors rather than being food-limited.
   Caps/floors are safety rails; making them non-binding is a tuning task for Stage 3/7 resilience work.
2. Observation throughput at 1,000 agents is 72k steps/min, below the 100k target (world dynamics exceed it).
3. Wildfires can burn ~10–15% of the land in a dry autumn. This is plausible but large; flagged.
4. Egocentric frames are world-aligned (north is always up); agents have no facing direction.
5. Juveniles have no dependency period yet (relevant to human comparison: long childhood). Candidate Stage 2 switch.
6. One body per cell. Agents cannot stack, so aggregation shows up as adjacency.
7. Affordances that make social behaviour possible are documented in `docs/WORLD.md`
   (cooperative large prey on by default; food drop/pickup; kin-similarity in the agent channel;
   huddle warmth off by default).
