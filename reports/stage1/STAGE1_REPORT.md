# Stage 1 report — world engine

Date: 2026-10-26 · Machine: Apple M4 (10 cores, 16 GB), Python 3.11, Numba 0.68, NumPy 2.4.
Reproduce everything: `make test && make baselines && make evidence && make bench`.

## Erratum (2026-10-26, session 3): baseline survival numbers below were biased; corrected

`analysis/survival.lifetimes()` measured the initial cohort's lifetimes from tick 0, but since Stage 1.1 `env.tick`
already includes the 1-year ecological spin-up, so every founder was credited with a year it never lived. Deaths
were counted correctly; exposure was inflated, so mortality was **under-stated by ~1/3** and 1-year survival
**over-stated**. Fixed (`start_tick=env.spinup_ticks`, regression test `test_lifetimes_start_at_run_start_not_tick_zero`)
and `make baselines` re-run (same seeds, identical deaths by cause). The original Stage 1 numbers further down had no
spin-up and are unaffected.

| policy | deaths / agent-year, as reported | **corrected** (95% CI) | KM S(1 yr), as reported | **corrected** (95% CI) |
|---|---|---|---|---|
| random | 2.55 | **3.83 [3.53, 4.19]** | 0.134 | **0.001 [0.000, 0.002]** |
| greedy | 2.21 | **3.31 [2.92, 3.77]** | 0.163 | **0.006 [0.000, 0.017]** |
| heuristic | 1.00 | **1.50 [1.04, 2.11]** | 0.485 | **0.214 [0.102, 0.331]** |

Consequences: the conclusions' direction holds (heuristic ≪ greedy < random; Hedges' g vs random −3.8 and −1.0),
but the Stage 1.1 retune hurt the heuristic far more than reported (0.87 → **1.50**, not 1.00), and almost
no random/greedy agent lives a full year. `fig5_baseline_survival.png` is regenerated with corrected data.

### Addendum: baseline population-density sweep (`density_sweep.json`, 3 seeds × 1 yr per cell)
| policy | 256 agents | 512 | 1,000 |
|---|---|---|---|
| random | 3.61 [3.12, 4.25] | 3.25 [3.13, 3.42] | 3.85 [3.52, 4.43] |
| greedy | 3.61 [2.78, 4.71] | 3.91 [3.48, 4.17] | 5.50 [4.97, 6.21] |
| heuristic | 1.72 [1.19, 2.46] | 1.64 [1.32, 2.08] | **4.94 [4.70, 5.39]** |

These are deaths per agent-year. Mortality of the water-seeking baselines rises with density, and at 1,000 agents almost
all of it is thirst (96% for the heuristic). Random stays flat. **At 1,000 agents the heuristic is worse than random.**
This is consistent with congestion at scarce fresh-water cells, but that mechanism has not been tested yet. Small prey
are hunted down to the floor at 1,000 agents. Learners must be compared with baselines at the same population size.

## Stage 1.1 update (user-requested tuning) — read this first

Two requests: (1) make fauna food-limited instead of cap-limited, (2) push observation throughput past
100k steps/min. Sections further below are the original Stage 1 results, kept for history.

### 1. Fauna is now food-limited
Fauna parameters moved to `configs/default.yaml: fauna`. Regulation now comes from food: energy stores,
metabolism with winter torpor (×0.2 below 5 °C), grazing only when not sated, births only in
spring–autumn with a full store (shared with the offspring), grazing-tolerant plant regrowth,
and a browsing niche for large prey (without it, small prey competitively excluded them). Runs start after
a 1-year ecological spin-up so the lush initial vegetation's boom-and-crash isn't part of the run.
Caps and immigration floors remain only as safety rails. `experiments/fauna_diagnostics.py`, 5 seeds × 3 years:

| | before | after, 256 heuristic agents | after, no agents |
|---|---|---|---|
| small prey: time at cap | 69% | **0%** | **0%** |
| large prey: time at cap | 5% | **0%** | **0%** |
| predators: time at cap | 55% | **0%** | **0%** |
| small prey: time at floor / immigrant share of recruits | 9% / 7% | 7% / 2% | 0.3% / 0% |
| large prey: time at floor / immigrant share | 14% / 11% | 0.6% / 0% | 13% / 6% |
| predators: time at floor / immigrant share | 0% / 0% | **31% / 21%** | 2% / 0% |

Deaths are now starvation, predation and agents' hunting. Figures: `fig9_fauna_after.png`,
`fig9_fauna_after_no_agents.png`. **Weak spot:** with 256 heuristic agents present, predators sit
at their floor about a third of the time. The agents cause ~half of all small-prey deaths and
out-compete predators. Reported as found, not tuned away.

### 2. Throughput (`bench.json`, median of 5 whole-day windows; random actions)
| world | agents | dynamics only | + observations |
|---|---|---|---|
| 256² | 256 | 3,522 steps/s (211k/min) | 2,437 steps/s (146k/min) |
| 256² | 1,000 | 2,860 steps/s (172k/min) | **1,867 steps/s (112k/min)** |
| 256² | 2,000 | 2,910 steps/s (175k/min) | 1,484 steps/s (89k/min) |
| 512² | 1,000 | 1,284 steps/s (77k/min) | 991 steps/s (59k/min) |

Fair before/after: interleaved runs of the Stage 1 commit and the current tree on the same machine in the
same thermal state (1,000 agents, 256², three rounds):

| | Stage 1 | Stage 1.1 |
|---|---|---|
| with observations | 1,196–1,348 steps/s (72–81k/min) | **1,662–1,728 steps/s (100–104k/min)** |
| dynamics only | 1,739–2,229 steps/s | 2,560–3,885 steps/s |

How:
- Observations are masked copies from an observation image (`env.img`) patched in place at every
  world mutation (a test compares it with a full rebuild after long random runs).
- Line of sight is precomputed per cell, since terrain is static.
- One Numba call per tick (`world_tick`).
- Small workloads use sequential twins of parallel kernels. Numba's macOS threading layer costs
  ~60 µs per parallel launch; outputs are identical, and a test checks this.

**Caveats:**
- This is a fanless MacBook Air: sustained runs heat-throttle by 15–30%, so the A/B table is the
  conservative figure.
- **512² got slower than Stage 1.** Vegetation, the daily image refresh and larger fauna all scale
  with area. This is a follow-up item; the 1,000-agent target is on the default 256² world.
- An OpenMP threading layer was tried and rejected (slower and erratic here).

### 3. Effect on the baselines (5 seeds × 2 years, 256 agents) — **corrected values, see Erratum above**
| policy | deaths / agent-year (95% CI) | KM survival at 1 yr |
|---|---|---|
| random | 3.83 [3.53, 4.19] (reported: 2.55) | 0.001 (reported: 0.134) |
| greedy | 3.31 [2.92, 3.77] (reported: 2.21) | 0.006 (reported: 0.163) |
| heuristic | 1.50 [1.04, 2.11] (reported: 1.00) | 0.214 (reported: 0.485) |

- Heuristic vs random: Hedges' g = −3.8.
- Greedy now edges out random (g = −1.0).
- **The heuristic got worse (0.87 → 1.50 corrected; first reported as 1.00) and its thirst deaths tripled.** An ablation reverting one
  change at a time pins this on the fauna retune. Prey are no longer artificially abundant (before:
  capped at 1,600 with a 200-animal immigration floor), so agents eat fewer animals and more plants.
  On seed 0: small-prey kills 1,871 → 551, meat meals 3,629 → 980, plant bites 17,382 → 31,674.
  They spend more time foraging, and the heuristic's "food first when energy ≤ hydration" rule leaves
  them away from water.
- In seed 0, agents hunted small prey down to 27 animals by the end of year 1. An overexploitation
  dynamic exists before any learning.

### Bug fixed along the way
Events from manual triggers made between steps (fire/flood/famine buttons) were wiped by the next
step's event-counter reset. Now fixed and covered by the image-invariant test.

---

# Original Stage 1 results (superseded where Stage 1.1 says so)

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
