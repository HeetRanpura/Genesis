# HANDOFF — where GENESIS stands and what to do next

Read this, then `CLAUDE.md` (engine facts, gotchas, decisions), then `PLAN.md` (stages + acceptance
criteria). `logs/AGENT_LOG.md` is the running record of how agents are doing.

Repo: https://github.com/HeetRanpura/Genesis (branch `main`, direct commits, no PRs).
Machine: MacBook Air M4 (fanless, 10 cores, 16 GB, no CUDA). Python 3.11 venv in `.venv` (`make setup`).

## Done
- **Stage 1 — world engine** (commit `ce1e40e`): procedural 256² world, climate/weather/disturbances,
  plants/fruit/fauna, agents with occluded egocentric perception, homeostatic survival-only reward,
  capability switches (off by default), deterministic hash RNG, PettingZoo wrapper, run logger,
  three baselines, evidence figures, `reports/stage1/STAGE1_REPORT.md`.
- **Stage 1.1 — user-requested tuning** (commit `3c1d52d` + later):
  - Fauna food-limited: parameters in `configs/default.yaml: fauna`; seasonal breeding, torpor,
    grazing-tolerant regrowth, large-prey browsing niche, 1-year spin-up. Diagnostics
    (`experiments/fauna_diagnostics.py`, 5 seeds × 3 years): caps bind 0% of the time for all species
    (before: small prey 69%, predators 55%). Floors: small prey 7%, large prey 0.6%, predators 31% with
    256 heuristic agents (agents out-compete predators for small prey); without agents floors are
    0.3% / 13% / 2%.
  - Throughput: observations are copies from an incrementally maintained obs image
    (`env.img`), line of sight is precomputed (`env.vismask`), and the whole tick is one Numba call
    (`kernels.world_tick`). Interleaved A/B vs Stage 1 on the heat-throttled laptop, 1,000 agents, 256²:
    with observations 1,662–1,728 steps/s (≈100–104k steps/min) vs 1,196–1,277 (≈72–77k);
    dynamics only 2,560–3,870 vs 1,740–2,230 steps/s.
  - 41 tests pass (`make test`), including bit-reproducibility, reward purity, obs-image invariant,
    sequential/parallel twin equivalence, PettingZoo API.

## In progress when this was written
1. **Heuristic baseline mortality shifted** under the new ecology: 1.00 deaths/agent-yr [0.70, 1.41]
   (was 0.87), thirst deaths 705 → 2,460 (5 seeds × 2 yrs). Verified the fused-tick observation equals
   `observe()` (not an obs bug). Deaths cluster in spring and winter of both years (runs now start
   right after the spin-up winter, in a lean early spring, instead of on lush tick-0 vegetation).
   Next: confirm by checking hydration/energy traces in early spring and whether the heuristic's
   eat-first hysteresis walks agents away from water when food is scarce. Report, don't tune the
   heuristic to look good.
2. **Final benchmark table**: run `make bench` after the machine has cooled (burst numbers) and keep the
   interleaved A/B numbers above as the sustained/throttled figures. Update
   `reports/stage1/STAGE1_REPORT.md` (add a Stage 1.1 section) and `logs/AGENT_LOG.md`.

## Next stage (needs the user's go-ahead): Stage 2 — agent minds
See PLAN.md. Suggested first slice: PyTorch (CPU/MPS) recurrent PPO with a parameter-shared GRU
policy over the channel-last obs (`grid[N, 11, 11, 16]` uint8 + `self` + `neighbors` + `heard`),
trained online against `GenesisEnv` (slot API: reset recurrent state where `info["born"]`).
Beat all three baselines on survival with CIs over ≥5 seeds before adding world models.
Watch out: torch bundles its own libomp; Numba here uses the workqueue layer (no conflict expected).

## Gotchas that cost time (details in CLAUDE.md)
- Never build a Numba twin with `njit(cache=True)(f.py_func)` — it overwrites f's cache entry.
- Every world mutation must also patch `env.img` (test catches drift); call `env.refresh_img()` after
  editing world arrays directly.
- Benchmarks on this laptop drift ±20% with heat; compare only interleaved runs, report medians.
