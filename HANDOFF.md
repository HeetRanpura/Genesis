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
    (`kernels.world_tick`). `make bench` (median of 5 whole-day windows): 1,000 agents on 256² with
    observations = 1,867 steps/s (112k steps/min); dynamics only 172k/min. Interleaved A/B vs Stage 1 in
    the same thermal state: with obs 1,662–1,728 vs 1,196–1,348 steps/s.
  - Baselines rerun (5 seeds × 2 yrs): random 2.55, greedy 2.21, heuristic 1.00 deaths/agent-yr.
    The heuristic's thirst deaths tripled. An ablation shows the cause is food-limited prey (no more
    free meat). This is explained in the report and agent log, not a bug.
  - 41 tests pass (`make test`): bit-reproducibility, reward purity, obs-image invariant,
    sequential/parallel twin equivalence, PettingZoo API, and the world rules.
  - Full write-up: `reports/stage1/STAGE1_REPORT.md` (Stage 1.1 section at the top).

## Open issues / follow-ups (not started)
1. **512² regression**: 77k steps/min dynamics (Stage 1: 111k). Suspects: vegetation pass + img writes
   over 262k cells, daily `refresh_img(terrain_changed=False)` (full image rebuild; only the water
   channel actually changes — rebuild just that), larger fauna. Profile with
   `experiments/bench_sim.py`-style windows at 512².
2. **Predators with agents present** sit at their floor ~31% of the time (agents out-compete them for
   small prey). Options if it matters for experiments: predators scavenging carcasses, prey refuges.
3. 2,000 agents with observations: 89k steps/min (no target set, but noted).

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
