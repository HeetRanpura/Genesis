# HANDOFF — where GENESIS stands and what to do next

Read this first, then `CLAUDE.md` (engine facts, gotchas, decisions) and `PLAN.md` (stages and acceptance
criteria). `logs/AGENT_LOG.md` records how agents are doing. `logs/SESSION_LOG.md` records engineering sessions.

Repo: https://github.com/HeetRanpura/Genesis (branch `main`, direct commits, no PRs).
**Machine (since session 3): MacBook Pro M3 Pro, 11 cores, 18 GB, no CUDA.** Stages 1–1.1 were benchmarked on an
M4 Air, so don't compare throughput across machines. Python 3.11 venv in `.venv` (`make setup`, which now also installs the viewer).

**Git:** session-3 work is committed locally (`41c19c2` viewer etc., then the recorder/TEACHER commit). **Not pushed.**

## Done
- **Stage 1: world engine** (`ce1e40e`) and **Stage 1.1: food-limited fauna, faster observations** (`3c1d52d`).
  See `reports/stage1/STAGE1_REPORT.md`.
- **Session 3 (2026-10-26):**
  1. **Live 3-D viewer (Stage 5, first slice), built with three.js at the user's request.**
     - Run `make viewer-setup` once, then `make viewer` and open http://localhost:8787. Variables: `SEED=`, `AGENTS=`, `POLICY=`, `PORT=`.
     - Features: real-time world, agents, animals, weather, seasons, fire and floods, a dashboard with 7 live charts plus deaths by cause,
       an agent inspector, follow mode, a minimap, a director, disturbance tools, restarts, screenshots and recording.
     - Each session logs to `runs/viewer/…` (including `controls.jsonl`), and adds an AGENT_LOG entry at close unless run with `--no-agent-log`.
     - Spec and an honest list of what is real and what is decoration: `docs/VIEWER.md`. Evidence: `reports/viewer/`. Smoke test: `tests/test_viewer.py`.
     - Measured: 85–115 fps with 256 agents and 75–90 fps with 1,000 agents (1600×900, headless Chromium/Metal). At MAX speed the sim runs
       about 1,450 ticks/s (256 agents) or 840 ticks/s (1,000 agents) while streaming.
  2. **Bug fix: survival analysis.** `lifetimes()` measured founders from tick 0, but `env.tick` includes the 1-year
     spin-up, so mortality was under-stated by about a third. Fixed with `start_tick=env.spinup_ticks`, plus a regression test.
     The baselines were re-run. Corrected deaths/agent-yr: **random 3.83, greedy 3.31, heuristic 1.50** (reported earlier as 2.55 / 2.21 / 1.00).
     Corrected KM survival at 1 yr: 0.001 / 0.006 / 0.214. The erratum is in the report, `fig5` is regenerated, and the AGENT_LOG has a correction entry.
  3. **Experiment: density sweep** (`experiments/density_sweep.py`, `reports/stage1/density_sweep.json`, 3 seeds × 1 yr).
     - Heuristic mortality is flat from 256 to 512 agents, then jumps to **4.94 at 1,000**, where 96% of deaths are thirst.
       **At 1,000 agents the heuristic is worse than random** (3.85). Greedy also worsens with density, while random stays flat.
     - This is consistent with congestion at water (one body per cell), but the mechanism is untested.
       Compare learners with baselines at the same population size.
  4. **Flight recorder** (`sim/recorder.py`, `analysis/trajectories.py`): every agent's full state, action, reward and death at
     every tick, plus animals, world, events, births with genes, and interventions. On by default in the viewer and experiments.
     It costs about 63 MB per sim year and runs at about 70% of full speed. Exact replay is verified bit-for-bit and can rebuild any agent's observation.
  5. `TEACHER.md`: RL study guide and FAQ for the team.
  6. Reproduction check: heuristic populations crash every winter, and random agents go extinct (AGENT_LOG).
  7. 45 tests pass (`make test`).

## Population (the user asked)
The default config starts **256 agents** in 2,048 slots. The population is fixed: a dead agent's slot is refilled at once.
The world also starts with 400 small prey, 48 large prey and 12 predators, which then vary with food.
**No agent learns yet.** Every agent is a non-learning baseline (heuristic by default), and the viewer labels this.

## Open issues / follow-ups
1. **Water congestion (new, measured):** test the mechanism by logging occupancy and waits at drinkable cells, and rerun the
   sweep with more water. Water access may be the natural commons for later Ostrom-style tests.
2. **512² regression** (from Stage 1.1): 77k steps/min for dynamics alone. The daily `refresh_img` rebuilds the whole image when
   only the water channel changes.
3. **Predators with agents present** sit at their floor about 30% of the time.
4. **Viewer, still open:**
   - replay scrubbing and bookmarks, side-by-side seeds, mid-run resource changes, spawning and removing agents
   - audio, 4K offline export
   - social-network and signal overlays, imagined futures (these need learners)
   - 1440p frame rate measured in a real browser
5. **Viewer, visual nits:**
   - rivers on steep slopes look stepped, because each cell has its own water level
   - flood edges are blocky, because they follow the cell grid
   - trees are near-black silhouettes at night
   - the director is basic (it cuts to events and follows a struggling agent)
6. Test sessions from this session's screenshots are still in `runs/viewer/` (git-ignored). Delete them if you don't want them.

## Next stage (needs the user's go-ahead): Stage 2, agent minds
See PLAN.md. Suggested first slice:
- PyTorch (CPU/MPS) recurrent PPO with a parameter-shared GRU over `grid[N,11,11,16]` uint8, plus `self`, `neighbors` and `heard`.
- Train online against `GenesisEnv`, and reset the recurrent state where `info["born"]`.
- Beat all three baselines on survival, with CIs over at least 5 seeds **at 256 agents**, before adding world models.
- To make learners drivable from the viewer, add a policy entry in `viewer/server.py: POLICIES` and `make_baseline`.
- Watch out: torch bundles its own libomp, and Numba here uses the workqueue layer.

## Gotchas that cost time (details in CLAUDE.md)
- `env.tick` includes the spin-up. Subtract `env.spinup_ticks` for anything measured from the start of a run.
- Never build a Numba twin with `njit(cache=True)(f.py_func)`.
- Every world mutation must also patch `env.img`. Call `env.refresh_img()` after editing world arrays directly.
- Port 8765 is used by another local service on this machine, so the viewer uses 8787.
- Viewer test sessions: always pass `--no-agent-log`, otherwise they append to AGENT_LOG.
- Benchmarks drift with heat. Compare only interleaved runs and report medians.
