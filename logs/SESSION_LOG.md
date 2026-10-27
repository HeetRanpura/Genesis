# GENESIS session log

Running record of development sessions: what was done, what was found, what was decided. Newest last.
(`logs/AGENT_LOG.md` is for how agents perform; this file is for engineering work.)

## 2026-10-26 — Session 3: live three.js viewer, survival-analysis bug fix

**Machine.** This session ran on a MacBook Pro with an M3 Pro (11 cores: 5 performance and 6 efficiency, 18 GB), not the
M4 Air the earlier benchmarks were measured on. Throughput numbers from different machines must not be compared.

**Setup.** This was a fresh clone with no `.venv`. `make setup` worked, and all 41 tests passed before any change.

**Answered for the user:** the default run has **256 agents** (2,048 slots, fixed population with respawn), plus
400 small prey, 48 large prey and 12 predators at the start, which are food-limited after that. No learning agents exist yet. Everything
on screen is a non-learning baseline (heuristic by default).

**Built: live viewer (Stage 5, first slice).** The user asked for three.js instead of the PixiJS named in PLAN.md.
- `viewer/server.py`: aiohttp WebSocket server, sim thread, binary protocol, command queue applied between ticks,
  and per-session run logging (`controls.jsonl`, chunked events, metrics, summary). An agent-log entry is added
  at session close.
- `viewer/static`: ES modules with no build step, and three 0.170.0 served from `viewer/node_modules`. It has heightmap terrain with
  plant, snow, fire, flood, drought and season shading; animated water whose surfaces rise with floods and vanish
  when streams dry; instanced wind-swayed trees, grass and flowers; agents with walk, rest, eat, drink and hunt animation,
  gene-coloured; three fauna species; predator eyes; need icons; day/night with moon and stars; rain, snow,
  fireflies and leaves; fire with flames, smoke and light; event bursts; bloom, tilt-shift and grading; a dashboard with
  7 live charts and deaths by cause; an inspector, follow mode, trail and perception ring; a minimap; a director;
  disturbance tools; restart; screenshots and recording.
- `tests/test_viewer.py`: smoke test covering server boot, init/fields/frame received, and a command applied and
  logged, with the run files written on close.
- `make viewer-setup`, `make viewer`; docs in `docs/VIEWER.md`.

**Bugs found and fixed during the build:**
- Port 8765 is used by another local service (`nap_whisper`), so the viewer defaults to 8787.
- Client registry held unhashable tuples, which made WebSocket handshakes fail. It's now a dict.
- Test sessions appended to `AGENT_LOG.md`. Those entries were reverted, `--no-agent-log` was added, and session years
  now exclude the spin-up.
- The event feed was invisible at high death rates, because items were replaced faster than their fade-in. Deaths are
  now batched every 1.5 s.

**Bug found in existing analysis (important):** `analysis/survival.py:lifetimes()` started the initial cohort's
lifetimes at tick 0. Since Stage 1.1, `env.tick` already includes the 1-year ecological spin-up, so every founding agent
was credited with an extra year. On seed 0 with the random policy over 1 year this gives 1.60 deaths/agent-yr against 3.20 when
fixed, and KM S(1 yr) 0.340 against 0.003. The Stage 1.1 baseline table (2.55 / 2.21 / 1.00) was affected. The fix adds
`start_tick=env.spinup_ticks` plus a regression test, and the baselines were re-run (see AGENT_LOG).

**Experiment (same session):** density sweep `experiments/density_sweep.py` (hypotheses in the docstring, written
before running). The heuristic dies 4.94 per agent-year at 1,000 agents against 1.6–1.7 at 256–512, almost all of it thirst, and does worse than
random at 1,000. The water-congestion mechanism is untested. Details are in AGENT_LOG.

**Evidence:** `reports/viewer/*.jpg` (day, overview, inspector, flood, wildfire, night, autumn, density overlay,
winter), captured with headless Chromium on Metal through scratch Playwright scripts (not in the repo). Render rate was 85–115 fps
with 256 agents and 75–90 fps with 1,000 agents at 1600×900.

**State at end:** 43 tests pass. Nothing is committed, and the user has not asked for a commit. The viewer server was left running on
http://localhost:8787 (seed 0, 256 heuristic agents, AGENT_LOG summary on close). Viewer test run directories from this session
remain in `runs/viewer/`; a safety check blocked a broad `rm`, so they were left for the user.

### Session 3, continued: flight recorder, teaching guide (2026-10-26, ~20:45–21:05)
- Committed the viewer work as `41c19c2` (not pushed).
- **Reproduction check:** with `capabilities.reproduction` on, heuristic populations crash 60–85% every winter and random agents go extinct (see AGENT_LOG).
- **User request: "record every agent's every activity at the atomic level."** Built `sim/recorder.py` and `analysis/trajectories.py`.
  - Per tick it stores every agent's full state, action, reward and death cause, plus the policy's internal mode and all lifetime counters.
  - It also stores world state per tick, every animal per tick, every event, every birth with genes, and every manual intervention.
  - Optionally it stores every observation.
  - Exact replay re-simulates and checks every tick, and can rebuild any agent's observation at any tick. A full sim year replays in 2.5 s.
  - **Cost:** the first version cut speed to 35% (structured rows assembled per tick, map reductions, slow zlib). After optimising it
    runs at 70%: 1,250 vs 1,780 ticks/s, 63 MB per sim year at 256 agents.
  - It is on by default in the viewer and the experiment scripts.
  - Engine change: manual floods now log `radius` and `days_given`, so floods whose duration was drawn at random replay exactly.
  - Tests: `tests/test_recorder.py`, and `test_viewer.py` now replays the viewer session. 45 tests pass.
- **Viewer shutdown bug:** SIGTERM left the server waiting up to 60 s for open WebSockets. The fix closes them in `on_shutdown` and sets
  `shutdown_timeout=3`. The 20:38 and 20:58 sessions both still closed cleanly, with summaries and AGENT_LOG entries.
- Wrote `TEACHER.md`: RL concepts explained through GENESIS, a glossary, and an FAQ built from the user's questions.

### Session 3, continued: PR #1, Stage 2 first slice (2026-10-26, ~21:05–23:10)
- **PR #1** opened on HeetRanpura/Genesis (branch `session3-viewer-recorder` on the fork advikdivekar/Genesis-advik).
  Author: Advik Divekar only, no co-authors.
- **Stage 2 learner** on branch `stage2-ppo`:
  - `agents/rppo.py` (recurrent PPO, shared network, per-agent GRU state reset at birth)
  - `experiments/train_rppo.py` (training worlds 1000+, full flight recording, disk guard)
  - `experiments/eval_policy.py` (same protocol as the baselines)
  - `agents/learned.py`, and viewer `--checkpoint` (the inspector shows the learner's value estimate and action probabilities)
  - `docs/PREREGISTRATION.md`, committed before any evaluation
- **Bugs caught:**
  - minibatch shuffle used numpy's unseeded global RNG. Fixed, and the run was restarted (the discarded run was moved to the scratchpad).
  - a test tolerance was too tight (9e-8 float rounding).
  - the queue loop's `pgrep -f` matched its own command line. It worked anyway, but my "fix" then killed the parent shell. The trainer survived and a watcher was added.
- **Performance:** about 4–7k agent-steps/s on MPS. The update dominates, especially the backward pass. CPU was 4× slower.
- **Results:** see AGENT_LOG and `reports/stage2/STAGE2_REPORT.md`. H1 supported, H2 not, H3 mixed.
- **Disk:** about 6 GB of recordings for Stage 2. 29 GB free.
- **Combined PR.** Heet had re-applied PR #1's commits on a rewritten main (PR #1 was closed unmerged; no shared history with our
  branches). He had also built his own Stage 2 (`agents/ppo.py` PPOTrainer). As the user asked, PR #1 and PR #2 were combined into one
  branch, `genesis-combined`, on top of his current main:
  - his learner is kept unchanged as the main one
  - ours is renamed `agents/rppo.py` (+ `train_rppo.py`, `eval_policy.py`, `tests/test_rppo.py`), so the reported results stay reproducible
  - the viewer lists both (`ppo:<run>`, `rppo:<run>`)
  - his `PolicyAgent` gained `.last` (value, probs, logp) for the recorder and inspector
  - 55 tests pass

## 2026-10-26 (Heet's machine): one learner implementation
- Folded `agents/rppo.py` into `agents/ppo.py` (PPOTrainer) at the user's request. The network in `agents/policy.py`
  is now rppo's (conv32→conv64→fc256 + fc64 → fc256 → GRU/MLP 256, orthogonal init). PPOConfig defaults are rppo's
  (128-tick rollouts, full BPTT, 4 epochs, raw rewards, 0.5-scaled value loss); truncated BPTT and return
  normalisation remain options. Action sampling is on the CPU with a seeded generator.
- One training script (`experiments/train_ppo.py`: rppo's world schedule, flight recording, disk guard, JSONL log,
  `checkpoints/last.pt`) and one evaluation script (`experiments/eval_policy.py`). Removed `agents/rppo.py`,
  `agents/learned.py`, `experiments/train_rppo.py`, `experiments/eval_agents.py`, `tests/test_rppo.py`; their tests
  are ported into `tests/test_ppo.py` (12 tests). Viewer: only `ppo:<run>`. Make targets: `make train`, `make eval`.
- Checkpoints carry `format: 2`; pre-unification PPOTrainer checkpoints are rejected (moved to `runs/stage2_legacy/`).
- Original rppo evaluation files moved to `reports/stage2/rppo_original/`. Replication with PPOTrainer
  (`gru_s0`, `ff_s0`, seed 0, 600 iterations) is running; results go into the Stage 2 report.
- 58 tests pass.

## 2026-10-27 (Heet's machine): replication, energy economy, behaviour, paper
- Merged PR #3 (Advik: articulated animated viewer, chase camera, frontend only).
- Retrained `gru_s0` and `ff_s0` with the unified PPOTrainer (600 iterations each, about 1.0 h and 1.7 h on the M4
  Air). Held-out evaluation replicates the rppo results (memoryless 1.50 [1.30, 1.75], GRU 1.67 [1.45, 1.92]).
- Added `--val-every` validation worlds to training; diet probes `heuristic:{grazer,fruit,hunter}` (the standard
  heuristic is unchanged bit-for-bit); `experiments/energy_economy.py`; `experiments/behaviour_analysis.py`;
  `experiments/paper_figures.py`.
- Wrote `paper/main.tex`, a NeurIPS-style preprint with TikZ diagrams, 12 pages, compiled with Tectonic.
- Correction: earlier, in conversation, night-time resting was described as learned, based on one agent seen in the
  viewer. The population analysis shows learners do **not** rest more at night.
- Kept compute sequential (one job at a time) to avoid heat-throttling the fanless laptop.

## Session 4 (Advik, 2026-10-27): pilot preprint, H6/H7 pipeline, diagrams
- Live demo of the viewer with every manual disturbance (logged in `runs/viewer/*/controls.jsonl`).
- `reports/stage2/pilot_preprint/`: NeurIPS-style preprint built from the pilot (rppo) runs. Adds exploratory analyses:
  half-year vs two-year horizon (the ranking flips), mortality and energy by season (a winter deficit), an energy budget
  from eat events, drought inflexibility. Also runs the preregistered H6/H7 analyses on the pilot checkpoints as a
  pipeline test: probes, value calibration, cooperative-hunt null, aggregation vs water/habitat nulls, and a social-blinding
  ablation. Nine SVG system diagrams (`diagrams/make_diagrams.py`). Ten citations verified online.
- Rebased onto upstream after the rppo fold: the pilot network is vendored in `pilot_net.py` (old checkpoint format),
  and pilot numbers are read from `reports/stage2/rppo_original/`.
- Gotcha found: `run_baselines.py` always overwrites `reports/stage1/baselines.json` (restored from git).

## 2026-10-27 (Heet's machine): PR #4 merged, pilot run-name collision fixed
- Reviewed and merged PR #4 (Advik's pilot preprint) by rebase. Tests: 59 pass.
- Found in review: the pilot scripts read `runs/stage2/{gru,ff}_s0` and `runs/stage2_eval/{gru,ff}_s0`, which on this
  machine hold the PPOTrainer replication. Checkpoint loads failed loudly, but recording-based analyses silently used
  replication data (`make_figures.py` rewrote `fig_seasons.png` before crashing; restored from git).
- Fix: `reports/stage2/pilot_preprint/pilot_runs.py` owns the pilot's paths (`runs/stage2_pilot/`, recordings under
  `runs/stage2_pilot/eval/`) and checks them: checkpoints must be rppo-format (no `format`, state dict under `net`), and a
  recording's deaths by cause must equal the pilot's published per-world results. Verified: all 5 replication
  recordings and both replication checkpoints are rejected when placed in the pilot tree; pilot-shaped files pass; with no
  pilot runs present `make_figures.py` stops before writing any run-dependent figure. Advik must move his pilot runs
  (commands in `HANDOFF_ADVIK.md`). The preprint itself and its committed figures are unchanged.

## 2026-10-27 (Heet's machine): one-command H2/H3 experiment for the M3 Pro
- Preregistered the H2/H3 protocol before any run (`docs/PREREGISTRATION.md`): seeds 1-3 × {GRU, memoryless} ×
  3,000 iterations; iteration 900 is the confirmatory H2'/H3'/H5 test, 3,000 the extension; decision rules fixed.
- `experiments/run_h2h3.py` (`make h2h3`, runbook `docs/H2H3_RUNBOOK.md`): runs the 6 trainings two at a time with
  tqdm bars, keeps checkpoints every 200 iterations and at 900, restarts crashed runs from checkpoint (3 retries),
  resumes after interruption, evaluates on test worlds 0-4 × 2 yr, and writes `reports/stage2/h2h3_results.{md,json}`.
- `train_ppo.py`: `--keep-every`, `--keep-at`; on resume, log rows after the checkpoint are dropped, and a finished
  run exits without touching AGENT_LOG. `eval_policy.py`: `--no-record`, auto device, `agent_years` in results.
  `tqdm` added to the `learn` extra.
- Verified on toy settings (32 agents, 4-70 iterations): full pipeline end to end; Ctrl-C at iteration 56 resumed
  from 50 with contiguous logs; a SIGKILLed run was restarted automatically and finished. Toy outputs deleted. The
  real run has not been started here (it belongs on the M3 Pro).
