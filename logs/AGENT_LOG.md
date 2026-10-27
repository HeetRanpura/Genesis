# GENESIS agent log

Running record of how agents are doing, newest last. Entries are appended by experiment scripts; numbers are measured, not estimated.

## 2026-10-26 17:17 — Stage 1 baseline survival (SUPERSEDED: engine before simultaneous prey-update change; kept for history)

Baselines, 5 seeds x 2.0 years, 256 agents, fixed population (respawn).

| policy | deaths / agent-year (95% CI) | Kaplan-Meier survival at 1 yr (95% CI) |
|---|---|---|
| random | 3.05 [2.57, 3.55] | 0.016 [0.006, 0.029] |
| greedy | 3.22 [2.54, 3.94] | 0.010 [0.003, 0.016] |
| heuristic | 0.87 [0.69, 1.05] | 0.427 [0.326, 0.529] |

Deaths by cause (summed over seeds): **random** {'starvation': 3255, 'thirst': 3150, 'cold': 807, 'fire': 209, 'predator': 384}; **greedy** {'starvation': 1493, 'thirst': 5092, 'cold': 1104, 'fire': 178, 'predator': 386}; **heuristic** {'starvation': 115, 'thirst': 705, 'cold': 382, 'fire': 82, 'predator': 899, 'old_age': 50}

## 2026-10-26 17:28 — Stage 1 baseline survival

Baselines, 5 seeds x 2.0 years, 256 agents, fixed population (respawn).

| policy | deaths / agent-year (95% CI) | Kaplan-Meier survival at 1 yr (95% CI) |
|---|---|---|
| random | 3.07 [2.66, 3.53] | 0.016 [0.007, 0.024] |
| greedy | 3.25 [2.62, 3.95] | 0.013 [0.004, 0.025] |
| heuristic | 0.87 [0.69, 1.05] | 0.426 [0.329, 0.516] |

Deaths by cause (summed over seeds): **random** {'starvation': 3243, 'thirst': 3221, 'cold': 825, 'fire': 237, 'predator': 327}; **greedy** {'starvation': 1490, 'thirst': 5152, 'cold': 1130, 'fire': 190, 'predator': 372}; **heuristic** {'starvation': 112, 'thirst': 665, 'cold': 413, 'fire': 82, 'predator': 894, 'old_age': 56}

## 2026-10-26 17:35 — Stage 1 summary (written by hand)

- No learning agents exist yet; only the three non-learning baselines have lived in the world.
- Hand-written survival rules (heuristic) cut mortality ~3.5x vs random (0.87 vs 3.07 deaths per agent-year).
  The world is survivable with competence, but not easy: even the heuristic loses ~57% of agents within a year.
- Greedy forager is no better than random: it dies of thirst (64% of its deaths) because fresh water is usually
  outside its 5-cell view and it has no memory. Prediction for Stage 2: memory/world-model agents should
  gain most on water-finding. Recorded here *before* training so it can be checked.
- Heuristic agents die mostly from predators (48%) and thirst (36%); cold deaths cluster in cold snaps.
- No social behaviour is expected or observed from baselines (none of them has any social rule).

## 2026-10-26 19:15 — Stage 1 baseline survival

Baselines, 5 seeds x 2.0 years, 256 agents, fixed population (respawn).

| policy | deaths / agent-year (95% CI) | Kaplan-Meier survival at 1 yr (95% CI) |
|---|---|---|
| random | 2.55 [2.36, 2.79] | 0.134 [0.123, 0.143] |
| greedy | 2.21 [1.95, 2.51] | 0.163 [0.135, 0.195] |
| heuristic | 1.00 [0.70, 1.41] | 0.485 [0.329, 0.633] |

Deaths by cause (summed over seeds): **random** {'starvation': 5128, 'thirst': 3359, 'cold': 910, 'fire': 271, 'predator': 142}; **greedy** {'starvation': 1699, 'thirst': 4685, 'cold': 1145, 'fire': 353, 'predator': 608}; **heuristic** {'starvation': 347, 'thirst': 2460, 'cold': 442, 'fire': 276, 'predator': 289, 'old_age': 17}

## 2026-10-26 — Stage 1.1: baselines under food-limited fauna (written by hand)

Engine changes: fauna food-limited (caps never bind), 1-year ecological spin-up, faster observations.
Rerun of `experiments/run_baselines.py` (5 seeds × 2 years, 256 agents):

| policy | deaths / agent-year (95% CI) | KM survival at 1 yr |
|---|---|---|
| random | 2.55 [2.36, 2.79] | 0.134 |
| greedy | 2.21 [1.95, 2.51] | 0.163 |
| heuristic | 1.00 [0.70, 1.41] | 0.485 |

- Heuristic still far ahead of random (Hedges' g = −3.8 on per-seed mortality); greedy is now slightly
  better than random (g = −1.0), no longer worse.
- Heuristic mortality rose (0.87 → 1.00) and its thirst deaths tripled. Ablation (3 seeds × 1.5 years,
  one change reverted at a time) pins it on the fauna retune, not spin-up/vegetation: with the old
  fauna, thirst deaths 383 vs 1,160 now. Mechanism, measured on seed 0: small-prey kills by agents fell
  1,871 → 551 and meat meals 3,629 → 980; agents switched to plants (31,674 vs 17,382 bites), spent 38%
  vs 22% of ticks in eat-mode, and the heuristic's "food first when energy ≤ hydration" rule leaves them
  away from water. Before, capped prey plus an immigration floor of 200 were effectively free food.
- Agents are the dominant predator of small prey: 256 heuristic agents hunted small prey down to 27
  animals by the end of year 1 (seed 0); agents cause ~half of all small-prey deaths. An overexploitation
  dynamic exists in the world before any learning — worth tracking once learners arrive (commons tests).
- Deaths cluster in spring (post-winter lean season) and winter for heuristic agents in both years.

## 2026-10-26 20:27 — Stage 1 baseline survival

Baselines, 5 seeds x 2.0 years, 256 agents, fixed population (respawn).

| policy | deaths / agent-year (95% CI) | Kaplan-Meier survival at 1 yr (95% CI) |
|---|---|---|
| random | 3.83 [3.53, 4.19] | 0.001 [0.000, 0.002] |
| greedy | 3.31 [2.92, 3.77] | 0.006 [0.000, 0.017] |
| heuristic | 1.50 [1.04, 2.11] | 0.214 [0.102, 0.331] |

Deaths by cause (summed over seeds): **random** {'starvation': 5128, 'thirst': 3359, 'cold': 910, 'fire': 271, 'predator': 142}; **greedy** {'starvation': 1699, 'thirst': 4685, 'cold': 1145, 'fire': 353, 'predator': 608}; **heuristic** {'starvation': 347, 'thirst': 2460, 'cold': 442, 'fire': 276, 'predator': 289, 'old_age': 17}

## 2026-10-26 20:28 — Baseline population-density sweep

Density sweep, 3 seeds x 1.0 yr per cell, 256x256 world, fixed population (respawn). Mean [95% bootstrap CI] over seeds. Hypotheses H1-H3 are in `experiments/density_sweep.py` (written before running).

| policy | agents | deaths / agent-yr | thirst share of deaths | small prey (mean) | large prey (mean) | predators (mean) |
|---|---|---|---|---|---|---|
| random | 256 | 3.61 [3.12, 4.25] | 0.38 [0.31, 0.42] | 557 [494, 675] | 34 [12, 74] | 4.4 [2.1, 6.3] |
| random | 512 | 3.25 [3.13, 3.42] | 0.33 [0.28, 0.40] | 658 [516, 877] | 38 [19, 56] | 4.0 [2.2, 5.0] |
| random | 1000 | 3.85 [3.52, 4.43] | 0.33 [0.25, 0.41] | 223 [80, 344] | 45 [40, 49] | 3.2 [2.2, 4.1] |
| greedy | 256 | 3.61 [2.78, 4.71] | 0.49 [0.33, 0.72] | 318 [219, 481] | 32 [10, 60] | 23.0 [10.0, 32.5] |
| greedy | 512 | 3.91 [3.48, 4.17] | 0.71 [0.69, 0.73] | 479 [298, 606] | 36 [19, 57] | 28.5 [11.6, 49.6] |
| greedy | 1000 | 5.50 [4.97, 6.21] | 0.78 [0.73, 0.81] | 374 [240, 572] | 30 [11, 39] | 22.1 [3.1, 32.9] |
| heuristic | 256 | 1.72 [1.19, 2.46] | 0.56 [0.36, 0.79] | 143 [81, 176] | 16 [12, 24] | 13.6 [2.3, 27.8] |
| heuristic | 512 | 1.64 [1.32, 2.08] | 0.86 [0.81, 0.91] | 92 [67, 132] | 97 [17, 169] | 9.1 [2.4, 13.9] |
| heuristic | 1000 | 4.94 [4.70, 5.39] | 0.96 [0.92, 0.99] | 12 [10, 13] | 46 [36, 58] | 17.7 [2.2, 28.7] |

## 2026-10-26 — Session 3: survival-analysis correction + density sweep (written by hand)

**Correction (supersedes the Stage 1.1 table above, which is kept for history).** `lifetimes()` started the
initial cohort at tick 0 while `env.tick` already includes the 1-year spin-up, so founders were credited with a
year they never lived. Deaths by cause were right; exposure was inflated. Re-run with the fix (entry above,
"Stage 1 baseline survival", 2026-10-26 ~20:30): **random 3.83 [3.53, 4.19], greedy 3.31 [2.92, 3.77], heuristic
1.50 [1.04, 2.11] deaths/agent-yr; KM S(1 yr) 0.001 / 0.006 / 0.214.** Previously reported: 2.55 / 2.21 / 1.00 and
0.134 / 0.163 / 0.485. Ranking and effect sizes unchanged (g vs random: heuristic −3.8, greedy −1.0). The Stage 1.1
retune cost the heuristic more than reported: 0.87 → 1.50 deaths/agent-yr. Almost no random/greedy agent lives a year.

**Density sweep** (`experiments/density_sweep.py`, table in the entry above; hypotheses written in the script before
running, but prompted by one uncontrolled viewer session — so not a blind prediction):
- H1 (heuristic mortality rises with N): **partly.** 256 → 512 flat (1.72 vs 1.64, CIs overlap); 1000 jumps to
  **4.94 [4.70, 5.39]** — a threshold between 512 and 1000 agents on a 256² world, not a smooth rise.
- H2 (the rise is thirst): **supported.** Thirst share of heuristic deaths 0.56 → 0.86 → 0.96.
- H3 (fewer small prey, more large prey): **small prey yes** (143 → 92 → 12, i.e. hunted to the floor at 1000);
  **large prey mixed** (16 → 97 → 46, wide CIs) — not supported as a monotone effect.
- Unexpected: **at 1,000 agents the heuristic is worse than random** (4.94 vs 3.85). Greedy also worsens with N
  (3.61 → 5.50, thirst share 0.49 → 0.78) while random is flat (3.61 / 3.25 / 3.85). Agents that *seek* water
  degrade with density; agents that wander don't. This is **consistent with** congestion at scarce fresh-water
  cells (one body per cell, few drinkable shore cells) but the mechanism was **not tested** (next: measure
  occupancy/queueing of drinkable cells, and rerun with more water).
- Implication for Stage 2/7: compare learners vs baselines at the **same N**; at N ≥ 1000 the heuristic is not a
  strong baseline. Water access is a natural commons — a candidate arena for Ostrom-style tests once learners exist.
- Scale: 3 seeds × 1 year per cell; CIs from 3 seeds are rough.

## 2026-10-26 — Reproduction switch on, baselines (quick check, written by hand)

`capabilities.reproduction: true` (asexual births when mature, energy ≥ 0.8, hydration ≥ 0.6, cooldown 0.25 yr;
no respawn), 256 founders, 3 seeds × 2 years, no other changes. Population every quarter-year:
- heuristic: s0 256→146→210→250→137→40→59→71 (end 68); s1 …→277→36→…→68 (end 71); s2 …→117→44→…→85 (end 39).
  230–342 births per run, max generation 3. Pattern: growth through spring–autumn, **crash every winter**
  (−60 to −85%), partial recovery next season. Persists 2 years but at ~15–30% of the founding size.
- random: **extinct in all 3 seeds** (0.8–1.6 years); almost never well-fed enough to breed (33–75 births).
Single quick check, not a preregistered experiment. Births are physiological (automatic when well-fed), not an action.

## 2026-10-26 20:58 — Live viewer session (heuristic, seed 0)

Viewer run `runs/viewer/20261026_203827_s0_heuristic`: 18636 ticks (4.85 sim years), 256 agents, policy **heuristic** (non-learning baseline).
- deaths/agent-year 1.51; deaths by cause {'starvation': 192, 'thirst': 1050, 'cold': 290, 'fire': 55, 'predator': 288, 'old_age': 3}; population at end 256.
- manual interventions: 0 (none). Sessions with interventions are not comparable to `run_baselines.py` numbers.
- eating food another agent killed/dropped: 377 meals (a co-feeding count, not evidence of intentional sharing).

## 2026-10-26 21:01 — Live viewer session (heuristic, seed 0)

Viewer run `runs/viewer/20261026_205803_s0_heuristic`: 1814 ticks (0.47 sim years), 256 agents, policy **heuristic** (non-learning baseline).
- deaths/agent-year 1.11; deaths by cause {'thirst': 132, 'predator': 2}; population at end 256.
- manual interventions: 0 (none). Sessions with interventions are not comparable to `run_baselines.py` numbers.
- eating food another agent killed/dropped: 24 meals (a co-feeding count, not evidence of intentional sharing).

## 2026-10-26 21:12 — Stage 2 training: smoke_mps

PPO (GRU), 256 agents, 3 iterations × 128 ticks (98,304 agent-steps), training worlds 1000+ (eval worlds 0-4 held out).
- deaths/agent-yr during training: first 1 iters 0.00 → last 1 iters 0.00 (training-time rate with exploration; not an evaluation).
- last-iteration causes: {}; entropy 2.39.
- logs: `runs/stage2/smoke_mps/train.jsonl`, checkpoints in `runs/stage2/smoke_mps/checkpoints/`.

## 2026-10-26 21:14 — Stage 2 training: smoke_cpu

PPO (GRU), 256 agents, 3 iterations × 128 ticks (98,304 agent-steps), training worlds 1000+ (eval worlds 0-4 held out).
- deaths/agent-yr during training: first 1 iters 0.00 → last 1 iters 0.00 (training-time rate with exploration; not an evaluation).
- last-iteration causes: {}; entropy 2.39.
- logs: `runs/stage2/smoke_cpu/train.jsonl`, checkpoints in `runs/stage2/smoke_cpu/checkpoints/`.

## 2026-10-26 21:14 — Stage 2 training: smoke_mps

PPO (GRU), 256 agents, 3 iterations × 128 ticks (98,304 agent-steps), training worlds 1000+ (eval worlds 0-4 held out).
- deaths/agent-yr during training: first 1 iters 0.00 → last 1 iters 0.00 (training-time rate with exploration; not an evaluation).
- last-iteration causes: {}; entropy 2.39.
- logs: `runs/stage2/smoke_mps/train.jsonl`, checkpoints in `runs/stage2/smoke_mps/checkpoints/`.

## 2026-10-26 22:10 — Stage 2 training: gru_s0

PPO (GRU), 256 agents, 600 iterations × 128 ticks (19,660,800 agent-steps), training worlds 1000+ (eval worlds 0-4 held out).
- deaths/agent-yr during training: first 60 iters 2.00 → last 60 iters 1.29 (training-time rate with exploration; not an evaluation).
- last-iteration causes: {}; entropy 1.09.
- logs: `runs/stage2/gru_s0/train.jsonl`, checkpoints in `runs/stage2/gru_s0/checkpoints/`.

## 2026-10-26 22:14 — Stage 2 evaluation: gru_s0

Checkpoint `runs/stage2/gru_s0/checkpoints/last.pt`, 5 held-out seeds × 2.0 yr, 256 agents, sampled actions. Same protocol and statistics as `run_baselines.py`.

| policy | deaths / agent-yr (95% CI) | Hedges' g (learner − baseline) |
|---|---|---|
| **learner (gru_s0)** | **1.80 [1.52, 2.10]** | |
| random | 3.83 [3.53, 4.19] | -4.58 |
| greedy | 3.31 [2.92, 3.77] | -2.97 |
| heuristic | 1.50 [1.04, 2.11] | +0.50 |

KM survival at 1 yr 0.062 [0.022, 0.111]; deaths by cause (summed) {'starvation': 3637, 'thirst': 231, 'cold': 700, 'fire': 17, 'predator': 15, 'old_age': 3}.
Fully recorded: `runs/stage2_eval/gru_s0/s*/trajectory/`.

## 2026-10-26 — Stage 2: first learner vs baselines, interpretation (written by hand)

Against the preregistration (`docs/PREREGISTRATION.md`, committed `4c84dec` before evaluation):
- **H1 (beats random and greedy): supported.** GRU learner 1.80 [1.52, 2.10] vs random 3.83 [3.53, 4.19] and greedy 3.31
  [2.92, 3.77] deaths/agent-yr; CIs don't overlap (g = −4.6, −3.0).
- **H2 (beats heuristic): not supported.** 1.80 [1.52, 2.10] vs 1.50 [1.04, 2.11]: CIs overlap, so the two are not
  distinguishable at n = 5 (g = +0.5, point estimate worse). KM survival at 1 year is 0.062 vs 0.214.
- **The causes of death changed completely.** Learner (5 seeds summed): starvation 3,637, cold 700, thirst 231,
  fire 17, predator 15. Heuristic: thirst 2,460, cold 442, starvation 347, predator 289, fire 276. The learner nearly
  removed the heuristic's main killers (thirst ÷10, predators ÷19, fire ÷16) and now dies mostly of starvation.
- **Diet (seed 0 recordings):** energy intake is the same for both, 0.00133 vs 0.00131 per agent-tick, which is about
  basal metabolism. Both live mostly on plants (0.03 energy per bite). The learner made 0 kills (hunting is 1.1% of its
  actions) and drinks about 15× more often than the heuristic (77 vs 5 per 1,000 agent-ticks; mean hydration 0.87). Reading:
  the learner solved water and danger, then hit the world's energy ceiling, which the heuristic never reached because thirst
  killed its agents first. Why predator deaths fell is **not established**. It could be avoidance, or simply staying near
  water and away from forests.
- **Caveats:** 1 training seed, about 20M agent-steps (small), stochastic actions, diet measured on 1 seed. The training
  curve was still improving (2.00 → 1.29 training-time rate), so H2 may be a budget limit.

## 2026-10-26 22:52 — Stage 2 training: ff_s0

PPO (memoryless), 256 agents, 600 iterations × 128 ticks (19,660,800 agent-steps), training worlds 1000+ (eval worlds 0-4 held out).
- deaths/agent-yr during training: first 60 iters 2.21 → last 60 iters 1.14 (training-time rate with exploration; not an evaluation).
- last-iteration causes: {}; entropy 1.00.
- logs: `runs/stage2/ff_s0/train.jsonl`, checkpoints in `runs/stage2/ff_s0/checkpoints/`.

## 2026-10-26 22:55 — Stage 2 evaluation: ff_s0

Checkpoint `runs/stage2/ff_s0/checkpoints/last.pt`, 5 held-out seeds × 2.0 yr, 256 agents, sampled actions. Same protocol and statistics as `run_baselines.py`.

| policy | deaths / agent-yr (95% CI) | Hedges' g (learner − baseline) |
|---|---|---|
| **learner (ff_s0)** | **1.58 [1.35, 1.83]** | |
| random | 3.83 [3.53, 4.19] | -5.54 |
| greedy | 3.31 [2.92, 3.77] | -3.63 |
| heuristic | 1.50 [1.04, 2.11] | +0.15 |

KM survival at 1 yr 0.101 [0.049, 0.167]; deaths by cause (summed) {'starvation': 2891, 'thirst': 618, 'cold': 483, 'fire': 27, 'predator': 26, 'old_age': 1}.
Fully recorded: `runs/stage2_eval/ff_s0/s*/trajectory/`.

## 2026-10-26 — Stage 2: memory ablation (H3), interpretation (written by hand)

Memoryless learner `ff_s0` (same budget): **1.58 [1.35, 1.83]** deaths/agent-yr vs GRU 1.80 [1.52, 2.10]. Memoryless is
better on **all 5** held-out worlds (all-cause g = +0.59 for GRU−FF). Thirst rate is GRU 0.09 [0.04, 0.14] vs FF 0.24
[0.11, 0.36] (g = −1.15; the CIs touch, so not distinguishable at n = 5 by the preregistered rule), but starvation is
GRU 1.42 vs FF 1.13. **H3 is mixed:** memory shifted deaths from thirst to starvation rather than reducing them.
Both learners are about 2× better than random and greedy, and not distinguishable from the heuristic. The binding constraint for every
competent policy is now the energy economy (intake ≈ basal metabolism, plants only, no hunting). Training-time
mortality curves are dominated by the world-switch and season cycles, so held-out evaluation during training is needed. Full write-up:
`reports/stage2/STAGE2_REPORT.md`.

## 2026-10-26 — Baselines reproduced on a second machine

Re-ran `experiments/run_baselines.py --seeds 5 --years 2` (with the spin-up fix) on the MacBook Air M4.
All 15 runs (3 baselines × 5 held-out seeds) match the numbers committed from the MacBook Pro M3 Pro
bit-for-bit (random 3.83, greedy 3.31, heuristic 1.50 deaths/agent-year). The simulator is
reproducible across machines; learner training on MPS is not (see reports/stage2/STAGE2_REPORT.md).

## 2026-10-27 00:22 — Stage 2 training: ff_s0

PPOTrainer (memoryless), 256 agents, 600 iterations × 128 ticks (19,660,800 agent-steps), training worlds 1000+ (eval worlds 0-4 held out).
- deaths/agent-yr during training: first 60 iters 2.04 → last 60 iters 1.07 (training-time rate with exploration; not an evaluation).
- last-iteration causes: {'starvation': 1}; entropy 1.37.
- logs: `runs/stage2/ff_s0/train.jsonl`, checkpoints in `runs/stage2/ff_s0/checkpoints/`.

## 2026-10-27 00:24 — Energy economy: can hunting or fruit pay?

Validation worlds [500, 501, 502] (not training, not test), 2 years, 256 agents. Hand-written foragers that differ only in what a hungry agent seeks; nothing in the world was changed.

| policy | deaths / agent-yr (95% CI) | energy eaten / agent-yr (plants, fruit, meat) | small kills / agent-yr | starvation share |
|---|---|---|---|---|
| heuristic | 1.24 [0.84, 1.52] | 5.25 (3.23, 1.05, 0.97) | 2.76 | 7% |
| heuristic:grazer | 3.12 [2.91, 3.54] | 3.89 (3.29, 0.55, 0.05) | 0.00 | 10% |
| heuristic:fruit | 2.58 [2.14, 2.81] | 4.27 (2.76, 1.44, 0.08) | 0.00 | 9% |
| heuristic:hunter | 1.33 [1.23, 1.49] | 5.21 (3.42, 0.49, 1.30) | 3.90 | 5% |
Basal metabolism alone costs 5.0 energy per agent-year (before movement).

## 2026-10-27 00:55 — Stage 2 evaluation: ff_s0

Checkpoint `runs/stage2/ff_s0/checkpoints/last.pt`, 5 held-out seeds × 2.0 yr, 256 agents, sampled actions. Same protocol and statistics as `run_baselines.py`.

| policy | deaths / agent-yr (95% CI) | Hedges' g (learner − baseline) |
|---|---|---|
| **learner (ff_s0)** | **1.50 [1.30, 1.75]** | |
| random | 3.83 [3.53, 4.19] | -5.71 |
| greedy | 3.31 [2.92, 3.77] | -3.78 |
| heuristic | 1.50 [1.04, 2.11] | +0.01 |

KM survival at 1 yr 0.100 [0.055, 0.148]; deaths by cause (summed) {'starvation': 2918, 'thirst': 373, 'cold': 531, 'fire': 18, 'old_age': 3, 'predator': 7}.
Fully recorded: `runs/stage2_eval/ff_s0/s*/trajectory/`.

## 2026-10-27 01:22 — Stage 2 training: gru_s0

PPOTrainer (GRU), 256 agents, 600 iterations × 128 ticks (19,660,800 agent-steps), training worlds 1000+ (eval worlds 0-4 held out).
- deaths/agent-yr during training: first 60 iters 2.09 → last 60 iters 1.26 (training-time rate with exploration; not an evaluation).
- last-iteration causes: {'starvation': 1}; entropy 1.11.
- logs: `runs/stage2/gru_s0/train.jsonl`, checkpoints in `runs/stage2/gru_s0/checkpoints/`.

## 2026-10-27 01:27 — Stage 2 evaluation: gru_s0

Checkpoint `runs/stage2/gru_s0/checkpoints/last.pt`, 5 held-out seeds × 2.0 yr, 256 agents, sampled actions. Same protocol and statistics as `run_baselines.py`.

| policy | deaths / agent-yr (95% CI) | Hedges' g (learner − baseline) |
|---|---|---|
| **learner (gru_s0)** | **1.67 [1.45, 1.92]** | |
| random | 3.83 [3.53, 4.19] | -5.23 |
| greedy | 3.31 [2.92, 3.77] | -3.40 |
| heuristic | 1.50 [1.04, 2.11] | +0.31 |

KM survival at 1 yr 0.055 [0.023, 0.088]; deaths by cause (summed) {'starvation': 3310, 'thirst': 278, 'cold': 662, 'fire': 12, 'predator': 20, 'old_age': 1}.
Fully recorded: `runs/stage2_eval/gru_s0/s*/trajectory/`.

## 2026-10-27 01:32 — Energy economy: can hunting or fruit pay?

Validation worlds [500, 501, 502] (not training, not test), 2 years, 256 agents. Hand-written foragers that differ only in what a hungry agent seeks; nothing in the world was changed.

| policy | deaths / agent-yr (95% CI) | energy eaten / agent-yr (plants, fruit, meat) | small kills / agent-yr | starvation share |
|---|---|---|---|---|
| heuristic | 1.24 [0.84, 1.52] | 5.25 (3.23, 1.05, 0.97) | 2.76 | 7% |
| heuristic:grazer | 3.12 [2.91, 3.54] | 3.89 (3.29, 0.55, 0.05) | 0.00 | 10% |
| heuristic:fruit | 2.58 [2.14, 2.81] | 4.27 (2.76, 1.44, 0.08) | 0.00 | 9% |
| heuristic:hunter | 1.33 [1.23, 1.49] | 5.21 (3.42, 0.49, 1.30) | 3.90 | 5% |
| ppo:ff_s0 | 1.36 [1.21, 1.45] | 6.44 (4.66, 1.49, 0.30) | 0.00 | 75% |
| ppo:gru_s0 | 1.48 [1.29, 1.64] | 6.66 (4.86, 1.56, 0.24) | 0.03 | 77% |
Basal metabolism alone costs 5.0 energy per agent-year (before movement).

## 2026-10-26 23:04 — Live viewer session (heuristic, seed 0)

Viewer run `runs/viewer/20261026_210323_s0_heuristic`: 67391 ticks (17.55 sim years), 256 agents, policy **heuristic** (non-learning baseline).
- deaths/agent-year 1.88; deaths by cause {'starvation': 520, 'thirst': 3473, 'cold': 616, 'fire': 150, 'predator': 3659, 'old_age': 10}; population at end 256.
- manual interventions: 0 (none). Sessions with interventions are not comparable to `run_baselines.py` numbers.
- eating food another agent killed/dropped: 1547 meals (a co-feeding count, not evidence of intentional sharing).

## 2026-10-26 23:18 — Stage 2 training: smoke_eval

PPO (GRU), 256 agents, 3 iterations × 128 ticks (98,304 agent-steps), training worlds 1000+ (eval worlds 0-4 held out).
- deaths/agent-yr during training: first 1 iters 0.00 → last 1 iters 0.00 (training-time rate with exploration; not an evaluation).
- last-iteration causes: {}; entropy 2.39.
- logs: `runs/stage2/smoke_eval/train.jsonl`, checkpoints in `runs/stage2/smoke_eval/checkpoints/`.

## 2026-10-26 23:18 — Live viewer session (rppo:ff_s0, seed 0)

Viewer run `runs/viewer/20261026_230454_s0_rppo:ff_s0`: 8979 ticks (2.34 sim years), 256 agents, policy **rppo:ff_s0** (learned PPO policy).
- deaths/agent-year 1.53; deaths by cause {'starvation': 736, 'thirst': 158, 'cold': 6, 'fire': 6, 'predator': 7}; population at end 256.
- manual interventions: 0 (none). Sessions with interventions are not comparable to `run_baselines.py` numbers.
- eating food another agent killed/dropped: 13 meals (a co-feeding count, not evidence of intentional sharing).

## 2026-10-26 23:39 — Live viewer session (rppo:ff_s0, seed 0)

Viewer run `runs/viewer/20261026_233724_s0_rppo:ff_s0`: 1352 ticks (0.35 sim years), 256 agents, policy **rppo:ff_s0** (learned PPO policy).
- deaths/agent-year 0.04; deaths by cause {'starvation': 3, 'predator': 1}; population at end 256.
- manual interventions: 0 (none). Sessions with interventions are not comparable to `run_baselines.py` numbers.
- eating food another agent killed/dropped: 1 meals (a co-feeding count, not evidence of intentional sharing).

## 2026-10-27 14:43 — Stage 1 baseline survival

Baselines, 2 seeds x 0.5 years, 256 agents, fixed population (respawn).

| policy | deaths / agent-year (95% CI) | Kaplan-Meier survival at 1 yr (95% CI) |
|---|---|---|
| random | 2.39 [2.19, 2.60] | 0.153 [0.089, 0.218] |
| greedy | 1.09 [0.76, 1.41] | 0.568 [0.419, 0.717] |
| heuristic | 1.22 [1.09, 1.34] | 0.586 [0.543, 0.630] |

Deaths by cause (summed over seeds): **random** {'starvation': 254, 'thirst': 332, 'fire': 4, 'predator': 23}; **greedy** {'starvation': 80, 'thirst': 163, 'fire': 13, 'predator': 22}; **heuristic** {'thirst': 257, 'fire': 5, 'starvation': 2, 'predator': 48}

## 2026-10-27 14:43 — Stage 2 evaluation: short_gru_s0

Checkpoint `runs/stage2/gru_s0/checkpoints/last.pt`, 2 held-out seeds × 0.5 yr, 256 agents, sampled actions. Same protocol and statistics as `run_baselines.py`.

| policy | deaths / agent-yr (95% CI) | Hedges' g (learner − baseline) |
|---|---|---|
| **learner (short_gru_s0)** | **0.19 [0.08, 0.30]** | |
| random | 2.39 [2.19, 2.60] | -5.34 |
| greedy | 1.09 [0.76, 1.41] | -1.47 |
| heuristic | 1.22 [1.09, 1.34] | -3.48 |

KM survival at 1 yr 0.905 [0.849, 0.961]; deaths by cause (summed) {'starvation': 27, 'thirst': 15, 'fire': 2, 'predator': 5}.
Fully recorded: `runs/stage2_eval/short_gru_s0/s*/trajectory/`.

## 2026-10-27 14:44 — Stage 2 evaluation: short_ff_s0

Checkpoint `runs/stage2/ff_s0/checkpoints/last.pt`, 2 held-out seeds × 0.5 yr, 256 agents, sampled actions. Same protocol and statistics as `run_baselines.py`.

| policy | deaths / agent-yr (95% CI) | Hedges' g (learner − baseline) |
|---|---|---|
| **learner (short_ff_s0)** | **0.33 [0.33, 0.34]** | |
| random | 2.39 [2.19, 2.60] | -5.69 |
| greedy | 1.09 [0.76, 1.41] | -1.31 |
| heuristic | 1.22 [1.09, 1.34] | -4.05 |

KM survival at 1 yr 0.837 [0.837, 0.837]; deaths by cause (summed) {'starvation': 49, 'thirst': 29, 'predator': 7}.
Fully recorded: `runs/stage2_eval/short_ff_s0/s*/trajectory/`.

## 2026-10-27 15:25 — Exploratory analyses for the preprint (reports/stage2/pilot_preprint/), pilot runs gru_s0 / ff_s0

Not preregistered (labelled exploratory in the paper). All numbers in `reports/stage2/paper/stats.json`, script `make_figures.py`.
The two `short_*` evaluations above (0.5 yr, worlds 0-1) were run for this; the "Stage 1 baseline survival" entry just above is the
matched 0.5-yr baseline run (NOT the canonical 2-yr baselines; `reports/stage1/baselines.json` was restored from git).
- **Horizon reverses the ranking.** Spring-summer (0.5 yr): GRU 0.19, memoryless 0.33 vs heuristic 1.22 deaths/agent-yr. Two years: 1.80 / 1.58 vs 1.50.
  Half-year evaluations (the in-training mini-evals) must not be used for claims.
- **Winter deficit.** Deaths/agent-yr by season (Sp/Su/Au/Wi): GRU 0.76/0.38/1.93/4.09; memoryless 0.78/0.73/1.71/3.03; heuristic 2.42/0.50/0.83/2.16.
  Winter mean energy of living agents: learners 0.35-0.37, heuristic 0.53.
- **Energy budget** (eat-event energy, 9.8M agent-ticks/arm): intake 1.31e-3 (GRU) / 1.34e-3 (memoryless) per agent-tick ≈ basal 1.3e-3.
  Plants 73-74% (mean 0.0094 energy per bite, a third of nominal 0.03), fruit 23-25%, meat 2-3% (mostly scavenged: GRU 25 small kills, 0 large kills
  in 2,560 agent-yr). First-order expenditure: GRU ≥1.52e-3 (moves 37%, rests 16%), memoryless ≥1.34e-3 (moves 24%, rests 28%; 48% rest in winter).
- **Drought inflexibility.** DRINK with no water in reach: 78% (GRU) / 91% (memoryless) during droughts vs 30% / 37% otherwise.
- Negative results unchanged: H2 not supported, H3 mixed. No social behaviour measured.

## 2026-10-27 — H6/H7 analyses applied to PILOT checkpoints (gru_s0, ff_s0); preprint §6.7

Preregistered for the replication; run here on the pilots to test the pipeline (replication stays confirmatory).
Scripts: `reports/stage2/pilot_preprint/{analysis_h6_h7,aggregation_strict,blind_ablation}.py`; numbers: `stats_h6_h7.json`.
- **Replay check:** re-running the networks on bit-exact replayed observations reproduces recorded values (corr 1.000, max |Δ| 1.3e-6).
- **H6 probes (world 0, 1 yr, split by agent):** GRU h: hydration R² 0.70, energy 0.85, predator AUC 0.91 → thresholds met. Memoryless core
  0.91/0.91/0.78. Caveat: hydration/energy are direct inputs (pre-core encodings decode them better, 0.81-0.94); predators visible on 0.2-0.4% of samples.
- **Memory test (exploratory), NEGATIVE:** log distance to nearest water when none in view: GRU h R² 0.32 < its pre-core encoding 0.36 ≈ memoryless 0.33.
  No evidence the GRU state stores water location.
- **Value calibration (exploratory):** AUC of low value for death within 200 ticks: GRU 0.98, memoryless 0.95 (lowest decile 71-78% vs base 9%).
- **Cooperative hunts (H7a):** GRU 14 large-prey attacks, 1 multi-attacker episode vs null 0.34 (p 0.34); memoryless 7 / 0; 0 large kills each.
  Baselines (no social rule) beat the shuffled-timing null (random p 0.004, heuristic p 0.04) → timing-only null is ANTI-CONSERVATIVE; use a
  co-location-preserving, baseline-calibrated null for the replication.
- **Aggregation (H7b), clause NOT supported but not social:** Clark-Evans R obs/null (water-matched) GRU 0.77, memoryless 0.82 vs baselines 0.95-0.98;
  water×terrain×fertility null 0.81 / 0.86 vs 0.97-0.98. Social-blinding ablation (agent channel + neighbour features zeroed, worlds 0-1, 1 yr):
  clustering INCREASES (0.77→0.71, 0.83→0.78), mortality unchanged (1.74→1.73, 1.49→1.52). Clustering is not mediated by perceiving others.
  Not counted as emergent social structure.
