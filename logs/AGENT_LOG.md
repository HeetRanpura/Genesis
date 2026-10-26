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
