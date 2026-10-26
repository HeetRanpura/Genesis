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
