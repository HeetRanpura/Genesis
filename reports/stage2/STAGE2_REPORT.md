# Stage 2 report: first learning agents (2026-10-26)

> **Which code:** the results tables below were first produced by `agents/rppo.py`, an implementation developed in
> parallel with `agents/ppo.py`. On 2026-10-26 the two were merged: PPOTrainer is now the only learner, with rppo's
> configuration as its defaults. The original evaluation files are in `reports/stage2/rppo_original/`, and the
> **Reproduction** section at the end re-runs the same protocol with PPOTrainer.

**Bottom line:** recurrent and memoryless PPO agents, rewarded only for their own homeostasis, learned in about
20M agent-steps to survive on unseen worlds about 2× better than random and greedy baselines. They are **not yet
distinguishable** from the hand-written heuristic. They solved thirst, predators and fire, and now hit the world's
**energy ceiling**, dying of starvation. At this budget, adding memory (GRU) **did not** improve overall survival. It cut
thirst deaths (large effect, CIs touching) but increased starvation. Hypotheses were preregistered
(`docs/PREREGISTRATION.md`, commit `4c84dec`) before any evaluation.

## Setup
- **Learner** (then `agents/rppo.py`, now `agents/policy.py` + `agents/ppo.py`): one parameter-shared actor-critic for all agents. The input is the 11×11×16 view plus the agent's
  own body, its neighbours and heard signals. Layers: conv(32) → conv(64, stride 2) → fc 256, plus an fc 64 for the vector inputs, then fc 256 →
  **GRU 256** (or a feed-forward layer in the memoryless ablation) → policy and value heads, about 1.1M parameters. Each agent has its own
  memory state, reset when a new agent fills a slot.
- **Training** (then `experiments/train_rppo.py`, now `experiments/train_ppo.py`): PPO with γ 0.995, λ 0.95, clip 0.2, lr 3e-4, entropy 0.01, 4 epochs, 128-tick
  rollouts × 256 agents, 600 iterations (**19.7M agent-steps**, about 50 min each on an M3 Pro, MPS).
  - Fixed population, all capabilities off, reward = own drive reduction + death penalty only.
  - A new training world every 40 iterations (seeds 1000+). The evaluation worlds (0–4) are never seen in training.
  - Every training rollout is flight-recorded, including value, action probabilities and log-prob (about 2 GB per run).
- **Evaluation** (`experiments/eval_policy.py`): identical to the baselines, with seeds 0–4, 2 years, 256 agents and sampled actions.
  It reports deaths/agent-year, KM survival, and bootstrap CIs over seeds. All runs are fully recorded.

## Results (held-out worlds, 5 seeds × 2 years)
| policy | deaths / agent-yr (95% CI) | KM survival 1 yr | g vs heuristic |
|---|---|---|---|
| random | 3.83 [3.53, 4.19] | 0.001 | |
| greedy | 3.31 [2.92, 3.77] | 0.006 | |
| heuristic | 1.50 [1.04, 2.11] | 0.214 | |
| **GRU learner** (`gru_s0`) | **1.80 [1.52, 2.10]** | 0.062 | +0.50 |
| **memoryless learner** (`ff_s0`) | **1.58 [1.35, 1.83]** | 0.101 | +0.15 |

Death rate by cause, per agent-year (95% CI):
| cause | GRU | memoryless | heuristic (counts, 5 seeds) |
|---|---|---|---|
| thirst | **0.09 [0.04, 0.14]** | 0.24 [0.11, 0.36] | 2,460 deaths (dominant) |
| starvation | 1.42 [1.17, 1.62] | 1.13 [0.96, 1.28] | 347 |
| cold | 0.27 [0.10, 0.44] | 0.19 [0.07, 0.31] | 442 |
| predator | 0.006 | 0.010 | 289 |

## Hypotheses (preregistered)
- **H1** (beats random and greedy): **supported** for both learners. The CIs don't overlap (g from −3.0 to −5.5).
- **H2** (beats heuristic): **not supported.** The CIs overlap and both point estimates are slightly worse.
- **H3** (memory helps with water): **mixed.**
  - Thirst deaths are 2.7× lower with GRU (g = −1.15), but the CIs touch, so by our rule this is not distinguishable at n = 5.
  - Overall mortality is *not* lower: memoryless did better on all 5 worlds (all-cause g = +0.59 for GRU−FF).
  - GRU agents traded thirst deaths for starvation.
- **H4** (no social behaviour): not yet measured with the Stage 3 metrics. Nothing in the diet analysis suggests sharing.
  Learners ate meat almost never (0.04% of energy) and made 0 small-prey kills on seed 0.

## What the learners do (seed 0 recordings)
- Actions: eat 21%, rest 16%, drink 15%, moves 37%, hunt 1.1%, pickup and drop 7%. Mean hydration while alive is 0.87, against roughly 0.6 for the heuristic.
- Energy intake is about 0.00133 per agent-tick for both learner and heuristic, which roughly equals basal metabolism. Almost all of it comes from plants, at 0.03 energy per bite.
  **Energy is the binding constraint**: once water and danger are handled, agents starve slowly, especially through winter.
- Predator and fire deaths fell about 20-fold compared with the heuristic. The mechanism (active avoidance, or simply staying in open land
  near water) is **not yet established**.

## Learning curves (`fig_training.png`)
Policy entropy falls steadily from 2.40 (uniform) to about 1.1, which is the clearest sign of learning. Training-time mortality oscillates
on a ~40-iteration cycle, which matches the world switch every 40 iterations plus the seasons. **It is not a usable learning curve.**
A fixed-world evaluation every N iterations is needed (next step).

## Limitations
- One training seed per learner, so the variance between training runs is unmeasured. 20M agent-steps is small.
- The memory comparison is at an equal step budget. Recurrent nets often need longer to pay off, so H3 may be budget-limited.
- MPS (GPU) arithmetic is not guaranteed bit-identical. Seeds, sampling and the CPU path are deterministic and tested.
- Actions are sampled, not greedy. Greedy evaluation has not been run.

## Next steps (proposed)
1. **Periodic held-out evaluation during training**, and longer runs (100M+ agent-steps) with 3 training seeds per arm. This decides H2 and H3.
2. **Energy economy:** measure whether hunting or fruit could pay. If the world simply doesn't allow a surplus, report it. Don't tune
   the world for the learner.
3. Stage 3 social metrics on the learner recordings (co-feeding, proximity, transfer of food), with baseline nulls.

Reproduce: `make train NAME=gru_s0`, `make train NAME=ff_s0 MEMORYLESS=1`, then `make eval NAME=...`, then
`python experiments/plot_training.py gru_s0 ff_s0`.

## Replication with the unified learner (2026-10-27)
After `agents/rppo.py` was folded into PPOTrainer, both arms were retrained from scratch with the same protocol (seed
0, 600 iterations × 128 ticks × 256 agents = 19.7M agent-steps, training worlds 1000+, MPS) and evaluated on the same
held-out worlds 0–4 (`reports/stage2/eval_{gru,ff}_s0.json`; originals in `rppo_original/`).

| learner | original (rppo) | replication (PPOTrainer) | g vs heuristic (replication) |
|---|---|---|---|
| memoryless | 1.58 [1.35, 1.83] | **1.50 [1.30, 1.75]** | +0.01 |
| GRU | 1.80 [1.52, 2.10] | **1.67 [1.45, 1.92]** | +0.31 |

All CIs overlap. The conclusions repeat:
- **H1:** supported.
- **H2:** not supported. The learners are not distinguishable from the heuristic (1.50 [1.04, 2.11]).
- **H3:** not supported. Memory cuts thirst deaths (6% vs 10% of deaths) but does not lower overall mortality.
- **Causes of death:** starvation is 76–77%.

## Energy economy and behavioural anatomy (2026-10-27)
- **Energy economy** (`experiments/energy_economy.py`, validation worlds 500–502). Hand-written foragers that differ
  only in diet show that **plants alone cannot cover basal metabolism**: a grazer eats 3.9 energy per agent-year
  against 5.0 basal, and dies at 3.12 deaths per agent-year. Hunting closes the gap (the hunter dies at 1.33, the mixed
  heuristic at 1.24). Learners eat more over the year (6.4–6.7), but in the wrong seasons: about 2× the heuristic in
  spring and summer, much of it wasted at the energy cap, and less than a fifth of it in winter (0.12–0.15 vs 0.71),
  because they take almost no meat. Their diet is a local optimum; the world does pay for hunting.
- **Behaviour** (`experiments/behaviour_analysis.py`, every agent and tick of test world 0):
  - **Homeostatic control:** graded. P(drink) is 0.52 when parched and 0.09 when hydrated; eating is U-shaped.
  - **Space use:** riverside sedentism. Learners stay a median 2.0–2.2 cells from water (heuristic 3.6) in 14–18-cell
    home ranges (heuristic 38).
  - **Daily rhythm:** none. Learners rest no more at night than at midday.
  - **Value estimate:** tracks the drive, r = −0.65 (memoryless) and −0.75 (GRU).
  - **Dawn:** the GRU policy sharpens (entropy 1.43 → 1.23).
  - **Behavioural types:** no distinct types (silhouette 0.34–0.41).
  - **Apparent clustering:** fully explained by shared place preference (NN ratio 0.84 against a habitat null,
    0.98–1.01 against a place-preference null).
  - **Social behaviour:** no group kills, and food transfer of under 0.01 energy per agent-year. **H4 holds.**
- **Paper:** all of this is written up in `paper/main.pdf` (source `paper/main.tex`; figures via
  `experiments/paper_figures.py` and `experiments/behaviour_analysis.py`).
