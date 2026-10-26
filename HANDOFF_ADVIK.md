# Handoff for Advik (2026-10-27)

Hi Advik, this is where GENESIS stands after the last session on Heet's machine, and what is most useful to pick up
next. The general handoff is `HANDOFF.md`; the rules and gotchas are in `CLAUDE.md`.

## 1. Sync first
`main` on `HeetRanpura/Genesis` has four commits after your PR #3 (`b066ee9`):

| commit | what |
|---|---|
| `a3f206c` | Validation curve during training (`--val-every`, worlds 500–502), heuristic diet probes, `experiments/energy_economy.py` |
| `0c89250` | Replication with the unified learner, energy-economy results, behavioural analysis, **the paper** (`paper/`) |
| `9e79ea6` | Launch config points the viewer at `ppo:gru_s0` |
| (this commit) | This handoff; `viewer/preview_server.py` loads current checkpoints |

Your PR #3 was already based on the learner unification (`79d713a`), so a plain rebase of your fork onto `main` is
enough (assuming your remote for `HeetRanpura/Genesis` is called `upstream`):

```bash
git fetch upstream && git rebase upstream/main
```

Then run `make setup` (or at least `uv pip install -e ".[dev,viewer,learn]"`) and `make test`. **59 tests should pass.**

## 2. What changed in your code (please skim)
- **`agents/rppo.py` is gone; its design lives on as the defaults of `agents/ppo.py` (PPOTrainer).**
  - Your network is now the only `ActorCritic` in `agents/policy.py`, with the same layers and orthogonal init.
  - PPOConfig defaults are your settings: 128-tick rollouts, full BPTT, 4×4 minibatches split by agent, raw reward,
    and a value loss of 0.5·(V−R)²·0.5.
  - Truncated BPTT and return normalisation remain options.
- **Training:** `experiments/train_ppo.py` replaces `train_rppo.py`. It keeps your world schedule (1000+100·seed+w,
  a new world every 40 iterations), flight recording with the disk guard, `train.jsonl` and `checkpoints/last.pt`.
  `make train NAME=gru_s0` / `make train NAME=ff_s0 MEMORYLESS=1`.
- **Evaluation:** `experiments/eval_policy.py` is the only evaluation script (`make eval NAME=...`).
  `agents/learned.py` and `eval_agents.py` are gone; use `agents.policy.load_policy_agent(run_or_path, n_slots)`.
- **Checkpoints:** they carry `format: 2`. Older ones are refused with a clear error.
- **Viewer:** lists only `ppo:<run>`.
- **Tests:** yours are ported into `tests/test_ppo.py`, covering determinism on CPU, the memory reset, GAE not
  bootstrapping through a death, the checkpoint round-trip, and the `.last` internals. Also covered: unroll vs.
  stepwise forward, the recorder hook, and repeatable validation.
- **Your original numbers** are kept in `reports/stage2/rppo_original/`. Nothing was overwritten.

## 3. Results since your report
- **Replication** (same protocol, PPOTrainer, M4 Air), in held-out deaths per agent-year. They match yours within CI:

  | learner | replication | your original |
  |---|---|---|
  | memoryless | **1.50 [1.30, 1.75]** | 1.58 |
  | GRU | **1.67 [1.45, 1.92]** | 1.80 |
  | heuristic (for reference) | 1.50 [1.04, 2.11] | — |

  H1 is supported, H2 and H3 are not, and H4 holds.
- **Energy economy** (`reports/stage2/energy_economy.json`, validation worlds):
  - **Plants alone are below basal metabolism:** a grazer eats 3.9/yr against 5.0 basal and dies at 3.12/yr.
  - **Hunting breaks even:** the hunter eats 5.2/yr (1.3 of it meat) and dies at 1.33/yr.
  - **Learners eat the wrong food at the wrong time:** they eat about 2× in spring and summer (much of it wasted at
    the energy cap) and under a fifth of the heuristic's intake in winter, because they never hunt.
  - **So the plateau is a learning problem, not a world limit.**
- **Behaviour** (`paper/stats.json`, `experiments/behaviour_analysis.py`, test world 0):
  - **Drinking:** graded with thirst, P(drink) 0.52 when parched vs 0.09 when hydrated.
  - **Space use:** riverside sedentism, a median 2.0–2.2 cells from water and 14–18-cell home ranges.
  - **Daily rhythm:** no night resting.
  - **Value estimate:** tracks the homeostatic drive (r −0.65 / −0.75).
  - **Dawn:** the GRU policy sharpens at dawn.
  - **Behavioural types:** none distinct.
  - **Clustering:** apparent clustering vanishes under a place-preference null.
  - **Social behaviour:** zero group kills, and food transfer under 0.01 energy per agent-year.
- **Paper:** `paper/main.pdf` (source `paper/main.tex`, TikZ diagrams in `paper/diagrams.tex`). Build:
  `tectonic -X compile paper/main.tex` (`brew install tectonic`). Every number comes from files in `reports/` or
  `paper/stats.json`. You are listed as second author; tell Heet if you want anything changed.

## 4. Next tasks, in priority order
Your M3 Pro trained 20M steps in about 50 min per arm (your report); the Air took 1.0–1.7 h with two runs sharing it,
and it is fanless and low on disk. So the long runs suit your machine best.

### Task 1: longer, multi-seed training (decides H2 and H3)
- **Runs:** 2 arms (GRU, memoryless) × 3 learner seeds (0, 1, 2) × **3,000 iterations (~98M agent-steps)**, with
  `--val-every 50`. On your machine that is about 4 h per run, so run two at a time.
- **Commands** (one per run):

  ```bash
  python experiments/train_ppo.py --name gru_s1 --seed 1 --iters 3000 --val-every 50 --record-every 20
  python experiments/train_ppo.py --name ff_s1 --seed 1 --iters 3000 --val-every 50 --record-every 20 --core mlp
  ```

  Recording every rollout costs about 2 GB per 20M steps, so roughly 10 GB per 100M-step run; `--record-every 20`
  cuts that to about 0.5 GB. The disk guard stops recording below 15 GB free anyway.
- **Preregister first:** append the predictions for H2/H3 at 100M steps to `docs/PREREGISTRATION.md` and commit
  that before evaluating. Never edit earlier entries.
- **Evaluate** each run with `make eval NAME=...`.
- **Report** mean ± CI across the 3 seeds per arm (seed-to-seed variance is the missing piece). Update Table 1
  and Fig. 5 in the paper and `reports/stage2/STAGE2_REPORT.md`.
- **Acceptance:** the validation curves (`val.jsonl`) are plotted (`experiments/plot_training.py` draws them),
  and 3-seed CIs are reported for both arms.

### Task 2: is hunting an exploration problem?
- **Probe A (world unchanged):** start agents hungry next to small prey for part of training (a curriculum flag in
  `train_ppo.py`, default off), then evaluate on the normal test worlds.
- **Probe B:** an intrinsic bonus would break HARD RULE 1, so if you try it, do it only as an explicitly labelled
  ablation in its own config, and never as a default. Prefer Probe A.
- **Probe C:** an experiment config with denser small prey (`configs/experiments/dense_prey.yaml`, **not** the
  default) to see whether hunting appears when prey are easy to encounter.
- **Acceptance:** small-prey kills per agent-year for the learners (currently about 0.0–0.03) under each probe, with
  baselines, written up whether or not hunting appears.

### Task 3: Stage 3 social metrics (`analysis/social.py`)
- **Metrics:** start from the flight recordings. Co-feeding (two or more agents eating the same carcass within k
  ticks), food transfer (`EV_EAT_MEAT` / `EV_PICKUP_FOOD` events whose owner field, column 3, is neither 0 nor the
  eater; the paper's §6 numbers come from exactly this count over `runs/stage2_eval/*/s*/events.npy`), proximity
  networks, and Ripley's K with the place-preference null used in `behaviour_analysis.py`.
- **Acceptance** (PLAN.md Stage 3): each metric is unit-tested on synthetic data with a known answer, and the
  baselines produce the expected nulls.

### Smaller items
- **Paper references:** the 24 citations were written from memory and not checked online. Verify them (authors,
  venues, years) before any submission.
- **Behavioural analysis on all 5 test worlds:** `behaviour_analysis.py --seed k` for k = 1..4 (needs recordings,
  so re-run `make eval` with the recorder on, or keep the existing `runs/stage2_eval/*/s*`), then aggregate.
- **Viewer frame rate on an idle machine:** your PR #3 left it unmeasured; the GPU was busy then.
- **512² throughput regression from Stage 1.1** (77k vs 111k steps/min, dynamics only): the likely causes are the
  daily full `img_rebuild` (only the water channel changes) and the vegetation pass at 4× the cells.

## 5. Rules (unchanged, please keep)
- Reward = own homeostasis only. Capabilities are off by default. Every run is seeded. No scripted agent behaviour.
  Baselines go in every comparison. Claims come from the held-out test worlds 0–4 only. Report negative results.
- Commits carry **no co-author or AI attribution lines**. PRs go from your fork to `HeetRanpura/Genesis` `main`;
  `make test` must pass.
- On laptops, run heavy jobs one or two at a time (the Air is fanless and throttles; your M3 Pro copes better).

## 6. Where things are
`sim/` engine · `agents/policy.py` network + loader · `agents/ppo.py` trainer · `agents/baselines.py` baselines + diet
probes · `experiments/` (`train_ppo`, `eval_policy`, `energy_economy`, `behaviour_analysis`, `paper_figures`,
`plot_training`, `run_baselines`) · `viewer/` (3-D viewer `server.py` + `static/`; quick 2-D preview
`preview_server.py`) · `reports/stage1|stage2/` · `paper/` · `docs/` (WORLD, AGENTS, VIEWER, PREREGISTRATION) ·
`logs/` (AGENT_LOG, SESSION_LOG).
Local only (git-ignored) on Heet's machine: `runs/stage2/{gru_s0,ff_s0}` (checkpoints), `runs/stage2_eval` (2.2 GB
of recordings), `runs/stage1_baselines` (2.0 GB). Ask Heet if you need the checkpoints. They are not in git, and
retraining takes about 1 h per arm on your machine.
