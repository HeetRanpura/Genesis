# Preregistered hypotheses

Hypotheses are written here **before** the runs that test them, with date and time. Results go to `logs/AGENT_LOG.md`
and reports; entries here are never edited after the fact (corrections are appended).

## Stage 2, first slice: recurrent PPO vs baselines (written 2026-10-26 21:20 IST, before any evaluation)

**Setup.** Parameter-shared actor-critic, survival reward only. 256 agents, fixed population, all capabilities off.
It trains on worlds 1000+ for 600 iterations × 128 ticks (about 19.7M agent-steps, about 50 min on the M3 Pro).
Evaluation uses held-out seeds 0–4 × 2 years, exactly like `run_baselines.py`. Two learners: `gru_s0` (GRU memory) and
`ff_s0` (memoryless ablation, otherwise identical).

- **H1 (learning works).** The GRU learner's deaths/agent-yr on held-out worlds is below random (3.83) and greedy (3.31).
- **H2 (beats the hand-written rules).** The GRU learner is below the heuristic (1.50). *My expectation: uncertain at this
  training budget.* 20M agent-steps is small for a 1.1M-parameter recurrent policy. A failure here is a budget result,
  not a verdict on learning.
- **H3 (memory helps with water).** GRU vs memoryless: GRU has fewer thirst deaths per agent-year, and its overall mortality is lower
  or equal. This is the prediction logged in AGENT_LOG at Stage 1 (greedy dies of thirst because water leaves its view).
- **H4 (null for social behaviour).** No measurable social behaviour is expected at this stage. "Ate others' kill" will be
  co-feeding at the level the environment produces, and there's no reason for agents to drop food near others.
  Any apparent grouping is expected to be explained by shared water and food sources.

**Decision rule.** "Beats" means the 95% bootstrap CIs over the 5 held-out seeds don't overlap, plus a report of Hedges' g.
Overlapping CIs are reported as "not distinguishable at n = 5", never as a win.

**Known limitations, stated in advance:** a single training seed per learner (seed-to-seed variance of training is unmeasured);
GPU (MPS) arithmetic isn't guaranteed bit-identical across runs, although seeds, sampling and the CPU path are deterministic and tested.

### Appended 2026-10-26 (Heet): implementation change, hypotheses unchanged
The learner code that produced the first evaluation (`agents/rppo.py`) was folded into the single learner
`agents/ppo.py` (PPOTrainer), with the same network, rollout length, BPTT, epochs and reward handling as defaults.
H1–H4 and the decision rule above are unchanged. They are re-tested on PPOTrainer runs `gru_s0`/`ff_s0`
(same seeds, 600 iterations) and reported next to the original results; MPS training is not bit-identical, so the
re-run is a replication, not a bitwise reproduction.

## Overnight replication: 3 training seeds × {GRU, memoryless} (written 2026-10-26 23:30 IST, before these runs start)

**Setup.** `agents/rppo.py`, the same hyper-parameters as the first slice, with **900 iterations (about 29.5M agent-steps) per run**.
- Training seeds 1, 2 and 3 for both arms (runs `gru_s1..3`, `ff_s1..3`).
- Held-out mini-evaluation every 100 iterations (worlds 0–1, 0.5 yr).
- Final evaluation on worlds 0–4 × 2 yr (full recording).
- The two first-slice runs (`gru_s0`, `ff_s0`, 600 iterations) are pilots and are **not** pooled into these tests.
- Statistics: the unit of replication is the training seed (n = 3 per arm). Each seed's score is its mean deaths/agent-yr over the 5 evaluation worlds.
  We report the mean and the range across seeds, and Hedges' g. With n = 3, CIs are wide, and that will be stated.

- **H2' (learners vs heuristic).** At 29.5M steps, the learners' mean deaths/agent-yr is ≤ the heuristic's 1.50. *Expectation:
  not distinguishable from the heuristic.*
- **H3' (memory).** GRU has a lower thirst-death rate than memoryless (direction replicates the pilot), and *no* advantage in all-cause
  mortality at this budget.
- **H5 (energy ceiling).** Starvation remains the leading cause of death for both arms, and learners make essentially no
  large-prey kills (≤ 1 per 1,000 agent-years).
- **H6 (internal states, probes).** A linear probe on the GRU hidden state decodes the agent's own hydration and energy
  (R² ≥ 0.5 on held-out ticks) and whether a predator is visible (AUC ≥ 0.7). The same probe on the memoryless trunk is the
  comparison. These are functional correlates only, never claims of feelings.
- **H7 (no social coordination, null).** The rate of large-prey attacks with ≥ 2 attackers is not above a shuffled-timing null.
  Agent spatial aggregation (Clark–Evans R) does not differ from the heuristic baseline's once both are measured relative to
  distance from water.

## Status note on the overnight replication (appended 2026-10-27; the entry above is unchanged)

The overnight runs were stopped by the user at 154/900 (`gru_s1`) and 262/900 (`ff_s1`) iterations, and
`agents/rppo.py` was then folded into PPOTrainer. H2', H3' and H5 therefore remain **untested** as preregistered.
The H6/H7 analyses were applied to the *pilot* checkpoints only as a pipeline test (AGENT_LOG 2026-10-27,
`reports/stage2/pilot_preprint/`); they are not the confirmatory test. The next replication (HANDOFF_ADVIK Task 1)
needs its own preregistration entry before evaluation.

## H2/H3 protocol: 3 training seeds × {GRU, memoryless} × 3,000 iterations (written 2026-10-27, before these runs start)

This entry carries out the overnight replication above (H2', H3', H5 unchanged) on the single learner, and adds a
longer-budget extension. Nothing above is edited. Everything here is implemented by `experiments/run_h2h3.py`.

**Setup.** `agents/ppo.py` (PPOTrainer) with its default hyper-parameters (the rppo design: 128-tick rollouts, full
BPTT, 4 epochs × 4 minibatches, raw rewards), 256 agents, all capabilities off, training worlds 1000 + 100·seed + w.
- Runs `h2h3_gru_s1..3` and `h2h3_ff_s1..3`: learner seeds **1, 2, 3** for both arms, **3,000 iterations each**
  (98.3M agent-steps). The `h2h3_` prefix keeps them apart from the pilot and replication runs.
- Checkpoints kept every 200 iterations and at iteration 900. **Two checkpoints are tested:** iteration **900**
  (29.5M agent-steps, the budget of H2'/H3'/H5 above: the *confirmatory* test) and iteration **3,000** (the extension).
- **Deviation from the entry above, fixed now:** no mini-evaluations on test worlds during training. The learning
  curve uses the validation worlds 500 and 501 (1 yr, every 50 iterations), so test worlds 0–4 are touched only by the
  final evaluations.
- Evaluation: held-out worlds 0–4 × 2 yr, 256 agents, sampled actions (`experiments/eval_policy.py`), the same
  protocol as the baselines. The iteration-3,000 evaluations are fully flight-recorded.
- Unit of replication: the training seed (n = 3 per arm). A seed's score is its mean deaths/agent-yr over the 5 worlds.
  We report the mean, the range, and Hedges' g (against the heuristic's 5 world scores, and GRU vs memoryless).
  With n = 3 nothing here is a significance test, and that will be stated.

**Decision rules** (applied mechanically by `run_h2h3.py`, written to `reports/stage2/h2h3_results.md`):
- **H2' at 900 / H2'' at 3,000 (learners vs heuristic):** supported for an arm if its mean seed score is ≤ the
  heuristic's mean (1.50). Reported per arm. *Expectation at 900: not distinguishable from the heuristic (as above).
  Expectation at 3,000: uncertain. The pilots die of winter starvation because they never hunt; more training may or
  may not find hunting.*
- **H3' at 900 / H3'' at 3,000 (memory):** (a) direction: the GRU arm's mean thirst-death rate is below the
  memoryless arm's; (b) all-cause: the GRU arm has an advantage only if all 3 GRU seed scores are below all 3
  memoryless seed scores (complete separation, the only n = 3 vs 3 pattern with exact one-sided Mann–Whitney p ≤ 0.05).
  *Expectation: (a) holds at both budgets; (b) no advantage at 900; uncertain at 3,000.*
- **H5 at 900 / H5'' at 3,000 (energy ceiling):** supported for an arm if starvation is its leading cause of death and it
  makes ≤ 1 large-prey kill per 1,000 agent-years. *Expectation: supported at 900; uncertain at 3,000.*
- Training-time mortality and validation curves are reported as exploratory, never as tests.
- Every outcome is reported, including null and negative ones. The runs are not restarted or re-seeded because of
  their results. If a run crashes it resumes from its checkpoint, and that is logged.
